import json
import re
from pathlib import Path
from typing import Any


# Narrow set of unambiguous shell-failure signatures. The earlier version had
# regexes like `error:`, `failed`, `not found:`, `permission denied`, which
# matched on:
#   - source code being read (e.g. `sys.exit("ERROR: ...")`)
#   - OpenAPI enum values literally named "Failed"
#   - Claude's plan-approval system text
#   - the agent's own narrative output leaking via injection
# Empirically on real Claude logs those produced ~45% false positives and
# destroyed exactly the failed-then-recovered trajectories that are the most
# valuable agent-SFT signal.
#
# Keep only signatures that are unambiguous evidence of a *command* (not a
# file read or an arbitrary string) failing.
_NO_SUCH_FILE_RX = re.compile(r"no such file or directory", re.IGNORECASE)
_COMMAND_NOT_FOUND_RX = re.compile(r"command not found", re.IGNORECASE)
_EXIT_CODE_RX = re.compile(r"Exit Code:\s*(\d+)", re.IGNORECASE)

# Tools whose nonzero exit codes are normal/expected.
_NONZERO_OK_TOKENS = ("grep", "ps aux |", "lsof -i :", "sleep ", "jobs -l", "curl")


def has_failed_command(content: str) -> bool:
    """True iff the tool result is unambiguously a failed shell command.

    Only fires on signatures that cannot plausibly appear in successful
    output: ``Exit Code: <nonzero>`` (with known-noisy commands excepted),
    ``command not found``, and ``no such file or directory``. Broader
    "error"/"failed" matching was removed — see module-level comment above.
    """
    if not isinstance(content, str):
        return False

    exit_match = _EXIT_CODE_RX.search(content)
    if exit_match and int(exit_match.group(1)) != 0:
        cl = content.lower()
        if not any(token in cl for token in _NONZERO_OK_TOKENS):
            return True

    if _COMMAND_NOT_FOUND_RX.search(content):
        return True
    if _NO_SUCH_FILE_RX.search(content):
        return True
    return False


def remove_orphaned_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip tool_calls entries that have no matching role:tool result."""
    result_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool":
            result_ids.add(msg.get("tool_call_id", ""))

    cleaned = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            kept_calls = [
                tc for tc in msg["tool_calls"] if tc.get("id", "") in result_ids
            ]
            if not kept_calls and not msg.get("content", "").strip():
                continue
            msg = (
                {**msg, "tool_calls": kept_calls}
                if kept_calls
                else {k: v for k, v in msg.items() if k != "tool_calls"}
            )
        cleaned.append(msg)
    return cleaned


def clean_conversation(
    conversation: dict[str, Any],
    remove_orphaned: bool = True,
) -> tuple[dict[str, Any], int]:
    """Clean a conversation by removing structurally-broken messages.

    Drops:
    - Tool results that are unambiguous shell-command failures
      (``has_failed_command``: nonzero exit code, ``command not found``,
      ``no such file or directory``).
    - Empty assistant turns (no content, no tool_calls).
    - Orphaned assistant ``tool_calls`` whose IDs have no matching
      ``role: "tool"`` result (when ``remove_orphaned=True``).

    Does NOT drop tool results just because they have ``is_error=True`` or
    because their content contains the words "error" or "failed". The
    earlier broad-regex filter destroyed ~45% of legitimate file-read and
    traceback results — exactly the failure-recovery trajectories that are
    the most valuable signal for agent SFT. Quality scoring of
    error-heavy sessions is the scorer's job, not the cleaner's.
    """
    messages = conversation.get("messages", [])
    removed_count = 0

    cleaned_messages: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")

        if role in ("system", "user"):
            cleaned_messages.append(msg)
            continue

        if role == "assistant":
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            if content.strip() or tool_calls:
                cleaned_messages.append(msg)
            continue

        if role == "tool":
            content = msg.get("content", "")
            if has_failed_command(content):
                removed_count += 1
                continue
            cleaned_messages.append(msg)

    if remove_orphaned:
        cleaned_messages = remove_orphaned_tool_calls(cleaned_messages)

    return (
        {
            **conversation,
            "messages": cleaned_messages,
            "_cleaned_stats": (
                {"removed_tool_results": removed_count} if removed_count > 0 else None
            ),
        },
        removed_count,
    )


def format_for_training(conversation: dict[str, Any]) -> dict[str, Any]:
    """Prepare a conversation record for ``trainer.py`` consumption.

    The trainer loads JSONL via ``load_dataset("json", …)`` and then calls
    ``ds.select_columns(["messages"])``, so any fields outside ``messages``
    are dropped.  To ensure the tool schemas and tool-call arguments survive:

    1. **Embed tools in first message** – moves the top-level ``tools`` list
       into ``messages[0]["tools"]`` where ``_prepare_messages_and_tools``
       will find it.
    2. **Parse ``arguments`` to dicts** – the parsers store ``arguments`` as
       JSON strings (OpenAI API format), but ``apply_chat_template`` expects
       dicts so it can serialise them itself.
    3. **Strip internal fields** – removes ``_cleaned_stats`` and other
       pipeline-only metadata that the trainer doesn't need.
    """
    conv = dict(conversation)
    messages = [dict(m) for m in conv.get("messages", [])]

    # 1. Embed top-level tools into the first message
    tools = conv.get("tools")
    if tools and messages:
        messages[0] = {**messages[0], "tools": tools}

    # 2. Convert tool_call arguments from JSON strings → dicts
    for msg in messages:
        if msg.get("tool_calls"):
            new_calls = []
            for tc in msg["tool_calls"]:
                tc = dict(tc)
                func = tc.get("function")
                if isinstance(func, dict):
                    func = dict(func)
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    tc["function"] = func
                new_calls.append(tc)
            msg["tool_calls"] = new_calls

    conv["messages"] = messages

    # 3. Strip internal pipeline fields
    for key in ("_cleaned_stats",):
        conv.pop(key, None)

    return conv


def clean_file(
    input_path: Path,
    output_path: Path,
    remove_orphaned: bool = True,
) -> None:
    """Clean all conversations in a JSONL file."""
    conversations: list[dict[str, Any]] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
                conv["id"] = conv.get("id", str(len(conversations)))
                conversations.append(conv)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(conversations)} conversations")
    total_removed = 0
    cleaned_conversations = []

    for conv in conversations:
        cleaned, removed = clean_conversation(conv, remove_orphaned=remove_orphaned)
        if removed > 0:
            print(f"Cleaned {conv.get('id', '')}: removed {removed} tool results")
        total_removed += removed
        cleaned_conversations.append(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for conv in cleaned_conversations:
            f.write(json.dumps(conv) + "\n")

    print(f"Total tool results removed: {total_removed}")
    print(f"Output written to: {output_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean up training data by removing problematic tool calls/results"
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--no-remove-orphaned",
        action="store_true",
        help="Keep orphaned tool calls",
    )
    args = parser.parse_args()

    clean_file(
        Path(args.input),
        Path(args.output),
        remove_orphaned=not args.no_remove_orphaned,
    )


if __name__ == "__main__":
    main()
