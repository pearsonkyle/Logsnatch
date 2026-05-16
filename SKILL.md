---
name: logsnatch
description: Parse, redact, score, and filter coding-agent conversation logs (Claude Code, OpenCode, Qwen Code) into JSONL training data, and add new parsers for additional agents. Use this skill whenever the user wants to turn raw agent session logs into a training-ready dataset, scrub secrets or anonymize paths in conversation logs, score or filter agent transcripts by quality, validate JSONL records against a Qwen-style chat template, or add support for a new coding-agent log format. Trigger even when the user does not name the CLI explicitly — phrases like "extract my Claude sessions", "make a fine-tuning dataset from my logs", "redact secrets in these JSONLs", or "add a Codex parser" all apply.
---

# logsnatch

A small Python CLI that converts raw coding-agent session logs into clean,
redacted, quality-scored JSONL suitable for SFT training. Out of the box it
understands Claude Code, OpenCode, and Qwen Code logs. The parser layer is
pluggable, so adding a new agent is a single-file change plus a registry entry.

## Quick mental model

The pipeline is a chain of pure-ish JSONL → JSONL transforms:

```
raw logs ──parse──▶ raw.jsonl ──redact──▶ redacted.jsonl
                                              │
                                              ▼
                                          clean ──▶ cleaned.jsonl
                                              │
                                              ▼
                                          evaluate (adds `score`) ──▶ scored.jsonl
                                              │
                                              ▼
                                          filter (drops by score, formats for training) ──▶ final.jsonl
```

Each stage has its own subcommand, and `run` chains them all. Intermediate
files are kept (suffixed `_raw`, `_redacted`, `_cleaned`, `_scored`) so you can
inspect or re-run any stage in isolation. That matters: most debugging of "why
is my dataset weird" is done by diffing two adjacent stages.

## Install

```bash
pip install -e .                  # core CLI, stdlib only
pip install -e .[validate]        # adds transformers for the validate subcommand
pip install -e .[test]            # adds pytest
```

The core stages depend only on the Python standard library, so a fresh
container or a constrained sandbox can run parse/redact/clean/evaluate/filter
without touching the network. Only `validate` pulls in `transformers` (it
needs a tokenizer to apply a chat template).

## Using the CLI

The entry point is `python -m logsnatch <command>`. Every subcommand reads and
writes JSONL.

### Common workflows

End-to-end, default settings:

```bash
python -m logsnatch run --source claude --output data/training.jsonl
```

Parse only, then validate against a tokenizer (good when you want to confirm
records will round-trip through `apply_chat_template` before spending time on
the rest of the pipeline):

```bash
python -m logsnatch parse --source claude --output data/raw.jsonl
python -m logsnatch validate --input data/raw.jsonl --model Qwen/Qwen3.5-4B
```

Pull from all supported sources and keep only high-quality sessions:

```bash
python -m logsnatch run --source all --output data/out.jsonl --min-score 0.7
```

### Subcommand reference

| Command | Purpose | Required flags | Notable optional flags |
|---|---|---|---|
| `parse` | Discover and parse provider logs into a uniform OpenAI-style messages format. | `--source {claude,opencode,qwen,all}`, `--output` | `--input` (defaults to the provider's standard log dir, e.g. `~/.claude/projects` for Claude). |
| `redact` | Anonymize paths/usernames and scrub secrets (JWTs, API keys, DB URLs, etc.) from message content and tool-call arguments. | `--input`, `--output` | — |
| `clean` | Drop malformed tool calls, orphaned tool results, and empty messages. | `--input`, `--output` | — |
| `evaluate` | Score each conversation on quality and attach a `score` field. | `--input`, `--output` | `--min-turns` (default 2), `--min-token-count` (default 1000). |
| `filter` | Keep records with `score >= --min-score` and emit them in training format. | `--input`, `--output` | `--min-score` (default 0.5). |
| `validate` | Apply a chat template via `transformers.AutoTokenizer` and report tokens, tool-call counts, and any failures. | `--input` *or* `--source` | `--output` (writes only records that passed), `--model` (default `Qwen/Qwen3.5-4B`). |
| `run` | Chains parse → redact → clean → evaluate → filter, writing each intermediate alongside the final output. | `--source`, `--output` | `--input`, `--min-score` (default 0.5). |

### Choosing flags

- **Score threshold.** `--min-score 0.5` is the default and is forgiving. Raise
  it (0.7+) when you want a tighter dataset and have plenty of source logs;
  lower it when you're data-starved or experimenting.
- **`--min-turns` / `--min-token-count`.** These define what counts as a
  "real" conversation. Drop them for sparse log sources; raise them when you
  want only substantive sessions.
- **`--source all`.** Convenient, but it parses every provider's default log
  dir. Pass `--source <name>` + `--input <path>` if you want to point at a
  specific export instead.

### Output schema

A record after `parse` looks roughly like this:

```json
{
  "id": "<session id>",
  "source": "claude",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."}
  ]
}
```

After `evaluate`, records gain a `score` field. After `filter`, the first
message also carries a `tools` array (the OpenAI-style tool schemas inferred
from observed tool calls), which the trainer pipeline expects.

## Adding a new parser (extending the skill to a new agent)

The parser layer is the right extension point about 90% of the time — the
pipeline stages are agent-agnostic and operate on the uniform message format.

### Where things live

```
logsnatch/parsers/
├── __init__.py     # REGISTRY dict + get_parser()
├── base.py         # BaseParser ABC + build_tool_schema() helper
├── claude.py
├── opencode.py
└── qwen.py
```

### The contract (`BaseParser`)

```python
class BaseParser(ABC):
    SOURCE: str                # short name used on the CLI (e.g., "claude")
    DEFAULT_LOG_DIR: Path      # where logs live by default

    def discover_sessions(self, input_path: Path) -> list[Path]: ...
    def parse_session(self, session_path: Path, **kwargs) -> dict | None: ...
    # parse_all() is provided; iterates discover → parse, swallowing per-session errors.
```

`parse_session` should return a dict shaped like the schema above, or `None`
if the session is empty/junk and should be silently dropped. Return shape is
load-bearing — downstream stages assume `messages` is a list of dicts with
`role` and (where applicable) `content`, `tool_calls`, `tool_call_id`.

### Steps to add a new agent (e.g., "codex")

1. **Create `logsnatch/parsers/codex.py`.** Subclass `BaseParser`, set `SOURCE =
   "codex"` and `DEFAULT_LOG_DIR` to wherever that agent stores logs (use
   `Path.home() / ...`). Implement `discover_sessions` (usually a `glob`) and
   `parse_session` (read the file, normalize each turn into the OpenAI
   message shape).
2. **Register it in `logsnatch/parsers/__init__.py`** by importing the class and
   adding `"codex": CodexParser` to `REGISTRY`. That single line makes
   `--source codex` work everywhere — `parse`, `validate`, and `run`.
3. **Reuse helpers.** `build_tool_schema(func_name, args)` infers an
   OpenAI-style tool schema from a sample call; use it when you need to emit
   a `tools` array. For `<think>...</think>` reasoning blocks, see how
   `qwen.py`'s `_extract_thinking` handles them.
4. **Add a test in `tests/test_parsers.py`.** The existing tests build a
   tmpdir with a fake session file and assert the parser returns the right
   message shape — copy that pattern.

### Practical tips

- **Sort by timestamp early.** Most agent logs interleave messages
  out-of-order; sort once at the top of `parse_session` and downstream code
  becomes much simpler.
- **Pair tool calls with their results.** Assistant `tool_calls` need
  matching `role: "tool"` messages with the same `tool_call_id`. Orphans get
  dropped by `clean`, but it's better to never emit them.
- **Be conservative about content.** If a turn is empty, telemetry, or a
  system housekeeping message, return nothing for it rather than emitting a
  blank string — the cleaner is a safety net, not a substitute for parser
  hygiene.
- **Don't try to score in the parser.** Quality scoring is the `evaluate`
  stage's job; the parser's only job is faithful normalization.

## Extending other stages

Less common, but if you need to:

- **New redaction rules:** add a regex + replacement in
  `logsnatch/redaction/secrets.py`. The function `redact_text` returns
  `(redacted, count)` — keep that contract so the CLI's reporting still
  works.
- **New scoring signal:** edit `logsnatch/pipeline/evaluate.py`. Combine your
  signal into the existing `score` (a float in roughly [0, 1]) rather than
  inventing a parallel field — `filter` only knows about `score`.
- **Custom cleaning rule:** edit `logsnatch/pipeline/cleanup.py`'s
  `clean_conversation`. The training-format conversion lives in
  `format_for_training` in the same file — keep cleaning and formatting
  separate.

## Loading the output into a HuggingFace trainer

The `filter` (and `run`) commands emit JSONL that loads cleanly via vanilla
`datasets.load_dataset("json", …)` — no post-processing required.

Two design choices in the writer make this work, and they're worth
understanding if you're extending the CLI:

1. **`messages` is serialized to `list[str]`** (each turn is a JSON-encoded
   string). PyArrow infers a single schema across all rows in a JSONL, and
   different sessions have different `tool_calls`/`tools`/`content` shapes,
   so `list[dict]` would fail to unify. `list[str]` is a uniform column.
   The trainer (`LLMTrainer._prepare_messages_and_tools`) already handles
   both shapes — it parses the JSON strings back into dicts transparently.
2. **Top-level `tools` and `metadata` are dropped on write.** Tools are
   already embedded inside `messages[0]["tools"]` (which the trainer reads
   after `select_columns(["messages"])`), so the top-level copy was
   redundant — and its heterogeneous shape across sessions was Arrow-hostile.
   `metadata` had the same problem and isn't used downstream.

To consume the output yourself, mirror the trainer's parse step:

```python
import json
from datasets import load_dataset

ds = load_dataset("json", data_files="train.jsonl", split="train")
for rec in ds:
    msgs = [json.loads(m) for m in rec["messages"]]   # list[str] → list[dict]
    tools = msgs[0].pop("tools", None) if msgs else None
    # ... apply_chat_template(msgs, tools=tools, ...)
```

Verified on a 75-record real Claude-Code extraction: 100% load + 100%
chat-template pass against `Qwen/Qwen3.5-4B`.

## Testing

```bash
pytest                      # runs tests/test_parsers.py, test_pipeline.py, test_redaction.py
pytest tests/test_parsers.py -k claude   # one parser at a time
```

When adding a parser, the minimum bar is one happy-path test that builds a
fake session file, runs `parse_session`, and asserts the message shape.
Round-tripping through `validate` against a real tokenizer is the strongest
end-to-end signal — do it once on real logs before declaring victory.

## When things go wrong

- **`validate` fails on a record:** open the failing record in
  `<output>_cleaned.jsonl` and look for missing `tool_call_id` pairings or
  empty assistant content. Most chat-template failures trace back to those.
- **Suspiciously few records survive `filter`:** check the distribution of
  `score` in `<output>_scored.jsonl`. If most scores are under 0.5, your
  source logs may be noisy/short — lower `--min-score` or raise
  `--min-turns`/`--min-token-count` thresholds in `evaluate` to better
  match your data.
- **Secrets leaking through:** add a regex to `secrets.py` and rerun just
  the `redact` stage on the existing `_raw` file — no need to re-parse.
