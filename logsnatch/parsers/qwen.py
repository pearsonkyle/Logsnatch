import json
import re
from pathlib import Path
from typing import Any

from logsnatch.parsers.base import BaseParser, build_tool_schema


def _extract_thinking(text: str) -> tuple[str, list[str]]:
    pattern = r"<think>(.*?)</think>"
    thoughts = re.findall(pattern, text, re.DOTALL)
    clean = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    return clean, thoughts


class QwenParser(BaseParser):
    SOURCE = "qwen"
    DEFAULT_LOG_DIR = Path.home() / ".qwen" / "projects"

    def discover_sessions(self, input_path: Path) -> list[Path]:
        return sorted(input_path.glob("*/chats/*.jsonl"))

    def parse_session(
        self,
        session_path: Path,
        include_thinking: bool = True,
        min_turns: int = 2,
        **kwargs: Any,
    ) -> dict | None:
        raw_messages: list[dict[str, Any]] = []
        with open(session_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not raw_messages:
            return None

        sorted_msgs = sorted(raw_messages, key=lambda x: x.get("timestamp", ""))
        conversation: list[dict[str, Any]] = []
        processed: set[str] = set()
        seen_tools: dict[str, dict[str, Any]] = {}
        timestamps: list[str] = []

        for msg in sorted_msgs:
            uuid = msg.get("uuid", "")
            if uuid in processed:
                continue
            ts = msg.get("timestamp", "")
            if ts:
                timestamps.append(ts)

            msg_type = msg.get("type", "")
            subtype = msg.get("subtype", "")

            if msg_type == "system" and subtype == "ui_telemetry":
                processed.add(uuid)
                continue

            if msg_type == "user":
                parts = msg.get("message", {}).get("parts", [])
                content_parts = []
                for part in parts:
                    text = part.get("text", "")
                    if text and "--- Content from referenced files ---" not in text:
                        content_parts.append(text)
                content = "\n".join(content_parts).strip()
                if content:
                    conversation.append({"role": "user", "content": content})
                processed.add(uuid)

            elif msg_type == "assistant":
                parts = msg.get("message", {}).get("parts", [])
                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []

                for part in parts:
                    if "text" in part:
                        text_parts.append(part["text"])
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        func_name = fc.get("name", "")
                        args = fc.get("args", {})
                        call_id = fc.get("id", f"call_{len(tool_calls)}")
                        if func_name and func_name not in seen_tools:
                            seen_tools[func_name] = build_tool_schema(func_name, args)
                        tool_calls.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": func_name,
                                    "arguments": json.dumps(args),
                                },
                            }
                        )

                full_text = "\n".join(text_parts)
                content, thoughts = _extract_thinking(full_text)

                if include_thinking and thoughts:
                    thought_block = "<think>\n" + "\n".join(thoughts) + "\n</think>"
                    content = (
                        (thought_block + "\n" + content).strip()
                        if content
                        else thought_block
                    )

                result: dict[str, Any] = {"role": "assistant"}
                if content:
                    result["content"] = content
                if tool_calls:
                    result["tool_calls"] = tool_calls
                if "content" in result or "tool_calls" in result:
                    conversation.append(result)
                processed.add(uuid)

            elif msg_type == "tool_result":
                tool_call_id = msg.get("toolCallResult", {}).get("callId", "")
                parts = msg.get("message", {}).get("parts", [])
                for part in parts:
                    if "functionResponse" in part:
                        fr = part["functionResponse"]
                        func_name = fr.get("name", "")
                        response = fr.get("response", {})
                        output = response.get("output", "")
                        error = response.get("error", "")
                        content_parts = []
                        if output:
                            content_parts.append(output)
                        if error:
                            content_parts.append(f"Error: {error}")
                        if func_name or content_parts:
                            conversation.append(
                                {
                                    "role": "tool",
                                    "name": func_name,
                                    "tool_call_id": tool_call_id,
                                    "content": "\n".join(content_parts).strip(),
                                }
                            )
                processed.add(uuid)

        valid = [m for m in conversation if m.get("content") or m.get("tool_calls")]
        if len(valid) < min_turns:
            return None

        start_time = timestamps[0] if timestamps else None
        end_time = timestamps[-1] if timestamps else None

        return {
            "id": session_path.stem,
            "source": self.SOURCE,
            "metadata": {
                "model": None,
                "cwd": str(session_path.parent.parent),
                "start_time": start_time,
                "end_time": end_time,
            },
            "tools": list(seen_tools.values()),
            "messages": valid,
        }
