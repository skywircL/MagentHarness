"""Bounded SSE framing and private DeepSeek completion assembly."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .model import (
    FINISH_CONTENT_FILTER,
    FINISH_LENGTH,
    FINISH_STOP,
    FINISH_TOOL_CALLS,
    ModelError,
    TransientModelError,
)

MAX_EVENT_BYTES = 1024 * 1024
MAX_STREAM_BYTES = 16 * 1024 * 1024
MAX_TOOL_CALLS = 128
_LINE_END = re.compile(rb"[\r\n]")
_INVALID = "DeepSeek returned an invalid completion stream"


class SSEDecoder:
    """Decode SSE events without assuming HTTP or UTF-8 character boundaries."""

    def __init__(self) -> None:
        self._line = bytearray()
        self._data: list[str] = []
        self._event = ""
        self._event_bytes = 0
        self._total_bytes = 0
        self._skip_lf = False
        self._first_line = True

    def feed(self, chunk: bytes) -> Iterator[tuple[str, str]]:
        self._total_bytes += len(chunk)
        if self._total_bytes > MAX_STREAM_BYTES:
            raise ModelError("DeepSeek stream exceeded the size limit")
        start = 0
        for delimiter in _LINE_END.finditer(chunk):
            position = delimiter.start()
            segment = chunk[start:position]
            is_lf = chunk[position] == 10
            start = position + 1
            if self._skip_lf and not segment and is_lf:
                self._skip_lf = False
                continue
            self._skip_lf = not is_lf
            self._append(segment)
            event = self._finish_line()
            if event is not None:
                yield event
        if start < len(chunk):
            self._skip_lf = False
            self._append(chunk[start:])

    def _append(self, segment: bytes) -> None:
        self._event_bytes += len(segment)
        if self._event_bytes > MAX_EVENT_BYTES:
            raise ModelError("DeepSeek stream event exceeded the size limit")
        self._line.extend(segment)

    def _finish_line(self) -> tuple[str, str] | None:
        try:
            line = self._line.decode("utf-8")
        except UnicodeDecodeError:
            raise ModelError("DeepSeek stream contained invalid UTF-8") from None
        self._line.clear()
        self._event_bytes += 1
        if self._first_line:
            line = line.removeprefix("\ufeff")
            self._first_line = False
        if not line:
            event = (self._event, "\n".join(self._data)) if self._data or self._event else None
            self._data = []
            self._event = ""
            self._event_bytes = 0
            return event
        if line.startswith(":"):
            return None
        name, separator, value = line.partition(":")
        if not separator:
            value = ""
        if value.startswith(" "):
            value = value[1:]
        if name == "data":
            self._data.append(value)
        elif name == "event":
            self._event = value
        return None


@dataclass
class _ToolFragments:
    id: str | None = None
    name: str | None = None
    kind: str | None = None
    arguments: list[str] = field(default_factory=list)

    def add(self, value: dict[str, Any]) -> None:
        if value.get("type") not in (None, "function"):
            raise ModelError(_INVALID)
        if value.get("type") == "function":
            self.kind = "function"
        self.id = self._metadata(self.id, value.get("id"))
        function = value.get("function")
        if function is None:
            return
        if not isinstance(function, dict):
            raise ModelError(_INVALID)
        self.name = self._metadata(self.name, function.get("name"))
        arguments = function.get("arguments")
        if arguments is not None:
            if not isinstance(arguments, str):
                raise ModelError(_INVALID)
            self.arguments.append(arguments)

    @staticmethod
    def _metadata(previous: str | None, value: Any) -> str | None:
        if value is None:
            return previous
        if not isinstance(value, str) or not value or (previous is not None and previous != value):
            raise ModelError(_INVALID)
        return value

    def body(self) -> dict[str, Any]:
        if self.id is None or self.name is None or self.kind != "function" or not self.arguments:
            raise ModelError(_INVALID)
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": "".join(self.arguments)},
        }


class CompletionStream:
    """Assemble only a single choice; expose text, never reasoning/tool fragments."""

    def __init__(self) -> None:
        self.done = False
        self._finish: str | None = None
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._has_reasoning = False
        self._tools: dict[int, _ToolFragments] = {}
        self._usage: dict[str, Any] = {}

    def accept(self, event: str, data: str) -> str | None:
        if event not in ("", "message"):
            raise ModelError(_INVALID)
        if data == "[DONE]":
            if self._finish is None:
                raise TransientModelError("DeepSeek stream ended before completion")
            self.done = True
            return None
        try:
            chunk = json.loads(data)
        except (ValueError, RecursionError):
            raise ModelError(_INVALID) from None
        if not isinstance(chunk, dict) or "error" in chunk:
            raise ModelError(_INVALID)
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            raise ModelError(_INVALID)
        usage = chunk.get("usage")
        if usage is not None:
            self._set_usage(usage)
        if not choices:
            if self._finish is None or usage is None:
                raise ModelError(_INVALID)
            return None
        if len(choices) != 1 or not isinstance(choices[0], dict) or self._finish is not None:
            raise ModelError(_INVALID)
        choice = choices[0]
        if type(choice.get("index")) is not int or choice["index"] != 0:
            raise ModelError(_INVALID)
        finish = choice.get("finish_reason")
        if finish in ("insufficient_system_resource", "aborted"):
            raise TransientModelError("DeepSeek generation was interrupted")
        if finish not in (
            None,
            FINISH_STOP,
            FINISH_TOOL_CALLS,
            FINISH_LENGTH,
            FINISH_CONTENT_FILTER,
        ):
            raise ModelError(_INVALID)
        delta = choice.get("delta")
        if not isinstance(delta, dict) or delta.get("role") not in (None, "assistant"):
            raise ModelError(_INVALID)
        content = self._text(delta, "content")
        reasoning = self._text(delta, "reasoning_content")
        calls = delta.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list):
                raise ModelError(_INVALID)
            for call in calls:
                if not isinstance(call, dict):
                    raise ModelError(_INVALID)
                index = call.get("index")
                if type(index) is not int or not 0 <= index < MAX_TOOL_CALLS:
                    raise ModelError(_INVALID)
                self._tools.setdefault(index, _ToolFragments()).add(call)
        if content is not None:
            self._content.append(content)
        if reasoning is not None:
            self._has_reasoning = True
            self._reasoning.append(reasoning)
        self._finish = finish
        return content

    def _set_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            raise ModelError(_INVALID)
        for name in ("prompt_tokens", "completion_tokens"):
            value = usage.get(name, 0)
            if type(value) is not int or value < 0:
                raise ModelError(_INVALID)
        self._usage = usage

    @staticmethod
    def _text(delta: dict[str, Any], name: str) -> str | None:
        text = delta.get(name)
        if text is not None and not isinstance(text, str):
            raise ModelError(_INVALID)
        return text

    def body(self) -> dict[str, Any]:
        if not self.done or self._finish is None:
            raise TransientModelError("DeepSeek stream ended before completion")
        if set(self._tools) != set(range(len(self._tools))):
            raise ModelError(_INVALID)
        calls = [self._tools[index].body() for index in sorted(self._tools)]
        if len({call["id"] for call in calls}) != len(calls):
            raise ModelError(_INVALID)
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(self._content),
            "tool_calls": calls,
        }
        if self._has_reasoning:
            message["reasoning_content"] = "".join(self._reasoning)
        return {
            "choices": [{"message": message, "finish_reason": self._finish}],
            "usage": self._usage,
        }
