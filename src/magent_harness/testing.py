"""Deterministic stand-ins for model calls, for tests and offline demos."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .messages import Message, ToolCall, Usage
from .model import FINISH_STOP, FINISH_TOOL_CALLS, ChatResponse
from .tools import ToolSpec

ScriptItem = ChatResponse | BaseException | Callable[[list[Message], list[ToolSpec]], ChatResponse]


@dataclass
class ModelCall:
    messages: list[Message]
    tools: list[ToolSpec]
    max_tokens: int | None

    @property
    def tool_names(self) -> list[str]:
        return [spec.name for spec in self.tools]


class ScriptedModel:
    """Returns scripted responses in order and records every request."""

    def __init__(self, script: Iterable[ScriptItem] = ()) -> None:
        self._script: list[ScriptItem] = list(script)
        self.calls: list[ModelCall] = []

    def extend(self, script: Iterable[ScriptItem]) -> None:
        self._script.extend(script)

    @property
    def remaining(self) -> int:
        return len(self._script)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        self.calls.append(ModelCall(list(messages), list(tools), max_tokens))
        if not self._script:
            raise AssertionError("ScriptedModel ran out of responses")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(messages, tools)
        return item


def tool_call(name: str, args: dict[str, Any] | str | None = None, call_id: str = "") -> ToolCall:
    arguments = args if isinstance(args, str) else json.dumps(args or {})
    return ToolCall(call_id or f"call_{name}", name, arguments)


def call_tools(*calls: ToolCall, content: str = "", usage: Usage | None = None) -> ChatResponse:
    return ChatResponse(
        content=content,
        tool_calls=list(calls),
        finish_reason=FINISH_TOOL_CALLS,
        usage=usage or Usage(),
    )


def reply(
    content: str, *, finish_reason: str = FINISH_STOP, usage: Usage | None = None
) -> ChatResponse:
    return ChatResponse(content=content, finish_reason=finish_reason, usage=usage or Usage())
