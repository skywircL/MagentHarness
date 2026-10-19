"""Conversation and step records shared by the engine, context manager and snapshots.

Everything here is plain data with explicit ``to_dict``/``from_dict`` so a run
can be persisted as JSON and resumed by another process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]

# Message.kind marks harness-generated messages that later stages must
# recognize; ordinary conversation leaves it empty.
KIND_COMPACTION_SUMMARY = "compaction_summary"


@dataclass(frozen=True)
class ToolCall:
    """A tool call as the model emitted it. ``arguments`` stays raw JSON text."""

    id: str
    name: str
    arguments: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(id=data["id"], name=data["name"], arguments=data["arguments"])


@dataclass(frozen=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str = ""
    name: str = ""
    kind: str = ""
    # Preserve exactly for providers that require it in subsequent tool rounds.
    reasoning_content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        if self.kind:
            data["kind"] = self.kind
        if self.reasoning_content is not None:
            data["reasoning_content"] = self.reasoning_content
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        reasoning = data.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("message reasoning_content must be a string or null")
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=tuple(ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])),
            tool_call_id=data.get("tool_call_id", ""),
            name=data.get("name", ""),
            kind=data.get("kind", ""),
            reasoning_content=reasoning,
        )


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Usage:
        return cls(data.get("prompt_tokens", 0), data.get("completion_tokens", 0))


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, output: str, data: dict[str, Any] | None = None) -> ToolResult:
        return cls(success=True, output=output, data=data or {})

    @classmethod
    def fail(cls, error: str) -> ToolResult:
        return cls(success=False, error=error)

    def model_content(self) -> str:
        """The text the model sees for this result."""
        if self.success:
            return self.output
        return f"Error: {self.error}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResult:
        return cls(
            success=data["success"],
            output=data.get("output", ""),
            error=data.get("error", ""),
            data=data.get("data", {}),
        )


@dataclass
class ToolCallRecord:
    """One executed (or refused) tool call and its result."""

    id: str
    name: str
    arguments: str
    result: ToolResult
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result.to_dict(),
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCallRecord:
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data["arguments"],
            result=ToolResult.from_dict(data["result"]),
            duration_ms=data.get("duration_ms", 0),
        )

    def to_message(self) -> Message:
        return Message(
            role="tool",
            content=self.result.model_content(),
            tool_call_id=self.id,
            name=self.name,
        )


@dataclass
class AgentStep:
    """One ReAct round: the model's thought and the tool calls it made."""

    round: int
    thought: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "thought": self.thought,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentStep:
        return cls(
            round=data["round"],
            thought=data.get("thought", ""),
            tool_calls=[ToolCallRecord.from_dict(tc) for tc in data.get("tool_calls", [])],
        )
