"""Versioned run snapshot: everything needed to resume a suspended run elsewhere.

The format is a persistence contract. Changing a field's meaning needs a new
``SNAPSHOT_VERSION`` and a decoder for the old one; readers reject versions
they do not know rather than guessing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .identity import Caller, ResourceScope
from .messages import AgentStep, Message, ToolCall, ToolCallRecord, ToolResult, Usage
from .policy import ToolPolicy

SNAPSHOT_VERSION = 2


class SnapshotError(Exception):
    pass


def _object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _array(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return value


def _text(value: Any, field_name: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        suffix = "a nonempty string" if nonempty else "a string"
        raise ValueError(f"{field_name} must be {suffix}")
    return value


def _integer(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


def _strings(value: Any, field_name: str) -> None:
    if value is None:
        return
    values = _array(value, field_name)
    for item in values:
        _text(item, field_name, nonempty=True)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")


def _call(data: Any) -> None:
    data = _object(data, "tool call")
    _text(data["id"], "tool call id", nonempty=True)
    _text(data["name"], "tool call name", nonempty=True)
    # An empty string remains valid: execution treats it as an empty object.
    _text(data["arguments"], "tool arguments")


def _calls(data: Any) -> None:
    calls = _array(data, "tool calls")
    for call in calls:
        _call(call)
    if len({call["id"] for call in calls}) != len(calls):
        raise ValueError("tool call ids must be unique within a round")


def _record(data: Any) -> None:
    _call(data)
    _integer(data.get("duration_ms", 0), "tool duration_ms")
    result = _object(data["result"], "tool result")
    if type(result["success"]) is not bool:
        raise ValueError("tool result success must be a boolean")
    _text(result.get("output", ""), "tool result output")
    _text(result.get("error", ""), "tool result error")
    _object(result.get("data", {}), "tool result data")


def _match_tool_record(record: ToolCallRecord, call: ToolCall) -> None:
    if (record.id, record.name) != (call.id, call.name):
        raise SnapshotError(f"finished call {record.id} does not match {call.id}")
    if record.arguments == call.arguments:
        return
    # Existing snapshots retain the model's original arguments and disclose
    # approved user edits in the completed record.
    disclosure = f"(The user edited the arguments before approving: {record.arguments})\n"
    if record.result.data.get(
        "modified_by_user"
    ) is not True or not record.result.output.startswith(disclosure):
        raise SnapshotError(f"finished call {record.id} arguments do not match")
    try:
        edited = json.loads(record.arguments or "{}", parse_constant=_reject_constant)
        _object(edited, "edited tool arguments")
    except (TypeError, ValueError) as exc:
        raise SnapshotError(f"finished call {record.id} has invalid edits") from exc


def validate_tool_record(data: dict[str, Any], call: ToolCall | None = None) -> ToolCallRecord:
    """Decode one durable result, optionally checking its planned call identity.

    Parallel completions may arrive before earlier calls, so validating a
    result must not require a contiguous ``PendingRound.done`` prefix.
    """
    try:
        _record(data)
        record = ToolCallRecord.from_dict(data)
        if call is not None:
            _match_tool_record(record, call)
        return record
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotError(f"malformed tool record: {exc}") from exc


def _messages(data: Any) -> None:
    outstanding: dict[str, str] = {}
    for message in _array(data, "messages"):
        message = _object(message, "message")
        role = message["role"]
        if role not in ("user", "assistant", "tool"):
            raise ValueError("snapshot messages must have user, assistant, or tool roles")
        for name in ("content", "tool_call_id", "name", "kind"):
            _text(message.get(name, ""), f"message {name}")
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("message reasoning_content must be a string or null")
        calls = message.get("tool_calls", [])
        _calls(calls)
        if role == "tool":
            call_id = message.get("tool_call_id", "")
            if calls or call_id not in outstanding:
                raise ValueError("tool message must match an unfinished assistant tool call")
            if message.get("name", "") not in ("", outstanding[call_id]):
                raise ValueError("tool message name does not match its call")
            del outstanding[call_id]
        else:
            if outstanding:
                raise ValueError("assistant tool calls must have paired results")
            if message.get("tool_call_id", ""):
                raise ValueError("only tool messages may have tool_call_id")
            if calls and role != "assistant":
                raise ValueError("only assistant messages may have tool calls")
            outstanding = {call["id"]: call["name"] for call in calls}
    if outstanding:
        raise ValueError("assistant tool calls must have paired results")


def _validate_data(data: dict[str, Any], *, pending_required: bool) -> None:
    _text(data["run_id"], "run_id", nonempty=True)
    _text(data["session_id"], "session_id")
    _text(data["query"], "query", nonempty=True)
    _text(data.get("memory_prompt", ""), "memory_prompt")
    context_limit = data.get("context_input_limit")
    if context_limit is not None and (type(context_limit) is not int or context_limit <= 0):
        raise ValueError("context_input_limit must be a positive integer or null")
    caller = _object(data["caller"], "caller")
    for name in ("workspace_id", "user_id", "role"):
        _text(caller[name], f"caller {name}", nonempty=True)
    Caller.from_dict(caller)
    scope = _object(data["scope"], "scope")
    _text(scope["workspace_id"], "scope workspace_id", nonempty=True)
    _strings(scope.get("document_ids"), "scope document_ids")
    policy = _object(data["policy"], "policy")
    _text(policy["access"], "policy access", nonempty=True)
    _strings(policy.get("allowed_tools"), "policy allowed_tools")
    ToolPolicy.from_dict(policy)
    _messages(data["messages"])
    current_round = _integer(data["current_round"], "current_round")
    for step in _array(data["steps"], "steps"):
        step = _object(step, "step")
        if _integer(step["round"], "step round") >= current_round:
            raise ValueError("completed step round must precede current_round")
        _text(step.get("thought", ""), "step thought")
        records = _array(step.get("tool_calls", []), "step tool calls")
        for record in records:
            _record(record)
        if len({record["id"] for record in records}) != len(records):
            raise ValueError("step tool call ids must be unique within a round")
    usage = _object(data.get("usage", {}), "usage")
    for name in ("prompt_tokens", "completion_tokens"):
        _integer(usage.get(name, 0), f"usage {name}")
    if not pending_required and data.get("pending") is None:
        return
    pending = _object(data["pending"], "pending")
    if _integer(pending["round"], "pending round") != current_round:
        raise ValueError("pending round must match current_round")
    _text(pending.get("thought", ""), "pending thought")
    reasoning = pending.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError("pending reasoning_content must be a string or null")
    _calls(pending["tool_calls"])
    for record in _array(pending.get("done", []), "pending done"):
        _record(record)
    _integer(pending["blocked_index"], "pending blocked_index")
    PendingRound.from_dict(pending).validate()


def validate_snapshot_data(data: dict[str, Any], *, pending_required: bool = True) -> None:
    """Validate shared persisted run fields, including a pending round when present.

    Ordinary run checkpoints use ``pending_required=False`` at a model boundary;
    approval snapshots always require an unfinished tool round. Format versions
    and authenticated ownership checks belong to their respective entry points.
    """
    try:
        _validate_data(_object(data, "snapshot"), pending_required=pending_required)
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotError(f"malformed run snapshot: {exc}") from exc


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value} is not supported")


@dataclass
class PendingRound:
    """The round interrupted by a prompt.

    ``done`` holds the results of the calls before ``blocked_index``; the
    blocked call and every call after it have not run.
    """

    round: int
    thought: str
    tool_calls: list[ToolCall]
    done: list[ToolCallRecord]
    blocked_index: int
    reasoning_content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "round": self.round,
            "thought": self.thought,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "done": [record.to_dict() for record in self.done],
            "blocked_index": self.blocked_index,
        }
        if self.reasoning_content is not None:
            data["reasoning_content"] = self.reasoning_content
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingRound:
        reasoning = data.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("pending reasoning_content must be a string or null")
        return cls(
            round=data["round"],
            thought=data.get("thought", ""),
            tool_calls=[ToolCall.from_dict(tc) for tc in data["tool_calls"]],
            done=[ToolCallRecord.from_dict(r) for r in data.get("done", [])],
            blocked_index=data["blocked_index"],
            reasoning_content=reasoning,
        )

    def validate(self) -> None:
        if not 0 <= self.blocked_index < len(self.tool_calls):
            raise SnapshotError(
                f"blocked call {self.blocked_index} outside round of {len(self.tool_calls)} calls"
            )
        if len(self.done) != self.blocked_index:
            raise SnapshotError(
                f"{len(self.done)} finished calls recorded before blocked call {self.blocked_index}"
            )
        for record, call in zip(self.done, self.tool_calls, strict=False):
            _match_tool_record(record, call)


@dataclass
class RunSnapshot:
    run_id: str
    session_id: str
    caller: Caller
    scope: ResourceScope
    policy: ToolPolicy
    query: str
    memory_prompt: str
    # The transcript without the system prompt, up to the pending round.
    messages: list[Message]
    steps: list[AgentStep]
    current_round: int
    usage: Usage
    pending: PendingRound
    version: int = field(default=SNAPSHOT_VERSION)
    # A derived model-input estimate; never grants identity, scope or approval.
    context_input_limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "caller": self.caller.to_dict(),
            "scope": self.scope.to_dict(),
            "policy": self.policy.to_dict(),
            "query": self.query,
            "memory_prompt": self.memory_prompt,
            "messages": [m.to_dict() for m in self.messages],
            "steps": [s.to_dict() for s in self.steps],
            "current_round": self.current_round,
            "usage": self.usage.to_dict(),
            "pending": self.pending.to_dict(),
            "context_input_limit": self.context_input_limit,
        }

    def encode(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def decode(cls, text: str) -> RunSnapshot:
        try:
            data = json.loads(text, parse_constant=_reject_constant)
        except (TypeError, ValueError) as exc:
            raise SnapshotError(f"unreadable run snapshot: {exc}") from exc
        if not isinstance(data, dict):
            raise SnapshotError("run snapshot must be a JSON object")
        version = data.get("version")
        if type(version) is not int or version not in (1, SNAPSHOT_VERSION):
            raise SnapshotError(f"unsupported run snapshot version {version!r}")
        try:
            validate_snapshot_data(data)
            snapshot = cls(
                run_id=data["run_id"],
                session_id=data["session_id"],
                caller=Caller.from_dict(data["caller"]),
                scope=ResourceScope.from_dict(data["scope"]),
                policy=ToolPolicy.from_dict(data["policy"]),
                query=data["query"],
                memory_prompt=data.get("memory_prompt", ""),
                messages=[Message.from_dict(m) for m in data["messages"]],
                steps=[AgentStep.from_dict(s) for s in data["steps"]],
                current_round=data["current_round"],
                usage=Usage.from_dict(data.get("usage", {})),
                pending=PendingRound.from_dict(data["pending"]),
                context_input_limit=data.get("context_input_limit"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotError(f"malformed run snapshot: {exc}") from exc
        snapshot.pending.validate()
        return snapshot

    def closed_steps(self, reason: str) -> list[AgentStep]:
        """Steps for a run that will never resume (prompt expired or canceled).

        Calls that never ran are recorded as failed with ``reason``, so the
        transcript keeps one result for every tool call the model issued.
        """
        pending = self.pending
        records = list(pending.done)
        for call in pending.tool_calls[len(records) :]:
            records.append(
                ToolCallRecord(call.id, call.name, call.arguments, ToolResult.fail(reason))
            )
        return [
            *self.steps,
            AgentStep(round=pending.round, thought=pending.thought, tool_calls=records),
        ]
