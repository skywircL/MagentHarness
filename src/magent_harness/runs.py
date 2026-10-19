"""Durable run facts and their rebuildable projection.

The JSONL journal is authoritative. Checkpoints cache the projection, never
authorize execution. A run's file lock is held across model and tool awaits.
"""

from __future__ import annotations

import json
import logging
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .file_store import (
    CorruptJournalError,
    FileJournal,
    JsonlPromptStore,
    PersistenceError,
    prompt_from_dict,
    prompt_to_dict,
)
from .hitl import AgentPrompt, PromptKind, PromptStatus, PromptStore
from .identity import Caller, ResourceScope
from .messages import AgentStep, Message, ToolCallRecord, Usage
from .policy import Risk, ToolPolicy
from .snapshot import (
    PendingRound,
    RunSnapshot,
    SnapshotError,
    validate_snapshot_data,
    validate_tool_record,
)

logger = logging.getLogger(__name__)
RUN_STATE_VERSION = 1


class RunJournal(Protocol):
    """Committed facts under an exclusive local lock or backend execution lease."""

    @property
    def records(self) -> list[dict[str, Any]]: ...

    def append(self, kind: str, data: dict[str, Any]) -> dict[str, Any]: ...

    def read_checkpoint(self) -> dict[str, Any] | None: ...

    def write_checkpoint(self, data: dict[str, Any]) -> None: ...


class RunStore(Protocol):
    """Applications own authentication, durable answers and execution fencing."""

    @property
    def prompts(self) -> PromptStore: ...

    def journal(self, caller: Caller, run_id: str) -> AbstractContextManager[RunJournal]: ...


class RunNotFound(PersistenceError):
    """Missing run or a run outside the authenticated caller/session."""


class RunAlreadyExists(PersistenceError):
    """Use recover rather than starting the same durable run again."""


class RecoveryUnsafe(PersistenceError):
    """A mutation may have happened. Reconcile it externally before proceeding."""


def validate_prompt_snapshot(prompt: AgentPrompt, snapshot: RunSnapshot) -> None:
    """Bind the restored operation to what the human was actually asked to approve."""
    call = snapshot.pending.tool_calls[snapshot.pending.blocked_index]
    if (
        call.id != prompt.tool_call_id
        or snapshot.run_id != prompt.run_id
        or snapshot.session_id != prompt.session_id
        or snapshot.caller.workspace_id != prompt.workspace_id
        or snapshot.caller.user_id != prompt.user_id
        or snapshot.scope.workspace_id != prompt.workspace_id
    ):
        raise SnapshotError("prompt and run snapshot do not match")
    try:
        args = json.loads(call.arguments or "{}")
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be an object")
        if prompt.kind is PromptKind.TOOL_APPROVAL:
            if (
                call.name == "ask_user"
                or prompt.request["tool"] != call.name
                or _json_value(prompt.request["arguments"]) != _json_value(args)
            ):
                raise ValueError("approval request differs from its tool call")
            Risk(prompt.request["risk"])
        elif call.name != "ask_user" or _json_value(prompt.request) != _json_value(
            {"question": args["question"], "options": args.get("options", [])}
        ):
            raise ValueError("question request differs from its tool call")
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotError("prompt request does not match the snapshot operation") from exc


def _json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


@dataclass
class RunState:
    run_id: str
    session_id: str
    caller: Caller
    scope: ResourceScope
    policy: ToolPolicy
    query: str
    messages: list[Message]
    steps: list[AgentStep]
    current_round: int
    usage: Usage
    empty_retries: int = 0
    length_rounds: int = 0
    overflow_recovered: bool = False
    pending: PendingRound | None = None
    started: dict[str, dict[str, Any]] = field(default_factory=dict)
    finished: dict[str, ToolCallRecord] = field(default_factory=dict)
    prompt: AgentPrompt | None = None
    resuming: bool = False
    status: str = "running"
    answer: str = ""
    stop_reason: str = ""
    error: str = ""
    truncated: bool = False
    context_input_limit: int | None = None

    @classmethod
    def from_core(cls, data: dict[str, Any]) -> RunState:
        if type(data.get("version")) is not int or data["version"] != RUN_STATE_VERSION:
            raise SnapshotError("unsupported durable run state version")
        validate_snapshot_data(data, pending_required=False)
        for key in ("empty_retries", "length_rounds"):
            value = data.get(key, 0)
            if type(value) is not int or value < 0:
                raise SnapshotError(f"invalid {key}")
        if type(data.get("overflow_recovered", False)) is not bool:
            raise SnapshotError("invalid overflow_recovered")
        pending = data.get("pending")
        state = cls(
            run_id=data["run_id"],
            session_id=data["session_id"],
            caller=Caller.from_dict(data["caller"]),
            scope=ResourceScope.from_dict(data["scope"]),
            policy=ToolPolicy.from_dict(data["policy"]),
            query=data["query"],
            messages=[Message.from_dict(m) for m in data["messages"]],
            steps=[AgentStep.from_dict(s) for s in data["steps"]],
            current_round=data["current_round"],
            usage=Usage.from_dict(data["usage"]),
            empty_retries=data.get("empty_retries", 0),
            length_rounds=data.get("length_rounds", 0),
            overflow_recovered=data.get("overflow_recovered", False),
            pending=None if pending is None else PendingRound.from_dict(pending),
            context_input_limit=data.get("context_input_limit"),
        )
        if state.scope.workspace_id != state.caller.workspace_id:
            raise SnapshotError("run scope is outside its owner's workspace")
        if state.pending is not None:
            state.finished = {r.id: r for r in state.pending.done}
        return state

    def core(self) -> dict[str, Any]:
        return {
            "version": RUN_STATE_VERSION,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "caller": self.caller.to_dict(),
            "scope": self.scope.to_dict(),
            "policy": self.policy.to_dict(),
            "query": self.query,
            "memory_prompt": "",
            "messages": [m.to_dict() for m in self.messages],
            "steps": [s.to_dict() for s in self.steps],
            "current_round": self.current_round,
            "usage": self.usage.to_dict(),
            "empty_retries": self.empty_retries,
            "length_rounds": self.length_rounds,
            "overflow_recovered": self.overflow_recovered,
            "pending": None if self.pending is None else self.pending.to_dict(),
            "context_input_limit": self.context_input_limit,
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            **self.core(),
            "started": self.started,
            "finished": {key: value.to_dict() for key, value in self.finished.items()},
            "prompt": None if self.prompt is None else prompt_to_dict(self.prompt),
            "resuming": self.resuming,
            "status": self.status,
            "answer": self.answer,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "truncated": self.truncated,
        }


def _same_owner(before: RunState, after: RunState) -> bool:
    return (
        before.run_id,
        before.session_id,
        before.caller.workspace_id,
        before.caller.user_id,
        before.scope,
        before.policy,
        before.query,
    ) == (
        after.run_id,
        after.session_id,
        after.caller.workspace_id,
        after.caller.user_id,
        after.scope,
        after.policy,
        after.query,
    )


def _apply(state: RunState | None, kind: str, data: dict[str, Any]) -> RunState:
    if kind == "created":
        if state is not None:
            raise SnapshotError("duplicate run creation")
        created = RunState.from_core(data["state"])
        if "prompt" in data:
            return _apply(created, "suspended", {"prompt": data["prompt"]})
        return created
    if state is None:
        raise SnapshotError("run journal has no creation record")
    if state.status in {"completed", "failed", "canceled"}:
        raise SnapshotError("record after terminal run state")
    if kind in {"progress", "round_committed", "completed", "failed", "canceled"}:
        if kind == "progress" and (state.pending is not None or state.status != "running"):
            raise SnapshotError("progress cannot replace unfinished tool facts")
        updated = RunState.from_core(data["state"])
        if not _same_owner(state, updated):
            raise SnapshotError("run ownership, query, scope or policy changed")
        if kind == "round_committed":
            if state.pending is None or len(state.finished) != len(state.pending.tool_calls):
                raise SnapshotError("committing a round without all tool results")
            if updated.current_round != state.current_round + 1:
                raise SnapshotError("invalid committed round number")
            records = [state.finished[call.id] for call in state.pending.tool_calls]
            expected_messages = [
                *state.messages,
                Message(
                    role="assistant",
                    content=state.pending.thought,
                    tool_calls=tuple(state.pending.tool_calls),
                    reasoning_content=state.pending.reasoning_content,
                ),
                *(record.to_message() for record in records),
            ]
            expected_steps = [
                *state.steps,
                AgentStep(state.pending.round, state.pending.thought, records),
            ]
            if updated.messages != expected_messages or updated.steps != expected_steps:
                raise SnapshotError("round commit differs from the authoritative plan/results")
        elif updated.current_round != state.current_round:
            raise SnapshotError("round changed outside round commit")
        if kind in {"completed", "failed", "canceled"}:
            if kind == "completed" and state.pending is not None:
                raise SnapshotError("cannot complete with an unfinished round")
            updated.status = kind
            for key in ("answer", "stop_reason", "error"):
                if not isinstance(data.get(key, ""), str):
                    raise SnapshotError(f"invalid terminal {key}")
            updated.answer = data.get("answer", "")
            updated.stop_reason = data.get("stop_reason", "")
            updated.error = data.get("error", "")
        return updated
    if kind == "context_committed":
        if state.pending is not None:
            raise SnapshotError("compaction inside an unfinished round")
        checked = RunState.from_core(
            {
                **state.core(),
                "messages": data["messages"],
                "context_input_limit": data.get("context_input_limit", state.context_input_limit),
            }
        )
        state.messages = checked.messages
        state.context_input_limit = checked.context_input_limit
    elif kind == "planned":
        if state.pending is not None:
            raise SnapshotError("new plan inside an unfinished round")
        checked = RunState.from_core(
            {
                **state.core(),
                "pending": data["pending"],
                "usage": data["usage"],
                "empty_retries": data["empty_retries"],
                "length_rounds": data["length_rounds"],
            }
        )
        state.pending = checked.pending
        assert state.pending is not None
        if state.pending.round != state.current_round or state.pending.done:
            raise SnapshotError("invalid new tool plan")
        state.usage = checked.usage
        state.empty_retries = checked.empty_retries
        state.length_rounds = checked.length_rounds
        state.truncated = data.get("truncated", False)
        if type(state.truncated) is not bool:
            raise SnapshotError("invalid truncated plan flag")
        state.started, state.finished = {}, {}
    elif kind in {"tool_started", "tool_finished"}:
        if state.pending is None:
            raise SnapshotError("tool fact outside a planned round")
        calls = {call.id: call for call in state.pending.tool_calls}
        if kind == "tool_started":
            call = calls.get(data["id"])
            if (
                call is None
                or call.name != data["name"]
                or not isinstance(data["arguments"], str)
                or type(data["read_only"]) is not bool
                or not isinstance(data.get("approval_id", ""), str)
                or call.id in state.finished
                or state.truncated
            ):
                raise SnapshotError("invalid tool start identity")
            previous = state.started.get(call.id)
            if previous is not None and (
                not previous["read_only"] or not data["read_only"] or previous != data
            ):
                raise SnapshotError("cannot retry an uncertain mutation")
            state.started[call.id] = data
        else:
            raw = data["record"]
            record = validate_tool_record(raw)
            call = calls.get(record.id)
            if call is None or record.name != call.name or record.id in state.finished:
                raise SnapshotError("invalid tool result identity")
            validate_tool_record(raw, call)
            started = state.started.get(record.id)
            if started is not None and started["arguments"] != record.arguments:
                raise SnapshotError("tool result differs from the started operation")
            if started is None and record.result.success and call.name != "ask_user":
                raise SnapshotError("successful tool result without a committed start")
            state.finished[record.id] = record
    elif kind in {"suspended", "resume_requested"}:
        prompt = prompt_from_dict(data["prompt"])
        if (
            state.pending is None
            or prompt.run_id != state.run_id
            or prompt.session_id != state.session_id
            or prompt.workspace_id != state.caller.workspace_id
            or prompt.user_id != state.caller.user_id
            or prompt.tool_call_id not in {c.id for c in state.pending.tool_calls}
        ):
            raise SnapshotError("prompt does not belong to this run")
        snapshot = RunSnapshot.decode(prompt.snapshot)
        validate_prompt_snapshot(prompt, snapshot)
        if (
            snapshot.run_id != state.run_id
            or snapshot.session_id != state.session_id
            or snapshot.caller.workspace_id != state.caller.workspace_id
            or snapshot.caller.user_id != state.caller.user_id
            or snapshot.scope != state.scope
            or snapshot.policy != state.policy
            or snapshot.messages != state.messages
            or snapshot.steps != state.steps
            or snapshot.pending.tool_calls != state.pending.tool_calls
            or snapshot.pending.thought != state.pending.thought
            or snapshot.pending.reasoning_content != state.pending.reasoning_content
            or snapshot.pending.tool_calls[snapshot.pending.blocked_index].id != prompt.tool_call_id
            or snapshot.current_round != state.current_round
            or snapshot.query != state.query
        ):
            raise SnapshotError("prompt snapshot differs from the authoritative run state")
        for record in snapshot.pending.done:
            if state.finished.get(record.id) != record:
                raise SnapshotError("prompt frontier differs from committed tool results")
        if len(snapshot.pending.done) != len(state.finished):
            raise SnapshotError("prompt frontier omits completed tool results")
        if kind == "resume_requested":
            if (
                state.prompt is None
                or state.prompt.id != prompt.id
                or state.resuming
                or state.status != "suspended"
                or prompt.status is not PromptStatus.ANSWERED
                or any(
                    getattr(state.prompt, key) != getattr(prompt, key)
                    for key in (
                        "kind",
                        "snapshot",
                        "request",
                        "tool_call_id",
                        "created_at",
                        "expires_at",
                    )
                )
            ):
                raise SnapshotError("resume without a matching suspended prompt")
            state.resuming, state.status = True, "running"
        else:
            if prompt.status is not PromptStatus.SUSPENDED:
                raise SnapshotError("suspension needs an unanswered prompt")
            if state.resuming and state.prompt is not None and state.prompt.id == prompt.id:
                raise SnapshotError("duplicate suspended prompt")
            state.resuming, state.status = False, "suspended"
        state.prompt = prompt
        state.pending = snapshot.pending
    else:
        raise SnapshotError(f"unknown run record kind {kind!r}")
    return state


class RunLog:
    def __init__(self, journal: RunJournal) -> None:
        self.journal = journal
        self.state: RunState | None = None
        try:
            for record in journal.records:
                self.state = _apply(self.state, record["kind"], record["data"])
        except (KeyError, TypeError, ValueError, SnapshotError) as exc:
            raise CorruptJournalError("invalid durable run facts") from exc
        if self.state is not None:
            checkpoint = self.state.checkpoint()
            if journal.read_checkpoint() != checkpoint:
                self._checkpoint(checkpoint)

    def append(self, kind: str, data: dict[str, Any]) -> None:
        # Validate on a detached projection before committing the authority.
        current = None if self.state is None else self._copy_state()
        try:
            updated = _apply(current, kind, data)
        except (KeyError, TypeError, ValueError, SnapshotError) as exc:
            raise PersistenceError(f"invalid run transition {kind}") from exc
        self.journal.append(kind, data)
        self.state = updated
        self._checkpoint(updated.checkpoint())

    def _copy_state(self) -> RunState:
        assert self.state is not None
        state = RunState.from_core(self.state.core())
        state.started = json.loads(json.dumps(self.state.started))
        state.finished = {
            key: ToolCallRecord.from_dict(value.to_dict())
            for key, value in self.state.finished.items()
        }
        state.prompt = self.state.prompt
        state.resuming, state.status = self.state.resuming, self.state.status
        state.truncated = self.state.truncated
        state.answer, state.stop_reason, state.error = (
            self.state.answer,
            self.state.stop_reason,
            self.state.error,
        )
        return state

    def _checkpoint(self, data: dict[str, Any]) -> None:
        try:
            self.journal.write_checkpoint(data)
        except (OSError, PersistenceError):
            logger.warning("run checkpoint unavailable; JSONL replay remains authoritative")


class JsonlRunStore:
    """Local POSIX JSONL runs, with a durable default PromptStore at the same root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.prompts = JsonlPromptStore(self.root)

    def journal(self, caller: Caller, run_id: str) -> FileJournal:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        key = json.dumps([caller.workspace_id, caller.user_id, run_id], ensure_ascii=False)
        return FileJournal(self.root, "runs", key, blocking=False)

    @staticmethod
    def require_owned(
        state: RunState | None, caller: Caller, run_id: str, session_id: str
    ) -> RunState:
        if state is None or (
            state.run_id != run_id
            or state.caller.workspace_id != caller.workspace_id
            or state.caller.user_id != caller.user_id
            or (session_id and state.session_id != session_id)
        ):
            raise RunNotFound(run_id)
        return state
