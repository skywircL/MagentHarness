"""Human-in-the-loop prompts: tool approvals and questions for the user.

A run that needs the user never blocks waiting. It saves a snapshot and ends
as *suspended*; the prompt row is authoritative for human answers and claims.
An optional run journal records execution facts and reconciles crash windows.
The user's answer is recorded with a compare-and-set, and a
resume must *claim* the answer before acting on it, so an approved side effect
runs at most once even if two workers try to resume the same run.

Lifecycle::

    suspended ──resolve──▶ answered ──claim──▶ executing ──mark_executed──▶ done
        │                                 └──(rejected / ask_user)──────▶ done
        ├──▶ expired   (not answered in time)
        └──▶ canceled  (the user moved on)

A legacy prompt stuck in ``executing`` is never replayed: claiming requires
``answered``. Durable run recovery additionally distinguishes no tool start,
committed results, and mutations with an unknown outcome.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol

from .identity import Caller

DEFAULT_PROMPT_EXPIRY_S = 7 * 24 * 3600


class PromptKind(StrEnum):
    TOOL_APPROVAL = "tool_approval"
    ASK_USER = "ask_user"


class PromptStatus(StrEnum):
    SUSPENDED = "suspended"
    ANSWERED = "answered"
    EXECUTING = "executing"
    DONE = "done"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELED = "canceled"


class PromptError(Exception):
    pass


class PromptNotFound(PromptError):
    """Unknown prompt, or one outside the caller's workspace, user or session.

    Ownership mismatches deliberately read as not-found so callers cannot probe
    for prompts they do not own.
    """


class PromptAlreadyResolved(PromptError):
    pass


class PromptExpired(PromptAlreadyResolved):
    pass


class PromptNotAnswered(PromptError):
    """The run cannot resume yet: the user has not answered the prompt."""


class PromptClaimLost(PromptError):
    """Another worker already claimed this answer; this one must not act on it."""


class InvalidPromptResponse(PromptError):
    pass


@dataclass(frozen=True)
class AgentPrompt:
    id: str
    kind: PromptKind
    workspace_id: str
    user_id: str
    session_id: str
    run_id: str
    tool_call_id: str
    request: dict[str, Any]
    status: PromptStatus = PromptStatus.SUSPENDED
    snapshot: str = ""
    response: dict[str, Any] | None = None
    created_at: float = 0.0
    expires_at: float | None = None
    resolved_at: float | None = None
    resolved_by: str = ""
    close_reason: str = ""

    @property
    def approved(self) -> bool:
        return (
            self.kind is PromptKind.TOOL_APPROVAL
            and self.response is not None
            and self.response.get("approved") is True
        )

    def public_view(self) -> dict[str, Any]:
        """What the UI needs to render the prompt; the snapshot stays server-side."""
        return {
            "id": self.id,
            "kind": self.kind.value,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "tool_call_id": self.tool_call_id,
            "request": self.request,
            "status": self.status.value,
            "expires_at": self.expires_at,
        }


class PromptStore(Protocol):
    def create(self, prompt: AgentPrompt) -> None: ...

    def get(self, prompt_id: str) -> AgentPrompt | None: ...

    def transition(
        self, prompt_id: str, from_statuses: Iterable[PromptStatus], **changes: Any
    ) -> AgentPrompt | None:
        """Atomically apply ``changes`` if the row's status is in ``from_statuses``.

        Returns the updated row, or None when the row is missing or the status
        did not match (another writer won).
        """
        ...

    def list_by_status(self, status: PromptStatus) -> list[AgentPrompt]: ...


class InMemoryPromptStore:
    """Process-local store. A database implementation must keep the same CAS semantics."""

    def __init__(self) -> None:
        self._rows: dict[str, AgentPrompt] = {}
        self._lock = threading.Lock()

    def create(self, prompt: AgentPrompt) -> None:
        with self._lock:
            if prompt.id in self._rows:
                raise ValueError(f"prompt {prompt.id} already exists")
            self._rows[prompt.id] = prompt

    def get(self, prompt_id: str) -> AgentPrompt | None:
        with self._lock:
            return self._rows.get(prompt_id)

    def transition(
        self, prompt_id: str, from_statuses: Iterable[PromptStatus], **changes: Any
    ) -> AgentPrompt | None:
        allowed = set(from_statuses)
        with self._lock:
            row = self._rows.get(prompt_id)
            if row is None or row.status not in allowed:
                return None
            updated = replace(row, **changes)
            self._rows[prompt_id] = updated
            return updated

    def list_by_status(self, status: PromptStatus) -> list[AgentPrompt]:
        with self._lock:
            return [row for row in self._rows.values() if row.status is status]


def _validate_tool_approval(response: dict[str, Any]) -> None:
    unknown = set(response) - {"approved", "reason", "modified_args"}
    if unknown:
        raise InvalidPromptResponse(f"unknown fields: {sorted(unknown)}")
    # A missing or misspelled "approved" must not read as a rejection or approval.
    if not isinstance(response.get("approved"), bool):
        raise InvalidPromptResponse('"approved" must be true or false')
    if "reason" in response and not isinstance(response["reason"], str):
        raise InvalidPromptResponse('"reason" must be a string')
    if "modified_args" in response:
        if response["approved"] is not True:
            raise InvalidPromptResponse("modified_args is only allowed on approval")
        if not isinstance(response["modified_args"], dict):
            raise InvalidPromptResponse("modified_args must be a JSON object")


def _validate_ask_user(response: dict[str, Any]) -> None:
    unknown = set(response) - {"answer"}
    if unknown:
        raise InvalidPromptResponse(f"unknown fields: {sorted(unknown)}")
    answer = response.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise InvalidPromptResponse('"answer" must be a non-empty string')


_VALIDATORS: dict[PromptKind, Callable[[dict[str, Any]], None]] = {
    PromptKind.TOOL_APPROVAL: _validate_tool_approval,
    PromptKind.ASK_USER: _validate_ask_user,
}


@dataclass
class PromptBroker:
    store: PromptStore = field(default_factory=InMemoryPromptStore)
    expiry_s: float | None = DEFAULT_PROMPT_EXPIRY_S
    clock: Callable[[], float] = time.time

    def open(
        self,
        *,
        kind: PromptKind,
        caller: Caller,
        session_id: str,
        run_id: str,
        tool_call_id: str,
        request: dict[str, Any],
        snapshot: str,
        before_create: Callable[[AgentPrompt], None] | None = None,
    ) -> AgentPrompt:
        """Persist the prompt and snapshot; optionally journal its identity first.

        ``before_create`` must durably save the complete prompt before returning.
        The run recovery path can then repair a crash before ``store.create``.
        """
        now = self.clock()
        prompt = AgentPrompt(
            id=uuid.uuid4().hex,
            kind=kind,
            workspace_id=caller.workspace_id,
            user_id=caller.user_id,
            session_id=session_id,
            run_id=run_id,
            tool_call_id=tool_call_id,
            request=request,
            snapshot=snapshot,
            created_at=now,
            expires_at=None if self.expiry_s is None else now + self.expiry_s,
        )
        if before_create is not None:
            before_create(prompt)
        self.store.create(prompt)
        return prompt

    def get_owned(self, caller: Caller, prompt_id: str, session_id: str = "") -> AgentPrompt:
        row = self.store.get(prompt_id)
        if (
            row is None
            or row.workspace_id != caller.workspace_id
            or row.user_id != caller.user_id
            or (session_id and row.session_id != session_id)
        ):
            raise PromptNotFound(prompt_id)
        return row

    def resolve(
        self, caller: Caller, prompt_id: str, response: dict[str, Any], *, session_id: str = ""
    ) -> AgentPrompt:
        """Record the user's answer. The caller must own the prompt."""
        row = self.get_owned(caller, prompt_id, session_id)
        if row.status is not PromptStatus.SUSPENDED:
            raise PromptAlreadyResolved(prompt_id)
        now = self.clock()
        if row.expires_at is not None and row.expires_at <= now:
            self.store.transition(
                prompt_id,
                [PromptStatus.SUSPENDED],
                status=PromptStatus.EXPIRED,
                close_reason="the prompt expired before it was answered",
            )
            raise PromptExpired(prompt_id)
        _VALIDATORS[row.kind](response)
        updated = self.store.transition(
            prompt_id,
            [PromptStatus.SUSPENDED],
            status=PromptStatus.ANSWERED,
            response=dict(response),
            resolved_at=now,
            resolved_by=caller.user_id,
        )
        if updated is None:
            raise PromptAlreadyResolved(prompt_id)
        return updated

    def claim(self, prompt: AgentPrompt) -> AgentPrompt:
        """Take exclusive ownership of an answered prompt before acting on it."""
        to = PromptStatus.EXECUTING if prompt.approved else PromptStatus.DONE
        claimed = self.store.transition(prompt.id, [PromptStatus.ANSWERED], status=to)
        if claimed is None:
            raise PromptClaimLost(prompt.id)
        return claimed

    def mark_executed(self, prompt_id: str) -> None:
        self.store.transition(prompt_id, [PromptStatus.EXECUTING], status=PromptStatus.DONE)

    def fail(self, prompt_id: str, reason: str) -> None:
        """Close an answered prompt whose run cannot continue (e.g. unreadable snapshot)."""
        self.store.transition(
            prompt_id,
            [PromptStatus.ANSWERED, PromptStatus.EXECUTING],
            status=PromptStatus.FAILED,
            close_reason=reason,
        )

    def cancel(self, caller: Caller, prompt_id: str, reason: str) -> AgentPrompt:
        """Close a prompt the user will not answer (for example they moved on)."""
        self.get_owned(caller, prompt_id)
        updated = self.store.transition(
            prompt_id,
            [PromptStatus.SUSPENDED],
            status=PromptStatus.CANCELED,
            close_reason=reason,
        )
        if updated is None:
            raise PromptAlreadyResolved(prompt_id)
        return updated

    def expire_due(self) -> list[AgentPrompt]:
        """Sweep suspended prompts past their expiry. Safe to run concurrently."""
        now = self.clock()
        expired: list[AgentPrompt] = []
        for row in self.store.list_by_status(PromptStatus.SUSPENDED):
            if row.expires_at is None or row.expires_at > now:
                continue
            updated = self.store.transition(
                row.id,
                [PromptStatus.SUSPENDED],
                status=PromptStatus.EXPIRED,
                close_reason="the prompt expired before it was answered",
            )
            if updated is not None:
                expired.append(updated)
        return expired
