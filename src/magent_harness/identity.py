"""Who is running the agent and what the run may touch.

Identity always comes from the authenticated server request that starts or
resumes a run. Nothing the model emits (tool arguments, text) can select a
caller, a workspace, or a memory space.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class PermissionDenied(Exception):
    """An operation falls outside the caller's role or the run's resource scope."""


class Role(StrEnum):
    """The caller's role inside one workspace."""

    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"

    @property
    def level(self) -> int:
        return _ROLE_LEVEL[self]

    def has_permission(self, required: Role) -> bool:
        return self.level >= required.level


_ROLE_LEVEL = {Role.VIEWER: 10, Role.EDITOR: 20, Role.OWNER: 30}


class AccessLevel(StrEnum):
    """How far the assistant may act for the caller. It only narrows the role."""

    READ_ONLY = "read_only"  # no tool that changes data is offered or run
    CONFIRM = "confirm"  # every change waits for the user's approval
    AUTO_WRITE = "auto_write"  # recoverable changes run; destructive ones still ask


@dataclass(frozen=True)
class Caller:
    workspace_id: str
    user_id: str
    role: Role

    def __post_init__(self) -> None:
        # Fail closed: an empty identity must never reach authorization checks
        # where "" could compare equal to another empty field.
        if not self.workspace_id.strip() or not self.user_id.strip():
            raise ValueError("caller requires a workspace_id and a user_id")
        if not isinstance(self.role, Role):
            object.__setattr__(self, "role", Role(self.role))

    @property
    def subject_id(self) -> str:
        """The per-person key for isolated features such as memory."""
        return f"user:{self.user_id}"

    def to_dict(self) -> dict[str, Any]:
        return {"workspace_id": self.workspace_id, "user_id": self.user_id, "role": self.role.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Caller:
        return cls(data["workspace_id"], data["user_id"], Role(data["role"]))


@dataclass(frozen=True)
class ResourceScope:
    """The workspace content one run may read or change.

    ``document_ids`` of ``None`` means the whole workspace; a set limits the run
    to those documents (for example the ones the user @-mentioned).
    """

    workspace_id: str
    document_ids: frozenset[str] | None = None

    def allows_document(self, workspace_id: str, document_id: str) -> bool:
        if not document_id or workspace_id != self.workspace_id:
            return False
        return self.document_ids is None or document_id in self.document_ids

    def require_document(self, workspace_id: str, document_id: str) -> None:
        """Raise unless the document is inside the scope.

        Tools must call this with the workspace of the document as loaded from
        storage, not with an id the model supplied: an id alone proves nothing
        about where the document lives.
        """
        if not self.allows_document(workspace_id, document_id):
            raise PermissionDenied(f"document {document_id!r} is outside this run's scope")

    def to_dict(self) -> dict[str, Any]:
        ids = None if self.document_ids is None else sorted(self.document_ids)
        return {"workspace_id": self.workspace_id, "document_ids": ids}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceScope:
        ids = data.get("document_ids")
        return cls(data["workspace_id"], None if ids is None else frozenset(ids))
