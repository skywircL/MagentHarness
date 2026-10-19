"""Tool permission policy: decides allow / ask the user / deny for each call.

The same decision is used twice: to choose which tools the model is offered,
and again when a call executes. The second check is the one that matters — a
model can name a tool it was never offered.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from .identity import AccessLevel, Caller, Role

if TYPE_CHECKING:
    from .tools import ToolSpec


class Risk(StrEnum):
    READ = "read"  # no side effects
    WRITE = "write"  # changes data in a recoverable way (a new document version)
    DESTRUCTIVE = "destructive"  # removes data or cannot be undone

    @property
    def mutates(self) -> bool:
        return self is not Risk.READ

    @property
    def default_role(self) -> Role:
        return Role.VIEWER if self is Risk.READ else Role.EDITOR


class Verdict(StrEnum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reason: str = ""


@dataclass(frozen=True)
class ToolPolicy:
    """What one run may do. Owned by the server, never by the model."""

    access: AccessLevel = AccessLevel.CONFIRM
    # None offers every registered tool; a set (even empty) is taken as given.
    allowed_tools: frozenset[str] | None = None

    def evaluate(self, caller: Caller, spec: ToolSpec) -> Decision:
        if self.allowed_tools is not None and spec.name not in self.allowed_tools:
            return Decision(Verdict.DENY, f"tool {spec.name} is not enabled for this run")
        required = spec.required_role
        if not caller.role.has_permission(required):
            return Decision(
                Verdict.DENY,
                f"role {caller.role.value} cannot use {spec.name} (requires {required.value})",
            )
        if self.access is AccessLevel.READ_ONLY and spec.risk.mutates:
            return Decision(Verdict.DENY, "the assistant is set to read-only for this run")
        if spec.always_confirm:
            return Decision(Verdict.REQUIRE_APPROVAL, f"{spec.name} always needs confirmation")
        if spec.risk is Risk.READ:
            return Decision(Verdict.ALLOW)
        if spec.risk is Risk.WRITE and self.access is AccessLevel.AUTO_WRITE:
            return Decision(Verdict.ALLOW)
        return Decision(Verdict.REQUIRE_APPROVAL, f"{spec.name} changes data ({spec.risk.value})")

    def offered(self, caller: Caller, spec: ToolSpec) -> bool:
        return self.evaluate(caller, spec).verdict is not Verdict.DENY

    def to_dict(self) -> dict[str, Any]:
        allowed = None if self.allowed_tools is None else sorted(self.allowed_tools)
        return {"access": self.access.value, "allowed_tools": allowed}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolPolicy:
        allowed = data.get("allowed_tools")
        return cls(AccessLevel(data["access"]), None if allowed is None else frozenset(allowed))
