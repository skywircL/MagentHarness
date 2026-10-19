"""Identity and permissions: role, access level, tool allowlist and resource scope
are enforced where calls execute, not only in what the model is offered."""

from __future__ import annotations

import pytest
from support import EDITOR, OTHER_WORKSPACE, VIEWER, build, run

from magent_harness import (
    AccessLevel,
    Caller,
    PermissionDenied,
    ResourceScope,
    Risk,
    Role,
    RunRequest,
    RunStatus,
    ToolPolicy,
    ToolSpec,
    Verdict,
)
from magent_harness.testing import call_tools, reply, tool_call


def request(caller: Caller = EDITOR, **kwargs: object) -> RunRequest:
    return RunRequest(caller=caller, session_id="s1", query="do it", **kwargs)  # type: ignore[arg-type]


def update_call(doc: str = "d1", call_id: str = "c1") -> object:
    return tool_call("update_document", {"document_id": doc, "content": "new"}, call_id)


@pytest.mark.parametrize(
    ("role", "access", "risk", "expected"),
    [
        (Role.VIEWER, AccessLevel.AUTO_WRITE, Risk.READ, Verdict.ALLOW),
        (Role.VIEWER, AccessLevel.AUTO_WRITE, Risk.WRITE, Verdict.DENY),
        (Role.EDITOR, AccessLevel.READ_ONLY, Risk.READ, Verdict.ALLOW),
        (Role.EDITOR, AccessLevel.READ_ONLY, Risk.WRITE, Verdict.DENY),
        (Role.EDITOR, AccessLevel.CONFIRM, Risk.WRITE, Verdict.REQUIRE_APPROVAL),
        (Role.EDITOR, AccessLevel.AUTO_WRITE, Risk.WRITE, Verdict.ALLOW),
        (Role.EDITOR, AccessLevel.AUTO_WRITE, Risk.DESTRUCTIVE, Verdict.REQUIRE_APPROVAL),
        (Role.OWNER, AccessLevel.CONFIRM, Risk.DESTRUCTIVE, Verdict.REQUIRE_APPROVAL),
    ],
)
def test_policy_matrix(role: Role, access: AccessLevel, risk: Risk, expected: Verdict) -> None:
    spec = ToolSpec("t", "T.", risk=risk)
    decision = ToolPolicy(access).evaluate(Caller("ws1", "u", role), spec)
    assert decision.verdict is expected


def test_min_role_and_always_confirm_override_defaults() -> None:
    owner_only = ToolSpec("export_all", "Export.", min_role=Role.OWNER)
    assert ToolPolicy().evaluate(EDITOR, owner_only).verdict is Verdict.DENY
    sensitive_read = ToolSpec("read_secrets", "Read.", always_confirm=True)
    policy = ToolPolicy(AccessLevel.AUTO_WRITE)
    assert policy.evaluate(EDITOR, sensitive_read).verdict is Verdict.REQUIRE_APPROVAL


def test_viewer_is_not_offered_writes_and_cannot_run_them() -> None:
    # The model calls update_document even though it was never offered.
    h = build([call_tools(update_call()), reply("ok")])
    result = run(h.engine.run(request(VIEWER, policy=ToolPolicy(AccessLevel.AUTO_WRITE))))

    assert "update_document" not in h.model.calls[0].tool_names
    assert "read_document" in h.model.calls[0].tool_names
    error = result.steps[0].tool_calls[0].result.error
    assert error.startswith("not permitted") and "requires editor" in error
    assert h.workspace.writes == []
    assert result.prompt is None


def test_read_only_access_withholds_and_denies_mutations() -> None:
    h = build([call_tools(update_call()), reply("ok")])
    result = run(h.engine.run(request(policy=ToolPolicy(AccessLevel.READ_ONLY))))
    offered = h.model.calls[0].tool_names
    assert "update_document" not in offered and "delete_document" not in offered
    assert "read-only" in result.steps[0].tool_calls[0].result.error
    assert h.workspace.writes == []


def test_allowed_tools_limits_offer_and_execution() -> None:
    policy = ToolPolicy(AccessLevel.AUTO_WRITE, allowed_tools=frozenset({"read_document"}))
    h = build([call_tools(update_call()), reply("ok")])
    result = run(h.engine.run(request(policy=policy)))
    assert set(h.model.calls[0].tool_names) == {"read_document", "ask_user"}
    assert "not enabled for this run" in result.steps[0].tool_calls[0].result.error
    assert h.workspace.writes == []


def test_scope_limits_documents_even_inside_the_workspace() -> None:
    scope = ResourceScope("ws1", frozenset({"d1"}))
    h = build(
        [
            call_tools(
                tool_call("read_document", {"document_id": "d1"}, "c1"),
                tool_call("read_document", {"document_id": "d2"}, "c2"),
            ),
            reply("ok"),
        ]
    )
    result = run(h.engine.run(request(scope=scope)))
    ok, denied = result.steps[0].tool_calls
    assert ok.result.success
    assert denied.result.error == "permission denied: document 'd2' is outside this run's scope"
    assert h.workspace.reads == ["d1"]


def test_documents_of_another_workspace_are_rejected() -> None:
    h = build([call_tools(tool_call("read_document", {"document_id": "d9"})), reply("ok")])
    result = run(h.engine.run(request()))
    assert "outside this run's scope" in result.steps[0].tool_calls[0].result.error
    assert h.workspace.reads == []


def test_search_only_returns_documents_in_scope() -> None:
    scope = ResourceScope("ws1", frozenset({"d2"}))
    h = build([call_tools(tool_call("search_documents", {"query": "search"})), reply("ok")])
    result = run(h.engine.run(request(scope=scope)))
    assert result.steps[0].tool_calls[0].result.output == "d2: Notes"


def test_run_scope_must_belong_to_the_caller_workspace() -> None:
    h = build([reply("never")])
    with pytest.raises(PermissionDenied):
        run(h.engine.run(request(scope=ResourceScope("ws2"))))
    with pytest.raises(PermissionDenied):
        run(h.engine.run(request(OTHER_WORKSPACE, scope=ResourceScope("ws1"))))
    assert h.model.calls == []


def test_caller_identity_is_required() -> None:
    with pytest.raises(ValueError):
        Caller("ws1", " ", Role.EDITOR)
    with pytest.raises(ValueError):
        Caller("", "alice", Role.EDITOR)
    with pytest.raises(ValueError):
        Caller("ws1", "alice", "admin")  # type: ignore[arg-type]


def test_scope_round_trips_and_checks_membership() -> None:
    scope = ResourceScope("ws1", frozenset({"a", "b"}))
    assert ResourceScope.from_dict(scope.to_dict()) == scope
    assert scope.allows_document("ws1", "a")
    assert not scope.allows_document("ws2", "a")
    assert not scope.allows_document("ws1", "")
    assert ResourceScope("ws1").allows_document("ws1", "anything")


def test_run_completes_normally_for_permitted_writes() -> None:
    h = build([call_tools(update_call()), reply("updated")])
    result = run(h.engine.run(request(policy=ToolPolicy(AccessLevel.AUTO_WRITE))))
    assert result.status is RunStatus.COMPLETED
    assert h.workspace.writes == [("d1", "new")]
