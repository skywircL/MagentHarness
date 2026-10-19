"""Memory revocation must hold at the actual model and recovery boundaries."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from support import EDITOR, VIEWER, FakeWorkspace, build, run

from magent_harness import (
    AgentConfig,
    AgentEngine,
    ContextSettings,
    EventType,
    FunctionTool,
    InMemoryMemoryStore,
    JsonlMemoryStore,
    JsonlRunStore,
    MemoryKind,
    MemoryService,
    Message,
    RunRequest,
    RunSnapshot,
    RunStatus,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    TransientModelError,
    search_memory_tool,
)
from magent_harness.memory import (
    MemoryForgottenError,
    MemoryItem,
    MemoryScope,
    MemoryStatus,
)
from magent_harness.testing import ScriptedModel, call_tools, reply, tool_call

SECRET = "unique obsolete cobalt memory"
CURRENT = "current jade memory"


@pytest.fixture(params=["memory", "jsonl"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    return InMemoryMemoryStore() if request.param == "memory" else JsonlMemoryStore(tmp_path)


@pytest.mark.parametrize("revoke", ["forget", "supersede"])
def test_revocation_during_usage_touch_excludes_the_current_recall(revoke: str) -> None:
    class RevokeOnTouch(InMemoryMemoryStore):
        revoked = False

        def touch(self, scope: MemoryScope, item_ids: Any, now: float) -> None:
            if not self.revoked:
                self.revoked = True
                if revoke == "forget":
                    assert service.forget(EDITOR, original.id)
                else:
                    service.remember(EDITOR, MemoryKind.PROFILE, CURRENT, topic="color")
            super().touch(scope, item_ids, now)

    store = RevokeOnTouch()
    service = MemoryService(store)
    original = service.remember(EDITOR, MemoryKind.PROFILE, SECRET, topic="color")
    recalled = service.recall(EDITOR, "color")

    assert store.revoked
    assert SECRET not in recalled.prompt
    assert original.id not in {item.id for item in recalled.items}


@pytest.mark.parametrize("change", ["body", "archived", "expired"])
def test_cached_candidates_are_resolved_against_authoritative_current_rows(change: str) -> None:
    class CachedCandidates(InMemoryMemoryStore):
        cached: list[MemoryItem] | None = None
        authoritative_reads = 0

        def list_items(
            self, scope: MemoryScope, status: MemoryStatus = MemoryStatus.ACTIVE
        ) -> list[MemoryItem]:
            if status is MemoryStatus.ACTIVE and self.cached is not None:
                return list(self.cached)
            return super().list_items(scope, status)

        def get_items(self, scope: MemoryScope, item_ids: Any) -> list[MemoryItem]:
            self.authoritative_reads += 1
            return super().get_items(scope, item_ids)

    store = CachedCandidates()
    service = MemoryService(store, clock=lambda: 100.0)
    item = service.remember(EDITOR, MemoryKind.PROFILE, SECRET)
    store.cached = [item]
    updated = {
        "body": replace(item, content=CURRENT),
        "archived": replace(item, status=MemoryStatus.ARCHIVED),
        "expired": replace(item, expires_at=99.0),
    }[change]
    store.update(MemoryScope.for_caller(EDITOR), updated)

    recalled = service.recall(EDITOR, "color")
    assert store.authoritative_reads > 0
    assert SECRET not in recalled.prompt
    if change == "body":
        assert CURRENT in recalled.prompt
    else:
        assert recalled.items == ()


def test_authoritative_lookup_failure_never_uses_cached_content() -> None:
    class UnavailableAuthority(InMemoryMemoryStore):
        def get_items(self, scope: MemoryScope, item_ids: Any) -> list[MemoryItem]:
            raise ConnectionError("authoritative store unavailable")

    service = MemoryService(UnavailableAuthority())
    service.remember(EDITOR, MemoryKind.PROFILE, SECRET)
    h = build([reply("ordinary answer")], memory=service)
    result = run(h.engine.run(RunRequest(EDITOR, "s1", "color")))

    assert result.status is RunStatus.COMPLETED
    assert all(SECRET not in message.content for message in h.model.calls[0].messages)


def test_legacy_store_without_authoritative_lookup_fails_closed() -> None:
    class LegacyStore:
        def __init__(self) -> None:
            self.delegate = InMemoryMemoryStore()

        def add(self, *args: Any) -> None:
            self.delegate.add(*args)

        def update(self, *args: Any) -> None:
            self.delegate.update(*args)

        def list_items(self, *args: Any) -> list[MemoryItem]:
            return self.delegate.list_items(*args)

    service = MemoryService(LegacyStore())
    service.remember(EDITOR, MemoryKind.PROFILE, SECRET)
    assert service.recall(EDITOR, "color").prompt == ""


def test_forgotten_content_cannot_be_recreated_under_a_new_id_in_its_scope(store: Any) -> None:
    service = MemoryService(store)
    original = service.remember(EDITOR, MemoryKind.FACT, "unique cobalt fact", topic="color")
    assert service.forget(EDITOR, original.id)

    with pytest.raises(MemoryForgottenError):
        service.remember(EDITOR, MemoryKind.FACT, " unique  cobalt\nfact ", topic="new topic")
    other = service.remember(VIEWER, MemoryKind.FACT, "unique cobalt fact", topic="color")
    assert other.id != original.id
    assert "unique cobalt fact" in service.recall(VIEWER, "cobalt").prompt
    assert service.recall(EDITOR, "cobalt").prompt == ""


@pytest.mark.parametrize("method", ["add", "update", "add_with_supersession"])
def test_content_tombstone_is_enforced_at_each_store_write_entry(store: Any, method: str) -> None:
    service = MemoryService(store)
    original = service.remember(EDITOR, MemoryKind.FACT, "unique cobalt fact")
    other = service.remember(EDITOR, MemoryKind.FACT, "different live fact")
    assert service.forget(EDITOR, original.id)
    new_id = replace(original, id=other.id if method == "update" else "different-memory-id")

    with pytest.raises(MemoryForgottenError):
        getattr(store, method)(MemoryScope.for_caller(EDITOR), new_id)
    assert [item.id for item in store.list_items(MemoryScope.for_caller(EDITOR))] == [other.id]


def test_jsonl_content_tombstone_survives_reopening_the_store(tmp_path: Path) -> None:
    first = MemoryService(JsonlMemoryStore(tmp_path))
    original = first.remember(EDITOR, MemoryKind.FACT, "unique cobalt fact")
    assert first.forget(EDITOR, original.id)
    fresh = MemoryService(JsonlMemoryStore(tmp_path))

    with pytest.raises(MemoryForgottenError):
        fresh.remember(EDITOR, MemoryKind.FACT, "unique cobalt fact")
    assert fresh.recall(EDITOR, "cobalt").prompt == ""


def test_forgetting_one_id_revokes_existing_duplicate_content_in_the_same_scope(store: Any) -> None:
    service = MemoryService(store)
    first = service.remember(EDITOR, MemoryKind.FACT, "unique cobalt fact", topic="first topic")
    duplicate = service.remember(
        EDITOR, MemoryKind.FACT, "unique cobalt fact", topic="second topic"
    )
    assert first.id != duplicate.id
    assert service.forget(EDITOR, first.id)

    assert service.search(EDITOR, "cobalt") == []
    assert service.recall(EDITOR, "cobalt").prompt == ""


def test_supersession_does_not_block_explicitly_returning_to_an_older_fact(store: Any) -> None:
    service = MemoryService(store)
    old = service.remember(EDITOR, MemoryKind.FACT, "Uses MySQL", topic="database")
    service.remember(EDITOR, MemoryKind.FACT, "Uses Postgres", topic="database")
    latest = service.remember(EDITOR, MemoryKind.FACT, "Uses MySQL", topic="database")

    assert latest.id != old.id
    assert [item.id for item in store.list_items(MemoryScope.for_caller(EDITOR))] == [latest.id]


def _revoking_registry(
    service: MemoryService, item: MemoryItem, *, expires: Any = None
) -> ToolRegistry:
    async def revoke(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if expires is None:
            assert service.forget(ctx.caller, item.id)
        else:
            expires[0] = 200.0
        return ToolResult.ok("memory state changed")

    registry = ToolRegistry()
    registry.register(
        FunctionTool(ToolSpec("change_memory", "Test state change.", {"type": "object"}), revoke)
    )
    return registry


@pytest.mark.parametrize("max_iterations", [1, 3])
@pytest.mark.parametrize("change", ["forget", "expiry"])
def test_next_model_call_and_final_synthesis_refresh_memory(
    max_iterations: int, change: str
) -> None:
    now = [100.0]
    service = MemoryService(clock=lambda: now[0])
    item = service.remember(EDITOR, MemoryKind.PROFILE, SECRET, expires_at=150.0)
    registry = _revoking_registry(service, item, expires=now if change == "expiry" else None)
    h = build(
        [call_tools(tool_call("change_memory", {})), reply("finished")],
        memory=service,
        registry=registry,
        config=AgentConfig(max_iterations=max_iterations, model_retry_delay_s=0),
    )
    result = run(h.engine.run(RunRequest(EDITOR, "s1", "color")))

    assert result.status is RunStatus.COMPLETED
    assert SECRET in h.model.calls[0].messages[0].content
    assert len(h.model.calls) == 2
    assert all(SECRET not in message.content for message in h.model.calls[1].messages)


def test_model_retry_refreshes_memory_after_a_transient_failure() -> None:
    service = MemoryService()
    item = service.remember(EDITOR, MemoryKind.PROFILE, SECRET)

    def fail_and_forget(messages: list[Message], tools: Any) -> Any:
        assert SECRET in messages[0].content
        assert service.forget(EDITOR, item.id)
        raise TransientModelError("retry requested")

    h = build([fail_and_forget, reply("finished")], memory=service)
    result = run(h.engine.run(RunRequest(EDITOR, "s1", "color")))
    assert result.status is RunStatus.COMPLETED
    assert len(h.model.calls) == 2
    assert all(SECRET not in message.content for message in h.model.calls[1].messages)


def test_stream_started_observer_revocation_is_checked_before_provider_request() -> None:
    service = MemoryService()
    old = service.remember(EDITOR, MemoryKind.PROFILE, SECRET)

    class StreamingModel(ScriptedModel):
        async def complete_stream(
            self,
            messages: list[Message],
            tools: Any,
            *,
            on_delta: Any,
            max_tokens: int | None = None,
        ) -> Any:
            response = await self.complete(messages, tools, max_tokens=max_tokens)
            on_delta(response.content)
            return response

    class RevokeOnStreamStarted:
        revoked = False

        def emit(self, event: Any) -> None:
            if event.type is EventType.MODEL_RESPONSE_STARTED:
                assert service.forget(EDITOR, old.id)
                self.revoked = True

    model = StreamingModel([reply("finished")])
    observer = RevokeOnStreamStarted()
    engine = AgentEngine(
        model,
        FakeWorkspace.default().registry(),
        memory=service,
        events=observer,
        config=AgentConfig(stream_model=True),
    )
    result = run(engine.run(RunRequest(EDITOR, "s1", "color")))

    assert observer.revoked
    assert result.status is RunStatus.COMPLETED
    assert all(SECRET not in message.content for message in model.calls[0].messages)


def _search_history(content: str = SECRET) -> list[Message]:
    call = tool_call("search_memory", {"query": "cobalt"}, "old-search")
    return [
        Message("user", "Search my previous memories."),
        Message("assistant", tool_calls=(call,)),
        Message("tool", content, tool_call_id=call.id, name="search_memory"),
        Message("assistant", "Previous request completed."),
    ]


@pytest.mark.parametrize("change", ["forget", "supersede"])
@pytest.mark.parametrize("tool_name", ["search_memory", ""])
def test_cross_session_search_history_uses_current_authority(change: str, tool_name: str) -> None:
    service = MemoryService()
    old = service.remember(EDITOR, MemoryKind.FACT, SECRET, topic="color")
    if change == "forget":
        assert service.forget(EDITOR, old.id)
    else:
        service.remember(EDITOR, MemoryKind.FACT, "current cobalt memory", topic="color")
    h = build([reply("finished")], memory=service)
    history = _search_history()
    history[2] = replace(history[2], name=tool_name)
    run(h.engine.run(RunRequest(EDITOR, "new-session", "hello", history=history)))

    messages = h.model.calls[0].messages
    assert all(SECRET not in message.content for message in messages)
    tool = next(message for message in messages if message.tool_call_id == "old-search")
    assert tool.role == "tool"
    if change == "supersede":
        assert "current cobalt memory" in tool.content


def test_unpaired_search_output_cannot_reinject_unverified_body() -> None:
    service = MemoryService()
    h = build([reply("finished")], memory=service)
    history = [Message("tool", SECRET, tool_call_id="unbound", name="search_memory")]
    run(h.engine.run(RunRequest(EDITOR, "s1", "hello", history=history)))
    assert all(SECRET not in message.content for message in h.model.calls[0].messages)


def test_disabled_memory_does_not_reuse_search_plaintext_from_history() -> None:
    h = build([reply("finished")])
    run(h.engine.run(RunRequest(EDITOR, "s1", "hello", history=_search_history())))
    assert all(SECRET not in message.content for message in h.model.calls[0].messages)


def test_history_lookup_failure_does_not_fall_back_to_old_plaintext() -> None:
    class UnavailableAuthority(InMemoryMemoryStore):
        def get_items(self, scope: MemoryScope, item_ids: Any) -> list[MemoryItem]:
            raise ConnectionError("authoritative store unavailable")

    service = MemoryService(UnavailableAuthority())
    service.remember(EDITOR, MemoryKind.FACT, SECRET)
    h = build([reply("finished")], memory=service)
    run(h.engine.run(RunRequest(EDITOR, "s1", "hello", history=_search_history())))
    assert all(SECRET not in message.content for message in h.model.calls[0].messages)


def test_reused_call_id_in_later_round_does_not_hide_previous_memory_result() -> None:
    service = MemoryService()
    history = _search_history()
    history[2] = replace(history[2], name="")
    reused = tool_call("read_document", {"document_id": "d1"}, "old-search")
    history.extend(
        [
            Message("assistant", tool_calls=(reused,)),
            Message("tool", "document content", tool_call_id=reused.id, name="read_document"),
        ]
    )
    h = build([reply("finished")], memory=service)
    run(h.engine.run(RunRequest(EDITOR, "s1", "hello", history=history)))
    assert all(SECRET not in message.content for message in h.model.calls[0].messages)


def test_new_engine_approval_resume_refreshes_previous_search_result() -> None:
    service = MemoryService()
    old = service.remember(EDITOR, MemoryKind.FACT, SECRET, topic="color")
    h = build(
        [
            call_tools(tool_call("search_memory", {"query": "cobalt"}, "search")),
            call_tools(tool_call("update_document", {"document_id": "d1", "content": "v2"})),
        ],
        memory=service,
    )
    h.engine.registry.register(search_memory_tool(service))
    suspended = run(h.engine.run(RunRequest(EDITOR, "s1", "search cobalt and update d1")))
    assert suspended.prompt is not None
    assert SECRET in suspended.steps[0].tool_calls[0].result.output
    assert service.forget(EDITOR, old.id)
    h.broker.resolve(EDITOR, suspended.prompt.id, {"approved": True}, session_id="s1")
    fresh = build([reply("finished")], memory=service, broker=h.broker, workspace=h.workspace)
    fresh.engine.registry.register(search_memory_tool(service))

    result = run(fresh.engine.resume(EDITOR, suspended.prompt.id, session_id="s1"))
    assert result.status is RunStatus.COMPLETED
    assert h.workspace.writes == [("d1", "v2")]
    assert all(SECRET not in message.content for message in fresh.model.calls[0].messages)


def test_durable_recovery_refreshes_committed_search_result(tmp_path: Path) -> None:
    class ProcessCrash(BaseException):
        pass

    service = MemoryService(JsonlMemoryStore(tmp_path / "memory"))
    old = service.remember(EDITOR, MemoryKind.FACT, SECRET, topic="color")
    registry = FakeWorkspace.default().registry()
    registry.register(search_memory_tool(service))
    first_model = ScriptedModel(
        [call_tools(tool_call("search_memory", {"query": "cobalt"}, "search")), ProcessCrash()]
    )
    first = AgentEngine(
        first_model, registry, memory=service, run_store=JsonlRunStore(tmp_path / "runs")
    )
    request = RunRequest(EDITOR, "s1", "search cobalt", run_id="memory-recovery")
    with pytest.raises(ProcessCrash):
        run(first.run(request))
    assert service.forget(EDITOR, old.id)
    fresh_model = ScriptedModel([reply("finished")])
    fresh = AgentEngine(
        fresh_model, registry, memory=service, run_store=JsonlRunStore(tmp_path / "runs")
    )

    result = run(fresh.recover(EDITOR, request.run_id, session_id="s1"))
    assert result.status is RunStatus.COMPLETED
    assert all(SECRET not in message.content for message in fresh_model.calls[0].messages)


def test_context_summarizer_does_not_read_revoked_memory_tool_output() -> None:
    service = MemoryService()
    old = service.remember(EDITOR, MemoryKind.FACT, SECRET, topic="color")
    assert service.forget(EDITOR, old.id)
    history = [
        Message("user", "old request " + "ordinary text " * 900),
        *_search_history(),
        Message("user", "later discussion " + "ordinary text " * 900),
        Message("assistant", "Previous work completed."),
    ]
    summarizer = ScriptedModel([reply("The previous ordinary work is complete.")])
    h = build(
        [reply("finished")],
        memory=service,
        summarizer=summarizer,
        config=AgentConfig(
            context=ContextSettings(
                max_context_tokens=2_000, reserve_tokens=500, keep_recent_tokens=400
            ),
            model_retry_delay_s=0,
        ),
    )
    result = run(h.engine.run(RunRequest(EDITOR, "s1", "continue", history=history)))

    assert result.status is RunStatus.COMPLETED
    assert summarizer.calls
    assert all(
        SECRET not in message.content
        for call in [*summarizer.calls, *h.model.calls]
        for message in call.messages
    )


def test_revocation_while_summary_is_generated_is_checked_before_main_model() -> None:
    service = MemoryService()
    old = service.remember(EDITOR, MemoryKind.PROFILE, SECRET)

    def summarize_and_forget(messages: list[Message], tools: Any) -> Any:
        assert service.forget(EDITOR, old.id)
        return reply("Previous ordinary work completed.")

    summarizer = ScriptedModel([summarize_and_forget])
    h = build(
        [reply("finished")],
        memory=service,
        summarizer=summarizer,
        config=AgentConfig(
            context=ContextSettings(
                max_context_tokens=2_000, reserve_tokens=500, keep_recent_tokens=400
            )
        ),
    )
    history = [
        Message("user", "previous ordinary discussion " * 900),
        Message("assistant", "Previous ordinary work completed."),
    ]
    result = run(h.engine.run(RunRequest(EDITOR, "s1", "continue", history=history)))

    assert result.status is RunStatus.COMPLETED
    assert summarizer.calls
    assert all(SECRET not in message.content for message in h.model.calls[0].messages)


@pytest.mark.parametrize("tool_name", ["search_memory", ""])
def test_active_memory_tool_plaintext_is_not_turned_into_a_compaction_summary(
    tool_name: str,
) -> None:
    service = MemoryService()
    service.remember(EDITOR, MemoryKind.PROFILE, SECRET)
    search_history = _search_history()
    search_history[2] = replace(search_history[2], name=tool_name)
    history = [
        Message("user", "previous ordinary request " + "ordinary text " * 900),
        *search_history,
        Message("user", "later ordinary discussion " + "ordinary text " * 900),
        Message("assistant", "Previous ordinary work completed."),
    ]
    summarizer = ScriptedModel([reply("The previous ordinary work is complete.")])
    h = build(
        [reply("finished")],
        memory=service,
        summarizer=summarizer,
        config=AgentConfig(
            context=ContextSettings(
                max_context_tokens=2_000, reserve_tokens=500, keep_recent_tokens=400
            )
        ),
    )
    result = run(h.engine.run(RunRequest(EDITOR, "s1", "continue", history=history)))

    assert result.status is RunStatus.COMPLETED
    assert summarizer.calls
    summary_input = "\n".join(message.content for message in summarizer.calls[0].messages)
    transcript = json.loads(summarizer.calls[0].messages[-1].content)["conversation"]
    assert any(record.get("name") == "search_memory" for record in transcript)
    assert SECRET not in summary_input
    assert SECRET in h.model.calls[0].messages[0].content


def test_real_approval_snapshot_does_not_save_the_injected_memory_envelope() -> None:
    service = MemoryService()
    service.remember(EDITOR, MemoryKind.PROFILE, SECRET)
    h = build(
        [call_tools(tool_call("update_document", {"document_id": "d1", "content": "v2"}))],
        memory=service,
    )
    suspended = run(h.engine.run(RunRequest(EDITOR, "s1", "update d1")))

    assert suspended.status is RunStatus.SUSPENDED
    assert suspended.prompt is not None
    assert SECRET in h.model.calls[0].messages[0].content
    snapshot = RunSnapshot.decode(suspended.prompt.snapshot)
    assert snapshot.memory_prompt == ""
    assert SECRET not in suspended.prompt.snapshot
    assert "<memory>" not in suspended.prompt.snapshot
    stored = h.broker.store.get(suspended.prompt.id)
    assert stored is not None and stored.snapshot == suspended.prompt.snapshot
