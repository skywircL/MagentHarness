"""Provider-independent model streaming selection, privacy and run isolation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from support import EDITOR, run

from magent_harness import (
    AgentConfig,
    AgentEngine,
    ChatResponse,
    CollectingSink,
    ContextSettings,
    EventType,
    Message,
    RunRequest,
    RunStatus,
    ToolSpec,
    stream_run,
)
from magent_harness.testing import ModelCall, ScriptedModel, reply


class DualModel:
    def __init__(self, *streamed_answers: str, complete_answer: str = "nonstream answer") -> None:
        self.streamed_answers = list(streamed_answers)
        self.complete_answer = complete_answer
        self.complete_calls: list[ModelCall] = []
        self.stream_calls: list[ModelCall] = []

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec], *, max_tokens: int | None = None
    ) -> ChatResponse:
        self.complete_calls.append(ModelCall(list(messages), list(tools), max_tokens))
        return reply(self.complete_answer)

    async def complete_stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        on_delta: Callable[[str], None],
        max_tokens: int | None = None,
    ) -> ChatResponse:
        self.stream_calls.append(ModelCall(list(messages), list(tools), max_tokens))
        answer = self.streamed_answers.pop(0)
        for content in answer:
            on_delta(content)
            await asyncio.sleep(0)
        return reply(answer)


def request(query: str = "hello", **kwargs: Any) -> RunRequest:
    return RunRequest(EDITOR, "model-session", query, **kwargs)


def test_default_config_preserves_nonstreaming_model_behavior() -> None:
    model = DualModel("must not stream")
    sink = CollectingSink()
    result = run(AgentEngine(model, events=sink).run(request()))
    assert result.final_answer == "nonstream answer"
    assert len(model.complete_calls) == 1
    assert model.stream_calls == []
    assert not sink.of_type(EventType.TEXT_DELTA)
    assert not sink.of_type(EventType.MODEL_RESPONSE_STARTED)


def test_streaming_config_falls_back_for_legacy_chat_models() -> None:
    model = ScriptedModel([reply("legacy answer")])
    sink = CollectingSink()
    engine = AgentEngine(model, config=AgentConfig(stream_model=True), events=sink)
    result = run(engine.run(request()))
    assert result.status is RunStatus.COMPLETED
    assert result.final_answer == "legacy answer"
    assert len(model.calls) == 1
    assert not sink.of_type(EventType.TEXT_DELTA)
    assert not sink.of_type(EventType.MODEL_RESPONSE_STARTED)


def test_empty_answer_discards_draft_and_assigns_new_response_id_to_retry() -> None:
    model = DualModel(" \n", "answer")
    sink = CollectingSink()
    result = run(
        AgentEngine(model, config=AgentConfig(stream_model=True), events=sink).run(request())
    )
    assert result.status is RunStatus.COMPLETED
    assert result.final_answer == "answer"
    started = sink.of_type(EventType.MODEL_RESPONSE_STARTED)
    discarded = sink.of_type(EventType.MODEL_RESPONSE_DISCARDED)
    completed = sink.of_type(EventType.MODEL_RESPONSE_COMPLETED)
    assert len(started) == 2 and len(discarded) == len(completed) == 1
    first_id, second_id = [event.data["response_id"] for event in started]
    assert first_id != second_id
    assert discarded[0].data == {"response_id": first_id, "round": 0, "reason": "empty"}
    assert completed[0].data["response_id"] == second_id
    assert model.complete_calls == []
    assert len(model.stream_calls) == 2
    assert [message.content for message in result.messages if message.role == "assistant"] == [
        "answer"
    ]


def test_final_synthesis_uses_streaming_without_offering_tools() -> None:
    model = DualModel("final synthesis")
    sink = CollectingSink()
    engine = AgentEngine(
        model,
        config=AgentConfig(stream_model=True, max_iterations=0, max_completion_tokens=42),
        events=sink,
    )
    result = run(engine.run(request()))
    assert result.status is RunStatus.COMPLETED
    assert result.stop_reason == "max_iterations"
    assert result.final_answer == "final synthesis"
    assert model.complete_calls == []
    assert len(model.stream_calls) == 1
    assert model.stream_calls[0].tools == []
    assert model.stream_calls[0].max_tokens == 42
    assert "".join(event.data["content"] for event in sink.of_type(EventType.TEXT_DELTA)) == (
        "final synthesis"
    )
    assert sink.of_type(EventType.MODEL_RESPONSE_COMPLETED)[0].data["kind"] == "answer"


def test_context_summary_stays_nonstreaming_and_never_appears_as_public_delta() -> None:
    model = DualModel("public answer", complete_answer="private compaction summary")
    sink = CollectingSink()
    settings = ContextSettings(max_context_tokens=1000, reserve_tokens=250, keep_recent_tokens=200)
    engine = AgentEngine(
        model, config=AgentConfig(stream_model=True, context=settings), events=sink
    )
    history = [
        Message("user" if index % 2 == 0 else "assistant", "old conversation " + "x" * 1000)
        for index in range(8)
    ]
    result = run(engine.run(request(history=history)))
    assert result.status is RunStatus.COMPLETED
    assert len(model.complete_calls) == len(model.stream_calls) == 1
    assert model.complete_calls[0].tools == []
    assert json.loads(model.complete_calls[0].messages[-1].content)["conversation"]
    assert "private compaction summary" in model.stream_calls[0].messages[1].content
    assert sink.of_type(EventType.CONTEXT_COMPACTED)[0].data["compacted"] is True
    assert len(sink.of_type(EventType.MODEL_RESPONSE_STARTED)) == 1
    assert "".join(event.data["content"] for event in sink.of_type(EventType.TEXT_DELTA)) == (
        "public answer"
    )
    assert "private compaction summary" not in json.dumps([event.data for event in sink.events])


def test_shared_engine_concurrent_streams_keep_deltas_and_response_ids_isolated() -> None:
    async def scenario() -> None:
        both_started = asyncio.Event()
        release = asyncio.Event()
        entered: list[str] = []

        class ConcurrentModel(DualModel):
            async def complete_stream(
                self,
                messages: list[Message],
                tools: list[ToolSpec],
                *,
                on_delta: Callable[[str], None],
                max_tokens: int | None = None,
            ) -> ChatResponse:
                query = messages[-1].content
                on_delta(query + "-")
                entered.append(query)
                if len(entered) == 2:
                    both_started.set()
                await release.wait()
                on_delta("answer")
                return reply(query + "-answer")

        model = ConcurrentModel()
        engine = AgentEngine(model, config=AgentConfig(stream_model=True))

        async def consume(query: str) -> list[dict[str, Any]]:
            async with stream_run(
                engine, request(query, run_id="shared-id"), heartbeat_interval=None
            ) as stream:
                payloads = [
                    json.loads(value.decode().splitlines()[1][6:]) async for value in stream
                ]
                assert (await stream.result()).final_answer == query + "-answer"
                return payloads

        tasks = [asyncio.create_task(consume(query)) for query in ("alpha", "beta")]
        try:
            await asyncio.wait_for(both_started.wait(), 1)
            release.set()
            results = await asyncio.gather(*tasks)
        finally:
            release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        response_ids = []
        for query, payloads in zip(("alpha", "beta"), results, strict=True):
            deltas = [event["data"] for event in payloads if event["type"] == "text_delta"]
            assert "".join(event["content"] for event in deltas) == query + "-answer"
            assert len({event["response_id"] for event in deltas}) == 1
            response_ids.append(deltas[0]["response_id"])
            assert all(event["run_id"] == "shared-id" for event in payloads)
        assert response_ids[0] != response_ids[1]
        assert model.complete_calls == []

    run(scenario())
