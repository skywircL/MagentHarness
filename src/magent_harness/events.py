"""Run events for streaming progress to the backend / UI."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    RUN_RESUMED = "run_resumed"
    ROUND_STARTED = "round_started"
    MODEL_RESPONSE_STARTED = "model_response_started"
    TEXT_DELTA = "text_delta"
    MODEL_RESPONSE_COMPLETED = "model_response_completed"
    MODEL_RESPONSE_DISCARDED = "model_response_discarded"
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CONTEXT_COMPACTED = "context_compacted"
    PROMPT_REQUIRED = "prompt_required"  # approval or question; the run suspends
    RUN_SUSPENDED = "run_suspended"
    FINAL_ANSWER = "final_answer"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELED = "run_canceled"


@dataclass(frozen=True)
class Event:
    type: EventType
    run_id: str
    session_id: str
    data: dict[str, Any] = field(default_factory=dict)


class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...


class NullSink:
    def emit(self, event: Event) -> None:
        return None


class CollectingSink:
    """Keeps every event in memory; for tests and simple callers."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)

    def of_type(self, event_type: EventType) -> list[Event]:
        return [e for e in self.events if e.type is event_type]


def safe_emit(sink: EventSink, event: Event) -> None:
    """A broken sink must not break the run it observes."""
    try:
        sink.emit(event)
    except Exception:
        logger.exception("event sink failed on %s", event.type)
