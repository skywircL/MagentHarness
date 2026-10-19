"""Framework-independent SSE bodies for one engine invocation.

Use ``stream_run`` / ``stream_resume`` as async context managers inside the
HTTP response body's lifetime. Exiting the context cancels and joins the
producer; the HTTP layer must close that context on disconnect. All operations
run on the same asyncio event loop. There is no event persistence or replay.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from .engine import AgentEngine, RunRequest, RunResult
from .events import Event, EventSink
from .identity import Caller

_HEARTBEAT = b": keep-alive\n\n"


class SSEStreamError(RuntimeError):
    """The stream could not deliver its events without loss or invalid encoding."""


def encode_sse_event(event: Event) -> bytes:
    """Encode a public event as one UTF-8 SSE frame with a JSON envelope.

    Only the enum's event name enters an SSE field. All other values live in
    JSON, where CR/LF are escaped, so user content cannot inject another frame.
    No ``id`` is emitted: this adapter does not implement resumable delivery.
    """
    payload = json.dumps(
        {
            "type": event.type.value,
            "run_id": event.run_id,
            "session_id": event.session_id,
            "data": event.data,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return f"event: {event.type.value}\ndata: {payload}\n\n".encode()


class SSEStream(AsyncIterator[bytes]):
    """Single-consumer stream; acquire through ``stream_run`` or ``stream_resume``.

    The buffer is bounded by both event count and encoded event size. An
    unrepresentable event or slow consumer fails the stream and requests
    cancellation of the run instead of silently dropping events. Cancellation
    is cooperative and cannot undo tool side effects that already happened.
    """

    def __init__(
        self,
        operation: Callable[[EventSink], Awaitable[RunResult]],
        *,
        heartbeat_interval: float | None = 15.0,
        max_buffered_events: int = 256,
        max_event_bytes: int = 1_048_576,
    ) -> None:
        if heartbeat_interval is not None and (
            isinstance(heartbeat_interval, bool)
            or not isinstance(heartbeat_interval, (float, int))
            or not math.isfinite(heartbeat_interval)
            or heartbeat_interval <= 0
        ):
            raise ValueError("heartbeat_interval must be positive and finite, or None")
        if type(max_buffered_events) is not int or max_buffered_events <= 0:
            raise ValueError("max_buffered_events must be a positive integer")
        if type(max_event_bytes) is not int or max_event_bytes <= 0:
            raise ValueError("max_event_bytes must be a positive integer")
        self._heartbeat_interval = heartbeat_interval
        self._max_buffered_events = max_buffered_events
        self._max_event_bytes = max_event_bytes
        self._frames: deque[bytes] = deque()
        self._ready = asyncio.Event()
        self._error: SSEStreamError | None = None
        self._closed = False
        self._reading = False
        self._cancel_requested = False
        self._task = asyncio.create_task(self._produce(operation))

    async def _produce(self, operation: Callable[[EventSink], Awaitable[RunResult]]) -> RunResult:
        try:
            # Even with an eager task factory, assign self._task before emit
            # can request cancellation on encoding failure or buffer overflow.
            await asyncio.sleep(0)
            return await operation(self)
        finally:
            # Completion must wake the reader even when no event was emitted
            # (for example, a rejected resume or invalid request).
            self._ready.set()

    def emit(self, event: Event) -> None:
        """Capture an event immediately, before mutable payloads can change."""
        if self._closed or self._error is not None:
            return
        if len(self._frames) >= self._max_buffered_events:
            self._fail("SSE event buffer is full; consumer is too slow")
            return
        try:
            frame = encode_sse_event(event)
        except (TypeError, ValueError, UnicodeError, RecursionError):
            self._fail("SSE event could not be encoded as UTF-8 JSON")
            return
        if len(frame) > self._max_event_bytes:
            self._fail("SSE event exceeds max_event_bytes")
            return
        self._frames.append(frame)
        self._ready.set()

    def _fail(self, message: str) -> None:
        # safe_emit deliberately contains observer exceptions. Keep failure
        # out of that channel so the HTTP consumer still sees the data loss.
        self._error = SSEStreamError(message)
        self._frames.clear()
        self._ready.set()
        self._cancel_producer()

    def _cancel_producer(self) -> None:
        if not self._cancel_requested and not self._task.done():
            self._cancel_requested = True
            # Do not interrupt cleanup already entered after cancellation.
            if not self._task.cancelling():
                self._task.cancel()

    async def __anext__(self) -> bytes:
        if self._reading:
            raise RuntimeError("SSEStream supports only one consumer")
        self._reading = True
        try:
            while True:
                if self._error is not None:
                    raise self._error
                if self._closed:
                    raise StopAsyncIteration
                if self._frames:
                    return self._frames.popleft()
                if self._task.done():
                    await self.result()
                    raise StopAsyncIteration
                self._ready.clear()
                try:
                    async with asyncio.timeout(self._heartbeat_interval):
                        await self._ready.wait()
                except TimeoutError:
                    return _HEARTBEAT
        finally:
            self._reading = False

    async def result(self) -> RunResult:
        """Return the engine result (including history), or propagate failure.

        Normally called after draining the stream; waiting here before reading
        may fill the buffer. Never serialize the whole result to the SSE client:
        a suspended result's prompt contains a private server-side snapshot.
        """
        if self._error is not None:
            raise self._error
        try:
            result = await asyncio.shield(self._task)
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            if caller is not None and caller.cancelling():
                raise
            if self._error is not None:
                raise self._error from None
            raise
        except Exception:
            if self._error is not None:
                raise self._error from None
            raise
        if self._error is not None:
            raise self._error
        return result

    async def aclose(self) -> None:
        """Stop delivery and join the producer, including cancellation cleanup."""
        self._closed = True
        self._frames.clear()
        self._ready.set()
        self._cancel_producer()
        # A disconnect may cancel the response task while the producer is
        # already cleaning up. Finish that cleanup before propagating caller
        # cancellation, without sending a second cancellation to the producer.
        joined = asyncio.gather(self._task, return_exceptions=True)
        canceled: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(joined)
                break
            except asyncio.CancelledError as exc:
                canceled = exc
        if canceled is not None:
            raise canceled


@asynccontextmanager
async def _stream(
    operation: Callable[[EventSink], Awaitable[RunResult]],
    *,
    heartbeat_interval: float | None,
    max_buffered_events: int,
    max_event_bytes: int,
) -> AsyncIterator[SSEStream]:
    stream = SSEStream(
        operation,
        heartbeat_interval=heartbeat_interval,
        max_buffered_events=max_buffered_events,
        max_event_bytes=max_event_bytes,
    )
    try:
        yield stream
    except BaseException:
        await stream.aclose()
        raise
    else:
        await stream.aclose()
        if stream._error is not None:
            raise stream._error
        if not stream._task.cancelled():
            stream._task.result()


@asynccontextmanager
async def stream_run(
    engine: AgentEngine,
    request: RunRequest,
    *,
    heartbeat_interval: float | None = 15.0,
    max_buffered_events: int = 256,
    max_event_bytes: int = 1_048_576,
) -> AsyncIterator[SSEStream]:
    """Stream one run; the backend supplies an authenticated ``RunRequest``."""
    async with _stream(
        lambda sink: engine.run(request, events=sink),
        heartbeat_interval=heartbeat_interval,
        max_buffered_events=max_buffered_events,
        max_event_bytes=max_event_bytes,
    ) as stream:
        yield stream


@asynccontextmanager
async def stream_resume(
    engine: AgentEngine,
    caller: Caller,
    prompt_id: str,
    *,
    session_id: str = "",
    heartbeat_interval: float | None = 15.0,
    max_buffered_events: int = 256,
    max_event_bytes: int = 1_048_576,
) -> AsyncIterator[SSEStream]:
    """Stream an answered prompt via the engine's ownership and claim checks."""
    async with _stream(
        lambda sink: engine.resume(caller, prompt_id, session_id=session_id, events=sink),
        heartbeat_interval=heartbeat_interval,
        max_buffered_events=max_buffered_events,
        max_event_bytes=max_event_bytes,
    ) as stream:
        yield stream


@asynccontextmanager
async def stream_recover(
    engine: AgentEngine,
    caller: Caller,
    run_id: str,
    *,
    session_id: str = "",
    heartbeat_interval: float | None = 15.0,
    max_buffered_events: int = 256,
    max_event_bytes: int = 1_048_576,
) -> AsyncIterator[SSEStream]:
    """Stream recovery after the backend has authenticated and claimed the run.

    A persisted terminal result may return without emitting new model events.
    This starts a new invocation; it does not replay old SSE frames.
    """
    async with _stream(
        lambda sink: engine.recover(caller, run_id, session_id=session_id, events=sink),
        heartbeat_interval=heartbeat_interval,
        max_buffered_events=max_buffered_events,
        max_event_bytes=max_event_bytes,
    ) as stream:
        yield stream
