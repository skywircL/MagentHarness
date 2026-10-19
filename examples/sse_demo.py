"""Print an SSE response body without a server or external model credentials.

Run from the repository root: ``uv run python examples/sse_demo.py``.
"""

from __future__ import annotations

import asyncio
import sys

from magent_harness import AgentEngine, Caller, Role, RunRequest, RunStatus, stream_run
from magent_harness.testing import ScriptedModel, reply


async def main() -> None:
    engine = AgentEngine(ScriptedModel([reply("SSE 已接通。")]))
    request = RunRequest(Caller("demo-workspace", "demo-user", Role.VIEWER), "demo", "你好")
    async with stream_run(engine, request) as stream:
        async for frame in stream:
            sys.stdout.buffer.write(frame)
            sys.stdout.buffer.flush()
        result = await stream.result()
        assert result.status is RunStatus.COMPLETED


if __name__ == "__main__":
    asyncio.run(main())
