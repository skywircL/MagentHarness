"""Offline restart demonstration; temporary state is removed when it finishes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from magent_harness import (
    AgentEngine,
    Caller,
    FunctionTool,
    JsonlMemoryStore,
    JsonlRunStore,
    MemoryKind,
    MemoryService,
    Role,
    RunRequest,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from magent_harness.testing import ScriptedModel, call_tools, reply, tool_call


class WorkerInterrupted(BaseException):
    """Leave the same durable boundary as a worker dying before its next model reply."""


async def demo(root: Path) -> None:
    caller = Caller("demo-workspace", "demo-user", Role.EDITOR)
    reads = 0

    async def read(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        nonlocal reads
        ctx.scope.require_document("demo-workspace", "roadmap")
        reads += 1
        return ToolResult.ok("Q4: ship the agent.")

    registry = ToolRegistry()
    registry.register(
        FunctionTool(ToolSpec("read_document", "Read the roadmap.", concurrency_safe=True), read)
    )
    memory = MemoryService(JsonlMemoryStore(root))
    memory.remember(caller, MemoryKind.PREFERENCE, "Answer briefly.")
    first = AgentEngine(
        ScriptedModel([call_tools(tool_call("read_document")), WorkerInterrupted()]),
        registry,
        run_store=JsonlRunStore(root),
        memory=memory,
    )
    try:
        await first.run(RunRequest(caller, "demo-session", "Read the roadmap", run_id="demo-run"))
    except WorkerInterrupted:
        pass

    restored_memory = MemoryService(JsonlMemoryStore(root))
    restored = AgentEngine(
        ScriptedModel([reply("Q4: ship the agent.")]),
        registry,
        run_store=JsonlRunStore(root),
        memory=restored_memory,
    )
    result = await restored.recover(caller, "demo-run", session_id="demo-session")
    assert result.final_answer == "Q4: ship the agent."
    assert reads == 1
    assert restored_memory.recall(caller, "roadmap").items
    print(f"Recovered: {result.final_answer}")
    print(f"Document reads: {reads}; memory survived a new store instance.")


if __name__ == "__main__":
    with TemporaryDirectory(prefix="magent-persistence-") as directory:
        asyncio.run(demo(Path(directory)))
