# Magent Harness

面向个人文档与知识库的独立 Magent Agent 运行时（agent harness）。它负责 ReAct 循环、上下文窗口管理、长期记忆、人工审批 / 提问的挂起与恢复，以及基于身份的工具权限。后端（Workspace、文档版本、RAG 索引）通过注册工具接入，不在本仓库内。

模块职责、依赖方向与设计取舍见 [docs/architecture.md](docs/architecture.md)。

## 快速开始

```bash
uv sync
uv run pytest
```

```python
import asyncio
from magent_harness import (
    AgentEngine, Caller, FunctionTool, Role, RunRequest, ToolRegistry, ToolResult, ToolSpec,
)
from magent_harness.testing import ScriptedModel, call_tools, reply, tool_call

async def search(ctx, args):  # 身份来自 ctx，而不是模型参数
    return ToolResult.ok(f"{ctx.caller.workspace_id}: 3 hits for {args['query']}")

registry = ToolRegistry()
registry.register(FunctionTool(
    ToolSpec("search_documents", "Search the workspace.",
             {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
             concurrency_safe=True),
    search,
))
model = ScriptedModel([call_tools(tool_call("search_documents", {"query": "roadmap"})),
                       reply("Found it.")])
engine = AgentEngine(model, registry)
result = asyncio.run(engine.run(RunRequest(Caller("ws1", "u1", Role.EDITOR), "s1", "Where is the roadmap?")))
print(result.status, result.final_answer)
```

真实模型通过 `ChatModel` 协议接入。已提供可选的 `DeepSeekChatModel`：

```bash
uv sync --extra deepseek
uv run python examples/deepseek_demo.py
uv run python examples/deepseek_demo.py --thinking
uv run python examples/deepseek_demo.py --stream --thinking
```

以上示例默认离线，在 HTTP 边界模拟响应，真实执行“请求 → 读取工具 → 回答”；`--stream` 输出含正文增量的 SSE。服务端配置 `DEEPSEEK_API_KEY` 后，可显式加 `--live` 接入真实模型；默认非流式、关闭思考，`thinking=True` 开启思考并支持审批挂起与恢复。配置、快照兼容和实测说明见 [DeepSeek 接入](docs/deepseek.md)。

## SSE 运行进度

`stream_run` 和 `stream_resume` 将运行事件转换为 SSE 字节流，供后端 HTTP 响应发送：

```python
from magent_harness import AgentConfig, AgentEngine, DeepSeekChatModel, stream_run

# 服务生命周期内复用 model，退出时 await model.aclose()。
model = DeepSeekChatModel()
engine = AgentEngine(model, registry, config=AgentConfig(stream_model=True))

async def response_body(engine, request):
    async with stream_run(engine, request) as stream:
        async for frame in stream:
            yield frame
        result = await stream.result()  # 服务端保存 result.messages / usage
```

HTTP 层设置 `Content-Type: text/event-stream; charset=utf-8` 和 `Cache-Control: no-cache`，并保证断开连接时取消响应任务、关闭 `response_body` 生成器。流上下文会取消并等待运行任务。审批挂起也会结束当前流；回答通过 `PromptBroker.resolve` 记录后，用 `stream_resume` 建立新的流。

运行 `uv run python examples/sse_demo.py` 可查看运行事件；`examples/deepseek_demo.py --stream` 还会推送模型正文增量。文本按 provider 返回的片段发送，不保证每片段只有一个字。完整事件格式、重试时丢弃临时文本、心跳及生命周期约定见 [SSE 接入](docs/sse.md)。

## 持久化与恢复

`AgentEngine(..., run_store=JsonlRunStore(state_dir))` 可开启本地 POSIX JSONL 运行日志和持久审批；`MemoryService(JsonlMemoryStore(state_dir))` 保存跨进程记忆。新进程通过 `engine.recover(caller, run_id, session_id=...)` 从事实恢复，通过原 `resume` 继续审批。已落盘工具结果不重跑，结果未知的变更抛 `RecoveryUnsafe` 并停止自动续跑。默认内存用法保持兼容。

`uv run python examples/persistence_demo.py` 提供离线演示。四类崩溃窗口、Checkpoint 校验、授权与接入边界见 [持久化与恢复](docs/persistence.md)。

MagWorkspace 已通过 MySQL 持久化跨 Session 的个人记忆，由 Go 授权、Python 私有回调读取。遗忘保留内容指纹墓碑，防止同内容换 ID 复活；每次模型调用前复核当前正文、状态和有效期。接入、保证范围及验证见 [长期记忆](docs/memory.md)。

长上下文管理在实际发送前计入 system、工具 schema 与回答预留；摘要有独立预算、超时和完整性检查，失败时可明确省略旧历史，不可缩减的请求直接失败。原始请求、完整工具组、当前 reasoning 和审批边界保留，模型可见历史改写先持久化。配置及 MagWorkspace 历史窗口见[长上下文与安全压缩](docs/context.md)。

工具按当前后端能力动态提供，`AgentConfig.enabled_tools` 同时约束模型可见列表和执行，恢复时重新检查。MagWorkspace 接入知识库发现、多库检索/文档筛选、文档阅读、历史会话、提问、个人记忆和审批写入；知识库筛选始终收窄原有文档权限。接口与边界见[工具能力与结果治理](docs/tools.md)。

## 文档

- [AGENTS.md](AGENTS.md)：贡献规则与入口
- [docs/architecture.md](docs/architecture.md)：模块职责、依赖方向、状态归属
- [docs/code-standards.md](docs/code-standards.md)：代码规范
- [docs/testing.md](docs/testing.md)：验证命令与证据要求
- [docs/sse.md](docs/sse.md)：SSE 响应接入与运行生命周期
- [docs/deepseek.md](docs/deepseek.md)：DeepSeek 适配、思考模式与审批恢复
- [docs/persistence.md](docs/persistence.md)：JSONL、持久审批 / 记忆和安全恢复
- [docs/memory.md](docs/memory.md)：跨 Session 数据库记忆、内容墓碑和注入前权威复核
- [docs/context.md](docs/context.md)：完整请求预算、安全摘要、降级与历史窗口
