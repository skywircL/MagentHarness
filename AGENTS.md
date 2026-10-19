# 项目开发规则

## 项目目标
为个人文档 / 知识库智能工作台提供独立的 Magent Agent 运行时（Python harness）：在工具范围、人工审批与资源权限约束下检索、读取和更新 Workspace 内容。Workspace、文档不可变版本与 RAG 索引由后端负责，本仓库只通过工具接入它们。

## 开始之前
- 先读当前任务、相关目录规则和相邻实现；保留已有的用户改动。
- 开发或修复前先读 `.agents/skills/project-iteration/SKILL.md`，按其中的迭代流程执行。
- 架构与模块职责：`docs/architecture.md`；代码要求：`docs/code-standards.md`；验证命令：`docs/testing.md`。
- 持久的设计取舍见 `.agents/notes/`。

## 工作约定
- 每次迭代写出可观察的验收标准和非目标；独立改动分开交付，受影响的调用方一起更新。
- 优先使用已有模块与扩展点（`Tool`、`ChatModel`、`PromptStore`、`MemoryStore`、`EventSink`）；行为变化同步更新测试与当前文档。
- 权限、身份、HITL 状态机、快照格式属于高风险改动：需要失败路径测试，并在真实入口（`AgentEngine.run/resume`、`PromptBroker.resolve`）验证拒绝规则。
- 数据库等基础设施一律用 Docker 创建（docker compose），不在宿主机直接安装；连接信息走环境变量，不提交凭据。
- 按影响面选择验证；区分通过、失败、未运行、跳过。
- 未经授权不提交、不推送、不合并、不部署；不覆盖已有改动，不暴露凭据。

## 命令
真实命令、环境要求与验证范围见 `docs/testing.md`。
