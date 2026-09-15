# 医疗健康科普助手 UI

该前端沿用通用智能体项目的 Agent Chat UI 交互协议，默认连接本项目的
LangGraph Agent Server，并保留医疗免责声明和受限文件上传能力。

## 启动

先在项目根目录启动后端：

```powershell
# 在克隆后的项目根目录执行
conda activate med_agent
langgraph dev
```

再启动前端：

```powershell
cd sub_projects\agent-chat-ui
pnpm install
pnpm dev
```

浏览器打开 `http://localhost:3000`。前端默认连接：

- Deployment URL：`http://127.0.0.1:2024`
- Graph ID：`agent`

## 与后端对齐的能力

- 使用 LangGraph thread 创建、恢复历史并流式展示状态；
- 通过 `context` 传递运行时上下文；
- 通过 `upload_files` 传递 Base64 文件块，后端按 thread ID 分目录保存；该目录边界不是用户身份认证或租户 ACL；
- 前端按 MIME 或扩展名筛选 `.txt`、`.md`、`.csv`；后端再次校验后缀、Base64、UTF-8，并对单次提交执行单文件不超过 10 MB、累计不超过 10 MB 的限制；
- 页面常驻医疗免责声明，后端仍会对每条回答执行独立合规审核。

应用名称和免责声明在 `src/config.ts` 中调整。后端地址与 Graph ID 优先通过
`.env` 中的 `NEXT_PUBLIC_API_URL` 和 `NEXT_PUBLIC_ASSISTANT_ID` 配置；
代码中的本地默认值位于 `src/providers/Thread.tsx` 与 `src/providers/Stream.tsx`。

> 本 AI 仅提供健康科普参考，不构成医疗建议，身体不适请前往正规医疗机构就诊。
