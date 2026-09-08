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
- 通过 `upload_files` 传递文件，后端按 thread ID 隔离保存；
- 仅允许上传 `.txt`、`.md`、`.csv` 文本资料，总大小不超过 10 MB；
- 页面常驻医疗免责声明，后端仍会对每条回答执行独立合规审核。

可在 `src/config.ts` 中调整应用名称、默认后端地址和默认 Graph ID。

> 本 AI 仅提供健康科普参考，不构成医疗建议，身体不适请前往正规医疗机构就诊。
