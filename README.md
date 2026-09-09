# 医疗健康科普 Agent（med_agent）

基于 **LangChain + LangGraph** 的医疗健康科普智能体，知识库为**默沙东诊疗手册（大众版）**离线数据。

> ⚠️ **重要合规提示**：本 Agent 不能替代执业医师，不做疾病诊断、不开处方、不推荐个人用药，
> 仅提供健康科普、资料整理、就医指引、病历信息整理。
> 风险提示文案：**"本 AI 仅提供健康科普参考，不构成医疗建议，身体不适请前往正规医疗机构就诊。"**

## 环境

```bash
conda create -n med_agent python=3.11 -y
conda activate med_agent
pip install -r requirements.txt
```

## 快速开始

```bash
# 1. 构建知识库索引（全量约 2600 主题，首次构建较久；支持断点续建）
python main.py --build

# 2. 交互对话
python main.py --session demo1

# 3. 评估（高危拦截 / 症状抽取 P-R-F1 / 科室推荐）
python evaluate.py            # 快速评估
python evaluate.py --full     # 追加高危图拦截、知识覆盖/幻觉代理指标与 Agentic 轨迹评估
# 也可通过统一入口运行
python main.py --eval
python main.py --eval --full
```

### LangGraph Agent Server / Studio

开发环境首次安装 CLI：

```powershell
python -m pip install -r requirements-dev.txt
```

请在项目根目录（包含 `langgraph.json` 的目录）启动，而不是在
`sub_projects/agent-chat-ui` 前端目录启动：

```powershell
# 在克隆后的项目根目录执行
langgraph dev
```

默认 Agent Server 地址为 `http://127.0.0.1:2024`，统一 graph ID 为 `agent`。该命令启动后端 API/Studio；
Streamlit 页面仍使用 `streamlit run app.py` 单独启动。

Agent Chat UI 与通用智能体项目保持相同接入协议：

```powershell
cd sub_projects\agent-chat-ui
pnpm dev
```

前端默认连接 `http://localhost:2024` 的 `agent` 图，支持 LangGraph thread 历史、
流式状态和 `upload_files`。上传的 txt/md/csv 文件会按 thread ID 隔离保存到
`user_upload/<thread_id>/`，保存前会遮盖常见直接身份标识，再由受限文件工具读取。
脱敏只是防护层，上传前仍应人工删除无关个人信息。

## 项目结构

```
med_agent/
├── main.py                    # 入口：对话 / 构建索引 / 评估
├── pyproject.toml             # Agent Server 可编辑安装与包发现配置
├── langgraph.json             # 统一 graph ID、依赖与自定义 HTTP 应用
├── evaluate.py                # 评估脚本（Precision/Recall/F1、高危拦截率）
├── data/
│   ├── eval_dataset.json      # 测试数据集（普通/复杂/高危/特殊人群 case）
│   └── checkpoints.sqlite     # LangGraph 会话持久化（按 thread_id 隔离）
├── docs/
│   ├── architecture/          # 双模式流程与 Agentic 工具循环图
│   └── evaluation-baseline.md # 可公开复现的合成开发集评估基线
├── base/config.py             # 配置（模型、路径、RAG 参数）
├── conn/llm.py                # LLM / BGE-M3 嵌入（SiliconFlow OpenAI 兼容接口）
├── medical/
│   ├── compliance.py          # 合规层：高危拦截 / 危急信号 / 输出违禁词审核 / 免责声明
│   ├── prompts.py             # 强约束 System Prompt（能力边界写死）
│   ├── department.py          # 症状→科室确定性规则映射
│   ├── state.py               # MedicalAgentState 自定义状态
│   ├── graph.py               # LangGraph 主流程（拦截→抽取→Agent→输出审核）
│   ├── api.py                 # Agent Server 自定义健康检查/能力说明接口
│   ├── uploads.py             # Agent Chat UI 上传协议与 thread 隔离落盘
│   ├── tools.py               # @tool 工具集
│   ├── secure_store.py        # 会话健康数据 Fernet 加密存储
│   └── rag/
│       ├── html_parser.py     # 默沙东 HTML 解析 + 按章节切片(512-1024)
│       ├── build_index.py     # Chroma 向量库 + BM25 语料构建
│       └── retriever.py       # 混合检索（向量+BM25+RRF+Rerank 重排）
└── user_upload/               # 用户上传检验报告文本目录（read_medical_doc 仅允许读这里）
```

## Agent 流程（LangGraph）

```
START → ingest_uploads（解析 UI 的 upload_files，按 thread 隔离）
          → input_guard（兼容纯文本与 LangGraph content blocks）
          ├─ 危急信号 → emergency（零模型 120/急诊提示+免责声明）→ END
          ├─ 高危(求诊断/求开药) → refuse（风险提示+替代帮助+免责声明）→ END
          └─ 其他请求 → extract（症状实体抽取→State）
                    → classify_task
                      ├─ fast_rag：普通科普首步固定检索，仅开放检索/科室 2 个工具
                      └─ agentic：复杂任务首步评估信息缺口，在 8 个白名单工具内编排
                    → agent ↔ tools → record_tool_round（轨迹记录/循环上限）
                    → output_check（违禁词审核 + 强制免责声明）→ END
```

- **自定义 State**：除用户画像外，另保存 `current_turn_symptoms / task_mode / task_goal /
  agent_tool_trace / attempted_queries / agent_plan / asked_questions`；模式路由使用本轮症状，避免历史状态污染；
- **会话隔离**：CLI/Streamlit 本地运行时使用 SQLite checkpoint 按 `thread_id` 持久化；
  `langgraph dev` 下的 Agent Server/Studio 使用服务端运行时管理的 thread/checkpoint。

架构图：

- [双模式问答流程](docs/architecture/med_agent_双模式问答流程.svg)
- [Agentic 工具循环与能力边界](docs/architecture/med_agent_agentic_loop_真Agent能力.svg)

### 为什么现在需要 Agent

普通科普仍走 `fast_rag`：首步强制检索，后续只允许补检索、调用科室工具或回答。以下开放式任务进入
`agentic` 模式，工具顺序和调用次数由执行结果动态决定：

- 结合上传报告，先整理报告，再分别检索其中的医学概念；
- 多症状、多子问题的综合整理，信息不足时逐项补问；
- 对比多个检查或医学概念，按子问题改写并重复检索；
- 收集信息后生成非诊断性的就医沟通摘要和准备清单；
- 检索证据不足时换查询重试，并在达到工具轮数上限后安全停止。

可用于验证复杂路径的提问示例：

```text
结合我上传的血常规报告，整理里面需要了解的指标，再生成就医时要问的问题清单。
我头痛、呕吐、视物模糊三天了，请先检查还缺什么信息，再帮我整理就医准备。
对比胃镜和呼气试验分别用于了解什么问题，请分开检索并给出来源。
```

## 工具集

| 工具 | 说明 |
|---|---|
| `medical_rag_search` | 检索默沙东权威知识库，**医学事实性回答必须优先调用**，无资料时如实告知，禁止编造 |
| `get_department_recommend` | 症状→科室确定性规则映射（含危急信号转急诊、儿童/老年通道） |
| `symptom_extract` | 仅供离线评测复用；在线抽取由图中的 `extract` 节点完成，不绑定主 Agent |
| `read_medical_doc` | 读取 `user_upload/` 内的检验报告/病历文本（限白名单后缀，防路径穿越） |
| `assess_information_gaps` | 复杂任务第一步评估信息缺口，由 Agent 决定补问或继续执行 |
| `build_visit_preparation` | 汇总多轮信息为非诊断性的就医沟通摘要和准备清单 |
| `create_task_plan` | 仅为至少 3 个独立步骤的复杂任务建立计划 |
| `update_task_progress` | 根据工具成功结果更新计划状态，拒绝结果不会写入 State |
| `check_evidence_sufficiency` | 从本轮真实 RAG 工具消息注入证据并检查子问题覆盖，不接受模型自填证据摘要 |

## 检索方案（RAG）

1. **切片**：按主题 h1/h2/h3 章节层级切分，目标块大小 400-1024 字符；无法继续合并的短小节可低于 400 字符，长段落保留约 80 字符重叠；
2. **嵌入**：使用 BGE-M3（`BAAI/bge-m3`，SiliconFlow API）生成稠密向量；当前项目尚未完成与通用 embedding 的同集对比实验；
3. **向量库**：Chroma（原型够用；生产可平移 Milvus，检索层接口不变）；
4. **混合检索**：向量召回 + jieba 分词 BM25 关键词召回 → RRF 融合；
5. **重排**：BGE-reranker-v2-m3（`.env` 中 `ENABLE_RERANK=true` 开启，失败自动降级）。

## 合规与安全设计

| 层 | 措施 |
|---|---|
| 输入过滤 | 规则正则前置拦截求诊断/求开药请求；危急信号强制转急诊指引 |
| 提示约束 | System Prompt 写死身份边界 + 检索优先 + 无资料如实告知 |
| 工具约束 | 无任何诊断/开药工具；科室推荐为确定性规则 |
| 输出审核 | 后置违禁词检测（"确诊为你得了…/建议服用…mg"），命中整句剔除并补风险提示 |
| 免责声明 | 回答末尾强制附带 |
| 隐私 | 普通运行日志脱敏并轮转；会话审计日志默认不落盘，显式开启后仅保存最小化、脱敏且 Fernet 加密的记录；禁止用于训练 |

## 配置（.env）

复制 `.env.example` 为 `.env` 后填写：`MODEL_API_BASE_URL / OPENAI_API_KEY / BASE_LLM /
EMBEDDING_MODEL / MSD_DATA_PATH / ENABLE_RERANK / MAX_AGENT_TOOL_ROUNDS` 等。`.env` 含密钥，不要提交到版本库。

普通运行日志默认写入项目根目录下的 `logs/med_agent.log`，单文件默认 5 MB、保留 5 个备份，
并在写入前遮盖常见直接标识符。可通过 `ENABLE_FILE_LOG / LOG_PATH / LOG_LEVEL /
LOG_MAX_BYTES / LOG_BACKUP_COUNT` 调整。它不会记录原始用户输入；需要会话审计时，另行显式开启
`ENABLE_SECURE_SESSION_LOG=true`，审计记录会加密写入 `data/sessions.enc`。

## 公开发布前检查

`data/` 中的运行数据、`user_upload/`、`vectorstore/`、`output/`、`logs/` 和本地 `.env`
已默认排除在 Git 之外。每次提交前运行：

```powershell
python scripts/check_public_ready.py
ruff format --check .
ruff check .
python -m unittest discover -s tests -v
```

需调用真实模型和本地索引时，可另行执行 `python scripts/smoke_e2e.py`。

## 评估口径

- `python -m unittest discover -s tests -v` 当前执行 75 项不依赖真实外部模型的自动化测试；
- `python evaluate.py` 使用合成开发集执行快速回归，当前覆盖 13 条高危规则、7 条症状抽取和 14 条科室路由；
- `python evaluate.py --full` 会调用真实模型和本地知识库，追加知识覆盖与 9 条 Agentic 场景轨迹评估；
- 小样本开发集结果仅用于回归，不能解释为临床准确率或线上生产效果。公开基线见 [评估基线](docs/evaluation-baseline.md)。

## 开源协议

项目代码以 [MIT License](LICENSE) 开源。医疗知识库原始数据、模型权重和用户数据
不因本代码协议自动获得再分发权，使用者需自行遵守各自的许可条款与隐私规则。
