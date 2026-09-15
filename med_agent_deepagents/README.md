# med_agent：Deep Agents 双模式医疗科普 Agent

这是一个用于学习、面试展示和本地验证的医疗健康科普 Agent。项目使用 `create_deep_agent` 提供模型—工具循环，使用 LangChain Middleware 承载完整业务流程。

## 当前流程

1. `before_agent` 处理上传、危急/高风险分流、医疗领域范围判断、症状抽取和任务模式分类；
2. `wrap_model_call` 按模式开放工具并设置首步 `tool_choice`；
3. `wrap_tool_call` 在执行前再次校验模式白名单；
4. `before_model` 在每批工具结果返回后记录轨迹，并执行默认 6 批次上限；
5. `after_agent` 统一处理真实引用、无引用降级、违规表述和免责声明。

`fast_rag` 开放 `medical_rag_search`、`get_department_recommend` 两个工具。`agentic` 开放 8 个业务工具，可处理报告整理、多症状综合、概念对比和就医准备。

## 检索

- BGE-M3 + Chroma 向量召回 Top-8；
- jieba + BM25 关键词召回 Top-8；
- RRF 常量 60；
- 可选 BGE Reranker；
- 最终默认返回 Top-5。

这些数字是默认工程配置，不是最优性结论。

## 运行

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Agent Server 导出对象位于 `medical/graph.py` 的 `graph`。CLI、Streamlit 与评估器均通过 `build_medical_agent()` 创建同一当前 Agent。

## 验证口径

当前 98 项自动化测试全部通过，其中 87 项覆盖业务与能力，11 项覆盖 Deep Agents 运行时与界面集成。测试证明确定性逻辑和已覆盖集成路径，不等于医学准确率、临床安全或生产稳定性。

当前评估结果与限制见 [docs/evaluation-current.md](docs/evaluation-current.md)。面试主文档见 [docs/interview_current/00_阅读入口.md](docs/interview_current/00_阅读入口.md)。

危急和高风险规则优先执行并直接进入固定处置；严格匹配的纯寒暄/能力咨询固定结束。其余未被安全硬规则终止的输入全部调用小模型，以 `SafetyIntent` 同时输出领域和安全意图。安全意图包含 `emergency / diagnosis_request / medication_request / normal / uncertain`；安全置信度低于 0.75、领域置信度低于 0.65、类别非法或模型异常时固定追问澄清。非医疗请求经小模型确认后固定结束，不调用主模型或业务工具。

## 当前边界

项目尚未实现生产级身份认证、用户/租户 ACL、持久化多实例 Agent Server、临床专家审核和线上效果验证。`thread_id` 目录隔离不等同于用户授权；6 个工具批次也不等同于总时间、token 或费用预算。
