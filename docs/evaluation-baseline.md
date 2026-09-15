# 开发评估基线

本页记录可公开复现的开发回归口径。数据全部来自
`data/eval_dataset.json` 中的合成用例，不包含真实患者记录，也不代表临床准确率或线上生产效果。

## 当前快速回归

| 模块 | 样本量 | 当前结果 | 适用范围 |
|---|---:|---:|---|
| 高危请求规则拦截 | 13 | 13/13 | 仅说明固定开发用例全部命中 |
| 症状实体抽取 | 7 | Precision/Recall/F1 = 1.00 | 小样本、同分布开发回归 |
| 科室规则路由 | 14 | 14/14 | 仅覆盖当前规则表中的代表性用例 |

高危规则和科室路由是本地确定性逻辑；症状实体抽取通过 `symptom_extract` 调用小模型。表中抽取结果是一次已保存运行的开发基线，会随模型、供应商和提示词变化，不是无需模型即可复现的固定结果。

## 自动化测试

`python -m unittest discover -s tests -v` 当前执行 77 项测试：

- `tests/test_core.py`：34 项；
- `tests/test_agent_capabilities.py`：26 项；
- `tests/test_agentic_eval.py`：15 项；
- `tests/test_logging_config.py`：2 项。

这些测试不依赖真实外部模型，主要验证安全分流、双模式路由、工具权限、计划状态、
证据注入、引用隔离、文件边界、日志脱敏和 Agentic 指标计算。

## Agentic 评估状态

项目已准备 9 条合成 Agentic 场景数据，覆盖报告整理、概念对比和就医准备，
并定义完成状态、工具调用条数、重复检索、工具覆盖、缺口评估调用次数和计划步骤数 6 项轨迹指标。代码字段仍名为 `tool_rounds` 和 `clarification_rounds`，但前者按每个 tool call 计数（并行调用也分别计数），后者按 `assess_information_gaps` 调用计数，并不证明问题已实际展示给用户。`plan_steps` 在逐例 metrics 中计算，当前报告汇总区未输出。

运行 `python evaluate.py --full` 时会调用真实模型和本地知识库生成完整报告。
在保存并复核完整运行结果之前，不宣称 Agentic 完成率、线上效果或真实用户收益。

## 仍需补充

- 独立检索 golden set 及 Recall@K、MRR、NDCG；
- 独立盲测集与对抗样本；
- 医学专业人员审核；
- 明确模型、索引版本、硬件、时间及 P50/P95 延迟；
- 线上 bad case 回流、成本和稳定性指标。
