"""新增 Agent 能力（任务 1-4）的回归测试：不依赖外部 LLM / 真实索引。

覆盖：
- 检索失败信号与重复查询拦截（medical_rag_search + InjectedState）
- Plan-and-Execute（create_task_plan / update_task_progress）
- 证据充分性自检（check_evidence_sufficiency）
- 信息缺口评估与跨轮追问去重（assess_information_gaps）
- 图状态簿记（record_tool_round_node 维护 attempted_queries / agent_plan / asked_questions）
- 输出审核不再误伤 agentic 结果（缺陷 #16 修复）
- 生产路径集成：真实图下 InjectedState 确实注入 attempted_queries
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import medical.graph as G
from medical.graph import output_check_node, record_tool_round_node
from medical.tools import (
    assess_information_gaps,
    check_evidence_sufficiency,
    create_task_plan,
    medical_rag_search,
    update_task_progress,
)


def _fake_retriever(results=None):
    retriever = MagicMock()
    retriever.ready.return_value = True
    retriever.search.return_value = results if results is not None else []
    return retriever


def _toolmsg(name, content, tool_call_id="x"):
    if isinstance(content, dict | list):
        content = json.dumps(content, ensure_ascii=False)
    return ToolMessage(content=content, name=name, tool_call_id=tool_call_id)


class RetrievalFailureSignalTests(unittest.TestCase):
    """medical_rag_search 必须能向 Agent 传达三种失败状态并拦截重复检索。"""

    def test_ok_returns_corpus_body(self):
        res = [
            {
                "text": "糖尿病是一种代谢疾病。",
                "metadata": {"source": "默沙东", "chapter": "糖尿病"},
            }
        ]
        with patch("medical.tools.get_retriever", return_value=_fake_retriever(res)):
            out = medical_rag_search.invoke({"query": "糖尿病症状"})
        self.assertIn("（来源：", out)

    def test_insufficient_when_empty(self):
        with patch("medical.tools.get_retriever", return_value=_fake_retriever([])):
            out = json.loads(medical_rag_search.invoke({"query": "罕见的基因病"}))
        self.assertEqual(out["status"], "insufficient")
        self.assertIn("suggested_actions", out)

    def test_partial_when_low_recall(self):
        res = [{"text": "血糖检查", "metadata": {"source": "S"}}]
        with patch("medical.tools.get_retriever", return_value=_fake_retriever(res)):
            out = json.loads(medical_rag_search.invoke({"query": "血糖检查", "top_k": 6}))
        self.assertEqual(out["status"], "partial")
        self.assertEqual(out["matched"], 1)

    def test_duplicate_query_blocked_without_extra_search(self):
        fake = _fake_retriever([])
        with patch("medical.tools.get_retriever", return_value=fake):
            first = json.loads(
                medical_rag_search.invoke(
                    {"query": "糖化血红蛋白 HbA1c", "state": {"attempted_queries": []}}
                )
            )
            self.assertEqual(first["status"], "insufficient")
            calls_before = fake.search.call_count
            second = json.loads(
                medical_rag_search.invoke(
                    {
                        "query": "糖化血红蛋白HbA1c",
                        "state": {"attempted_queries": ["糖化血红蛋白  HbA1c  "]},
                    }
                )
            )
            self.assertEqual(second["status"], "duplicate_query")
            # 重复查询不应再消耗一次检索预算
            self.assertEqual(fake.search.call_count, calls_before)

    def test_different_query_not_blocked(self):
        with patch("medical.tools.get_retriever", return_value=_fake_retriever([])):
            out = json.loads(
                medical_rag_search.invoke(
                    {"query": "血糖检查项目", "state": {"attempted_queries": ["糖化血红蛋白"]}}
                )
            )
        self.assertEqual(out["status"], "insufficient")


class PlanAndExecuteTests(unittest.TestCase):
    def test_create_plan_success(self):
        out = json.loads(
            create_task_plan.invoke({"steps": ["检索糖尿病", "检索高血压", "对比分析"]})
        )
        self.assertEqual(out["status"], "plan_created")
        self.assertEqual([s["id"] for s in out["plan"]], [1, 2, 3])
        self.assertTrue(all(s["status"] == "pending" for s in out["plan"]))

    def test_create_plan_rejects_empty(self):
        out = json.loads(create_task_plan.invoke({"steps": []}))
        self.assertEqual(out["status"], "rejected")

    def test_create_plan_rejects_too_many(self):
        out = json.loads(create_task_plan.invoke({"steps": [f"步{i}" for i in range(9)]}))
        self.assertEqual(out["status"], "rejected")

    def test_update_progress_success(self):
        plan_state = {
            "agent_plan": [{"id": 1, "description": "检索", "status": "pending", "note": ""}]
        }
        out = json.loads(
            update_task_progress.invoke(
                {"step_id": 1, "status": "done", "note": "已检索", "state": plan_state}
            )
        )
        self.assertEqual(out["status"], "plan_updated")
        self.assertEqual(out["step"]["status"], "done")

    def test_update_progress_invalid_status(self):
        plan_state = {
            "agent_plan": [{"id": 1, "description": "检索", "status": "pending", "note": ""}]
        }
        out = json.loads(
            update_task_progress.invoke({"step_id": 1, "status": "banana", "state": plan_state})
        )
        self.assertEqual(out["status"], "invalid_status")

    def test_update_progress_unknown_step(self):
        plan_state = {
            "agent_plan": [{"id": 1, "description": "检索", "status": "pending", "note": ""}]
        }
        out = json.loads(
            update_task_progress.invoke({"step_id": 9, "status": "done", "state": plan_state})
        )
        self.assertEqual(out["status"], "unknown_step")

    def test_update_without_plan(self):
        out = json.loads(
            update_task_progress.invoke(
                {"step_id": 1, "status": "done", "state": {"agent_plan": []}}
            )
        )
        self.assertEqual(out["status"], "no_plan")


class EvidenceSufficiencyTests(unittest.TestCase):
    @staticmethod
    def _state_with_evidence(text: str):
        return {
            "messages": [
                HumanMessage(content="复杂问题"),
                _toolmsg("medical_rag_search", text, "rag-1"),
            ]
        }

    def test_invalid_without_sub_questions(self):
        out = json.loads(
            check_evidence_sufficiency.invoke(
                {"sub_questions": [], "state": self._state_with_evidence("资料")}
            )
        )
        self.assertEqual(out["status"], "invalid")

    def test_insufficient_without_evidence(self):
        out = json.loads(
            check_evidence_sufficiency.invoke(
                {"sub_questions": ["糖尿病症状", "高血压症状"], "state": {"messages": []}}
            )
        )
        self.assertFalse(out["ready"])
        self.assertEqual(out["missing"], ["糖尿病症状", "高血压症状"])

    def test_ready_when_covered(self):
        digest = "糖尿病常见的症状包括多饮多尿；高血压常见头晕头痛。"
        out = json.loads(
            check_evidence_sufficiency.invoke(
                {
                    "sub_questions": ["糖尿病症状", "高血压症状"],
                    "state": self._state_with_evidence(digest),
                }
            )
        )
        self.assertTrue(out["ready"])
        self.assertEqual(out["coverage"], "2/2")

    def test_missing_when_partial(self):
        digest = "糖尿病常见的症状包括多饮多尿。"
        out = json.loads(
            check_evidence_sufficiency.invoke(
                {
                    "sub_questions": ["糖尿病症状", "高血压症状"],
                    "state": self._state_with_evidence(digest),
                }
            )
        )
        self.assertFalse(out["ready"])
        self.assertEqual(out["missing"], ["高血压症状"])

    def test_shared_disease_name_does_not_fake_coverage(self):
        out = json.loads(
            check_evidence_sufficiency.invoke(
                {
                    "sub_questions": ["糖尿病症状"],
                    "state": self._state_with_evidence("糖尿病饮食管理需要关注总能量。"),
                }
            )
        )
        self.assertFalse(out["ready"])


class GapAssessmentDedupTests(unittest.TestCase):
    def test_fresh_gaps_reported(self):
        out = json.loads(
            assess_information_gaps.invoke(
                {
                    "task_goal": "visit_preparation",
                    "symptoms": ["头痛"],
                    "state": {"asked_questions": []},
                }
            )
        )
        self.assertIn("duration", out["missing_fields"])
        self.assertIn("age", out["missing_fields"])
        self.assertTrue(out["suggested_questions"])

    def test_already_asked_excluded(self):
        out = json.loads(
            assess_information_gaps.invoke(
                {
                    "task_goal": "visit_preparation",
                    "symptoms": ["头痛"],
                    "state": {"asked_questions": ["duration", "age"]},
                }
            )
        )
        self.assertEqual(out["missing_fields"], ["duration", "age"])  # 仍是缺口
        self.assertEqual(out["suggested_questions"], [])  # 但不再追问
        self.assertEqual(out["already_asked"], ["duration", "age"])
        self.assertIn("不要再重复问", out["instruction"])


class RecordToolRoundNodeTests(unittest.TestCase):
    """图节点必须准确维护 attempted_queries / agent_plan / asked_questions。"""

    def test_bookkeeping(self):
        state = {
            "messages": [
                HumanMessage(content="对比糖尿病和高血压"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "medical_rag_search",
                            "args": {"query": "糖尿病"},
                            "id": "a1",
                            "type": "tool_call",
                        },
                        {
                            "name": "create_task_plan",
                            "args": {"steps": ["检索糖尿病", "检索高血压", "对比"]},
                            "id": "a2",
                            "type": "tool_call",
                        },
                        {
                            "name": "update_task_progress",
                            "args": {"step_id": 1, "status": "done", "note": "ok"},
                            "id": "a3",
                            "type": "tool_call",
                        },
                        {
                            "name": "assess_information_gaps",
                            "args": {"task_goal": "comparison"},
                            "id": "a4",
                            "type": "tool_call",
                        },
                    ],
                ),
                _toolmsg("medical_rag_search", "（来源：默沙东）", "a1"),
                _toolmsg(
                    "create_task_plan",
                    {
                        "status": "plan_created",
                        "plan": [
                            {"id": 1, "description": "检索糖尿病", "status": "pending", "note": ""},
                            {"id": 2, "description": "检索高血压", "status": "pending", "note": ""},
                        ],
                    },
                    "a2",
                ),
                _toolmsg(
                    "update_task_progress",
                    {"status": "plan_updated", "step": {"id": 1, "status": "done", "note": "ok"}},
                    "a3",
                ),
                _toolmsg(
                    "assess_information_gaps",
                    {
                        "missing_fields": ["comparison_target", "age"],
                        "next_question_field": "comparison_target",
                    },
                    "a4",
                ),
            ],
            "attempted_queries": [],
            "agent_plan": [],
            "asked_questions": [],
            "agent_tool_rounds": 0,
            "agent_tool_trace": [],
        }
        out = record_tool_round_node(state)
        self.assertEqual(out["attempted_queries"], ["糖尿病"])
        self.assertEqual(out["agent_tool_rounds"], 1)
        self.assertIn("medical_rag_search", out["agent_tool_trace"])
        updated = next(s for s in out["agent_plan"] if s["id"] == 1)
        self.assertEqual(updated["status"], "done")
        self.assertEqual(out["asked_questions"], ["comparison_target"])

    def test_rejected_plan_is_not_written_to_state(self):
        steps = [f"步骤{i}" for i in range(9)]
        state = {
            "messages": [
                HumanMessage(content="复杂任务"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "create_task_plan",
                            "args": {"steps": steps},
                            "id": "p1",
                            "type": "tool_call",
                        }
                    ],
                ),
                _toolmsg("create_task_plan", {"status": "rejected"}, "p1"),
            ],
            "agent_plan": [],
        }
        self.assertEqual(record_tool_round_node(state)["agent_plan"], [])


class OutputCheckAgenticTests(unittest.TestCase):
    """缺陷 #16 修复：agentic 多工具结果不应被"无引用降级"整条抹掉。"""

    def test_agentic_structured_result_preserved(self):
        state = {
            "messages": [
                HumanMessage(content="帮我整理就医准备"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "build_visit_preparation",
                            "args": {},
                            "id": "b1",
                            "type": "tool_call",
                        },
                        {
                            "name": "medical_rag_search",
                            "args": {"query": "糖尿病"},
                            "id": "b2",
                            "type": "tool_call",
                        },
                    ],
                ),
                _toolmsg("build_visit_preparation", "就医准备清单：建议携带既往病历。", "b1"),
                _toolmsg("medical_rag_search", {"status": "insufficient", "matched": 0}, "b2"),
                AIMessage(
                    content="根据您提供的信息，建议准备以下材料，并在就诊时向医生说明症状持续时间。"
                ),
            ]
        }
        out = output_check_node(state)
        final = out["messages"][-1].content
        # 主体被保留，仅追加资料缺口说明
        self.assertIn("建议准备以下材料", final)
        self.assertNotIn("知识库中未检索到可验证的权威资料，因此我不能补充医学事实性内容。", final)
        self.assertIn("未获得可引用的权威资料", final)

    def test_fast_rag_pure_retrieval_still_wipes(self):
        state = {
            "messages": [
                HumanMessage(content="什么是地中海贫血"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "medical_rag_search",
                            "args": {"query": "地中海贫血"},
                            "id": "c1",
                            "type": "tool_call",
                        }
                    ],
                ),
                _toolmsg("medical_rag_search", {"status": "insufficient", "matched": 0}, "c1"),
                AIMessage(content="地中海贫血是一种……（来源：未知）"),
            ]
        }
        out = output_check_node(state)
        final = out["messages"][-1].content
        self.assertIn("知识库中未检索到可验证的权威资料", final)

    def test_citations_collected_only_from_rag(self):
        state = {
            "messages": [
                HumanMessage(content="糖尿病症状"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "medical_rag_search",
                            "args": {"query": "糖尿病症状"},
                            "id": "d1",
                            "type": "tool_call",
                        },
                        {
                            "name": "get_department_recommend",
                            "args": {"symptoms": ["多饮"]},
                            "id": "d2",
                            "type": "tool_call",
                        },
                    ],
                ),
                _toolmsg("medical_rag_search", "正文\n（来源：默沙东手册）", "d1"),
                _toolmsg("get_department_recommend", "推荐科室：内分泌科", "d2"),
                AIMessage(content="糖尿病常见症状包括多饮多尿。"),
            ]
        }
        out = output_check_node(state)
        self.assertIn("默沙东手册", out["citations"][0])

    def test_partial_json_result_keeps_citation(self):
        state = {
            "messages": [
                HumanMessage(content="血糖检查"),
                _toolmsg(
                    "medical_rag_search",
                    {
                        "status": "partial",
                        "results": "正文\n（来源：默沙东手册-血糖检查）",
                    },
                    "partial-1",
                ),
                AIMessage(content="血糖检查可用于了解血糖水平。"),
            ]
        }
        out = output_check_node(state)
        self.assertEqual(len(out["citations"]), 1)
        self.assertIn("默沙东手册-血糖检查", out["messages"][-1].content)


class ProductionInjectionTests(unittest.TestCase):
    """端到端确认：编译后的真实图会把 attempted_queries 通过 InjectedState 注入工具。"""

    class _ScriptedLLM:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools, tool_choice=None):
            self._tools = tools
            return self

        def invoke(self, messages):
            self.calls += 1
            if self.calls >= 3:
                return AIMessage(content="总结：知识库暂无糖尿病相关资料。")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "medical_rag_search",
                        "args": {"query": "糖尿病"},
                        "id": f"c{self.calls}",
                        "type": "tool_call",
                    }
                ],
            )

    def test_duplicate_query_intercepted_in_real_graph(self):
        from langgraph.checkpoint.memory import MemorySaver

        fake = _fake_retriever([])
        llm = self._ScriptedLLM()
        with (
            patch("medical.tools.get_retriever", return_value=fake),
            patch("medical.graph.get_llm", return_value=llm),
            patch("medical.graph.get_small_llm") as small_llm,
        ):
            structured = MagicMock()
            structured.invoke.return_value = G.SymptomInfo()
            small_llm.return_value.with_structured_output.return_value = structured
            app = G.build_medical_graph(checkpointer=MemorySaver())
            res = app.invoke(
                {
                    "messages": [
                        HumanMessage(content="帮我对比糖尿病和高血压的异同，并整理就医准备")
                    ]
                },
                {"configurable": {"thread_id": "inject-test"}},
            )
        tool_msgs = [m for m in res["messages"] if getattr(m, "name", None) == "medical_rag_search"]
        statuses = []
        for m in tool_msgs:
            try:
                statuses.append(json.loads(m.content).get("status"))
            except Exception:
                statuses.append("raw")
        self.assertIn("duplicate_query", statuses)
