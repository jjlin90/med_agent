"""真实 Deep Agents 编排的契约测试，业务模型用可控响应替代。"""

import asyncio
import base64
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from base import config as cfg
from medical import graph as G
from medical.deep_agent import MedicalWorkflowMiddleware
from medical.tools import AGENT_TOOLS, FAST_RAG_TOOLS


def tool_call(name, args, identifier):
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": identifier, "type": "tool_call"}]
    )


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.bindings = []
        self.inputs = []

    def bind_tools(self, tools, tool_choice=None):
        self.bindings.append(
            ([t.name if hasattr(t, "name") else t["function"]["name"] for t in tools], tool_choice)
        )
        return self

    def invoke(self, messages):
        self.inputs.append(messages)
        return next(self.responses)


class RuntimeTests(unittest.TestCase):
    def run_agent(
        self,
        question,
        responses,
        info=None,
        payload=None,
        checkpointer=None,
        thread="runtime-test",
    ):
        model = ScriptedLLM(responses)
        small = Mock()
        symptom_info = info or G.SymptomInfo()

        def structured_output(schema):
            structured = Mock()
            if schema is G.SafetyIntent:
                structured.invoke.return_value = G.SafetyIntent(
                    domain="medical",
                    risk_intent="normal",
                    domain_confidence=0.99,
                    risk_confidence=0.99,
                    reason="测试中的正常医疗请求",
                )
            else:
                structured.invoke.return_value = symptom_info
            return structured

        small.with_structured_output.side_effect = structured_output
        with (
            patch("medical.graph.get_llm", return_value=model),
            patch("medical.graph.get_small_llm", return_value=small),
        ):
            result = G.build_medical_agent(checkpointer).invoke(
                {"messages": [HumanMessage(content=question)], **(payload or {})},
                {"configurable": {"thread_id": thread}},
            )
        return result, model

    def test_fast_mode_actual_whitelist_and_first_choice(self):
        fake = Mock()
        fake.ready.return_value = True
        fake.search.return_value = [{"text": "检查科普", "metadata": {"source": "测试手册"}}] * 3
        with patch("medical.tools.get_retriever", return_value=fake):
            result, model = self.run_agent(
                "什么是CT",
                [
                    tool_call("medical_rag_search", {"query": "CT"}, "r1"),
                    AIMessage(content="检查科普。（来源：伪造来源）"),
                ],
            )
        self.assertEqual(
            model.bindings[0], ([t.name for t in FAST_RAG_TOOLS], "medical_rag_search")
        )
        self.assertEqual(result["agent_tool_rounds"], 1)
        self.assertIn("测试手册", result["messages"][-1].content)
        self.assertNotIn("伪造来源", result["messages"][-1].content)

    def test_complex_first_choice_and_evidence_gate(self):
        fake = Mock()
        fake.ready.return_value = True
        fake.search.return_value = [{"text": "胃镜科普", "metadata": {"source": "测试手册"}}] * 3
        with patch("medical.tools.get_retriever", return_value=fake):
            result, model = self.run_agent(
                "比较胃镜和呼气试验",
                [
                    tool_call("assess_information_gaps", {"task_goal": "comparison"}, "g1"),
                    tool_call("medical_rag_search", {"query": "胃镜"}, "r1"),
                    tool_call("check_evidence_sufficiency", {"sub_questions": ["胃镜"]}, "e1"),
                    AIMessage(content="依据资料整理。"),
                ],
            )
        self.assertEqual(
            model.bindings[0], ([t.name for t in AGENT_TOOLS], "assess_information_gaps")
        )
        self.assertEqual(model.bindings[2][1], "check_evidence_sufficiency")
        self.assertEqual(
            result["agent_tool_trace"],
            ["assess_information_gaps", "medical_rag_search", "check_evidence_sufficiency"],
        )
        evidence = next(
            m
            for m in result["messages"]
            if isinstance(m, ToolMessage) and m.name == "check_evidence_sufficiency"
        )
        self.assertTrue(json.loads(evidence.content)["ready"])

    def test_plan_and_update_reach_custom_state(self):
        result, _ = self.run_agent(
            "请分步骤整理就医准备清单",
            [
                tool_call(
                    "assess_information_gaps",
                    {"task_goal": "visit_preparation", "symptoms": ["咳嗽"]},
                    "g",
                ),
                tool_call("create_task_plan", {"steps": ["整理资料", "检索概念", "生成清单"]}, "p"),
                tool_call(
                    "update_task_progress", {"step_id": 1, "status": "done", "note": "已整理"}, "u"
                ),
                tool_call("build_visit_preparation", {"symptoms": ["咳嗽"], "age": 30}, "v"),
                AIMessage(content="已整理就医准备。"),
            ],
        )
        self.assertEqual(result["agent_plan"][0]["status"], "done")
        self.assertEqual(result["agent_plan"][0]["note"], "已整理")
        self.assertIn("duration", result["asked_questions"])

    def test_tool_budget_stops_at_current_round_limit(self):
        responses = [
            tool_call("create_task_plan", {"steps": ["整理资料", "检索概念", "就医准备"]}, f"t{i}")
            for i in range(10)
        ]
        with patch.object(cfg, "MAX_AGENT_TOOL_ROUNDS", 3):
            result, model = self.run_agent("请比较检查项目并整理就医准备", responses)
        self.assertEqual(result["agent_tool_rounds"], 3)
        self.assertEqual(len(model.inputs), 3)
        self.assertIn("停止继续自动调用", result["messages"][-1].content)

    def test_parallel_calls_count_as_one_round(self):
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_department_recommend",
                    "args": {"symptoms": ["咳嗽"]},
                    "id": "a",
                    "type": "tool_call",
                },
                {
                    "name": "get_department_recommend",
                    "args": {"symptoms": ["鼻塞"]},
                    "id": "b",
                    "type": "tool_call",
                },
            ],
        )
        result, _ = self.run_agent(
            "咳嗽挂什么科", [response, AIMessage(content="请参考科室指引。")]
        )
        self.assertEqual(result["agent_tool_rounds"], 1)
        self.assertEqual(len(result["agent_tool_trace"]), 2)

    def test_actual_upload_and_injected_thread_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "upload_files": [
                    {
                        "type": "file",
                        "metadata": {"filename": "report.txt"},
                        "data": base64.b64encode("姓名：测试人\n血红蛋白 示例".encode()).decode(),
                    }
                ]
            }
            with (
                patch.object(cfg, "USER_UPLOAD_PATH", directory),
                patch("medical.tools._UPLOAD_ROOT", root),
            ):
                result, _ = self.run_agent(
                    "结合上传报告整理就医准备",
                    [
                        tool_call(
                            "assess_information_gaps",
                            {"task_goal": "report_review", "has_uploaded_document": True},
                            "g",
                        ),
                        tool_call("read_medical_doc", {"file_name": "document_1.txt"}, "r"),
                        AIMessage(content="已经读取报告。"),
                    ],
                    payload=payload,
                    thread="report-session",
                )
            result_msg = next(
                m
                for m in result["messages"]
                if isinstance(m, ToolMessage) and m.name == "read_medical_doc"
            )
            self.assertIn("血红蛋白", result_msg.content)
            self.assertNotIn("测试人", result_msg.content)
            self.assertEqual(result["uploaded_files"], ["report-session/document_1.txt"])

    def test_sdk_extra_tool_is_rejected(self):
        request = Mock()
        request.state = {"task_mode": "agentic"}
        request.tool_call = {"name": "execute", "args": {"command": "anything"}, "id": "blocked"}
        handler = Mock()
        reply = MedicalWorkflowMiddleware().wrap_tool_call(request, handler)
        handler.assert_not_called()
        self.assertEqual(reply.status, "error")

    def test_sync_sqlite_restore_and_new_turn_reset(self):
        model = ScriptedLLM([AIMessage(content="请补充资料。"), AIMessage(content="请补充资料。")])
        small = Mock()
        symptom_results = iter([G.SymptomInfo(symptoms=["咳嗽"], age=30), G.SymptomInfo()])

        def structured_output(schema):
            structured = Mock()
            if schema is G.SafetyIntent:
                structured.invoke.return_value = G.SafetyIntent(
                    domain="medical",
                    risk_intent="normal",
                    domain_confidence=0.99,
                    risk_confidence=0.99,
                    reason="测试中的正常医疗请求",
                )
            else:
                structured.invoke.side_effect = lambda *_args, **_kwargs: next(symptom_results)
            return structured

        small.with_structured_output.side_effect = structured_output
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "state.sqlite")
            with (
                patch("medical.graph.get_llm", return_value=model),
                patch("medical.graph.get_small_llm", return_value=small),
            ):
                connection = sqlite3.connect(database, check_same_thread=False)
                try:
                    app = G.build_medical_agent(SqliteSaver(connection))
                    app.invoke(
                        {"messages": [HumanMessage(content="咳嗽三天，30岁，整理就医准备")]},
                        {"configurable": {"thread_id": "persist"}},
                    )
                finally:
                    connection.close()
                connection = sqlite3.connect(database, check_same_thread=False)
                try:
                    app = G.build_medical_agent(SqliteSaver(connection))
                    result = app.invoke(
                        {"messages": [HumanMessage(content="什么是CT")]},
                        {"configurable": {"thread_id": "persist"}},
                    )
                    self.assertEqual(result["user_symptoms"], ["咳嗽"])
                    self.assertEqual(result["user_age"], 30)
                    self.assertEqual(result["current_turn_symptoms"], [])
                    self.assertEqual(result["task_mode"], "fast_rag")
                    self.assertFalse(app.get_state({"configurable": {"thread_id": "other"}}).values)
                finally:
                    connection.close()

    def test_async_runtime_invocation(self):
        model = ScriptedLLM(
            [
                tool_call("get_department_recommend", {"symptoms": ["咳嗽"]}, "d"),
                AIMessage(content="已整理。"),
            ]
        )
        small = Mock()

        def structured_output(schema):
            structured = Mock()
            if schema is G.SafetyIntent:
                structured.invoke.return_value = G.SafetyIntent(
                    domain="medical",
                    risk_intent="normal",
                    domain_confidence=0.99,
                    risk_confidence=0.99,
                    reason="测试中的正常医疗请求",
                )
            else:
                structured.invoke.return_value = G.SymptomInfo(symptoms=["咳嗽"])
            return structured

        small.with_structured_output.side_effect = structured_output
        with (
            patch("medical.graph.get_llm", return_value=model),
            patch("medical.graph.get_small_llm", return_value=small),
        ):
            app = G.build_medical_agent(MemorySaver())
            result = asyncio.run(
                app.ainvoke(
                    {"messages": [HumanMessage(content="咳嗽挂什么科")]},
                    {"configurable": {"thread_id": "async"}},
                )
            )
        self.assertEqual(result["agent_tool_rounds"], 1)
        self.assertEqual(result["current_thread_id"], "async")

    def test_async_emergency_never_calls_models(self):
        app = G.build_medical_agent()
        with (
            patch("medical.graph.get_llm", side_effect=AssertionError),
            patch("medical.graph.get_small_llm", side_effect=AssertionError),
        ):
            result = asyncio.run(app.ainvoke({"messages": [HumanMessage(content="剧烈胸痛")]}))
        self.assertTrue(result["is_emergency"])
        self.assertIn("120", result["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()
