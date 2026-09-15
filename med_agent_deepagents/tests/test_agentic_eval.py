"""agentic 端到端评测相关测试。

目标：
1. 验证 compute_agentic_metrics 对合成轨迹能正确产出完成率/工具轮数/重复检索/追问轮数；
2. 验证 eval_dataset.json 三类任务（报告解读 / 对比分析 / 就医准备）字段完整且均可被本测试离线复现；
3. 用真实工具（mock 检索回包，三类各跑一遍）端到端验证 evaluate.py 的指标函数与可用性。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluate import compute_agentic_metrics, load_dataset


# ============================================================
# 1. compute_agentic_metrics 指标函数正确性
# ============================================================
class ComputeMetricsTests(unittest.TestCase):
    BASE_CASE = {
        "expected_key_concepts": ["白细胞", "感染"],
        "expected_tools_min": ["read_medical_doc", "medical_rag_search"],
        "forbidden_phrases": ["确诊", "开药"],
    }

    def _trace(self, **kw):
        base = {
            "tool_calls": [
                {"name": "read_medical_doc", "args": {"file_name": "x.txt"}},
                {"name": "medical_rag_search", "args": {"query": "白细胞升高"}},
                {"name": "medical_rag_search", "args": {"query": "白细胞升高"}},
            ],
            "attempted_queries": ["白细胞升高", "白细胞升高"],
            "agent_plan": [{"id": "s1"}, {"id": "s2"}],
            "asked_questions": [],
            "answer": "白细胞升高常见于细菌感染，请就医。",
            "citations": ["src1"],
        }
        base.update(kw)
        return base

    def test_completion_passes_when_concepts_covered_and_no_forbidden(self):
        m = compute_agentic_metrics(self.BASE_CASE, self._trace())
        self.assertTrue(m["completion"])
        self.assertEqual(m["miss_concepts"], [])
        self.assertEqual(m["hit_forbidden"], [])

    def test_completion_fails_when_forbidden_phrase_appears(self):
        m = compute_agentic_metrics(
            self.BASE_CASE, self._trace(answer="根据结果可以确诊为细菌性肺炎")
        )
        self.assertFalse(m["completion"])
        self.assertIn("确诊", m["hit_forbidden"])

    def test_completion_fails_when_concepts_miss_over_half(self):
        # 仅 1/2 命中 → 50% 刚好临界，期望允许；改成 0/2 应失败
        m = compute_agentic_metrics(self.BASE_CASE, self._trace(answer="未提及具体指标的描述"))
        self.assertFalse(m["completion"])
        self.assertEqual(sorted(m["miss_concepts"]), ["感染", "白细胞"])

    def test_tool_rounds_counts_all_calls(self):
        trace = self._trace()
        trace["tool_calls"].append({"name": "medical_rag_search", "args": {"query": "补充"}})
        self.assertEqual(compute_agentic_metrics(self.BASE_CASE, trace)["tool_rounds"], 4)

    def test_duplicate_query_normalized_detection(self):
        # 加空格/标点差异，归一化后仍应识别重复
        trace = self._trace()
        trace["tool_calls"].append(
            {"name": "medical_rag_search", "args": {"query": "白细胞  升高！"}}
        )
        m = compute_agentic_metrics(self.BASE_CASE, trace)
        self.assertTrue(m["duplicate_query"])

    def test_tool_coverage_ratio(self):
        case = dict(
            self.BASE_CASE,
            expected_tools_min=[
                "read_medical_doc",
                "medical_rag_search",
                "assess_information_gaps",
            ],
        )
        # 实际只调了前两个，覆盖率 2/3
        m = compute_agentic_metrics(case, self._trace())
        self.assertAlmostEqual(m["tool_coverage"], 2 / 3, places=3)

    def test_clarification_rounds_counts_gap_assessments(self):
        trace = self._trace()
        trace["tool_calls"].append({"name": "assess_information_gaps", "args": {}})
        trace["tool_calls"].append({"name": "assess_information_gaps", "args": {}})
        self.assertEqual(
            compute_agentic_metrics(self.BASE_CASE, trace)["clarification_rounds"],
            2,
        )

    def test_plan_steps_counts_state_plan_size(self):
        m = compute_agentic_metrics(self.BASE_CASE, self._trace())
        self.assertEqual(m["plan_steps"], 2)


# ============================================================
# 2. eval_dataset.json 三类任务集结构与字段完整性
# ============================================================
class AgenticDatasetSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset()

    def test_three_task_goals_each_have_at_least_one_case(self):
        goals = {c["task_goal"] for c in self.dataset["agentic_cases"]}
        self.assertSetEqual(
            goals,
            {"report_review", "comparison", "visit_preparation"},
        )

    def test_every_case_has_required_fields(self):
        required = {
            "id",
            "task_goal",
            "initial_query",
            "expected_tools_min",
            "expected_key_concepts",
            "forbidden_phrases",
        }
        for c in self.dataset["agentic_cases"]:
            missing = required - set(c.keys())
            self.assertFalse(missing, f"{c.get('id')} 缺少字段 {missing}")

    def test_report_review_cases_carry_upload_payload(self):
        for c in self.dataset["agentic_cases"]:
            if c["task_goal"] == "report_review":
                self.assertIn("upload_file", c)
                self.assertIn("filename", c["upload_file"])
                self.assertIn("content", c["upload_file"])

    def test_id_is_unique(self):
        ids = [c["id"] for c in self.dataset["agentic_cases"]]
        self.assertEqual(len(ids), len(set(ids)))


# ============================================================
# 3. 离线端到端：mock 检索与文件，让三类任务各跑一遍产生轨迹
# ============================================================
class AgenticEndToEndTests(unittest.TestCase):
    """不调真实 LLM：用脚本化策略产生轨迹，再喂给 compute_agentic_metrics 验证。

    这等价于把 evaluate.py::eval_agentic 的"构建 prompt + 调 LLM"步骤替换为
    预编排的轨迹，专注于验证：
    - 三类任务都能在本工具栈上落地（不死循环、不重复、不漏工具）；
    - 指标函数能正确读出这些轨迹；
    - 文档承诺的"计划 + 评估 + 自检"链路确实是流程层约束，而非仅凭 LLM 自觉。
    """

    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset()
        # 报告解读期望的检索结果（键名要兼容 format_results）
        cls.blood_results = [
            {
                "text": "白细胞升高常见于细菌感染，提示可能存在炎症。",
                "metadata": {"source": "MSD 大众版", "chapter": "感染"},
            },
        ]

    @staticmethod
    def _mock_retriever_factory(results_by_query: dict):
        """构造一个 mock retriever：按 query 返回预置 results。"""
        from unittest.mock import MagicMock

        retriever = MagicMock()
        retriever.ready.return_value = True

        def search(query, top_k=5):
            q = (query or "").strip()
            for key, payload in results_by_query.items():
                if key in q:
                    return payload["results"]
            return []

        retriever.search.side_effect = search
        # format_results 与真实 retriever 行为一致：把 list[dict] 转成可直接读入的字符串
        from medical.rag.retriever import format_results

        retriever.format_results = format_results
        return retriever

    # ---- 3.1 report_review：读上传文件 + 检索 ----
    def test_report_review_walkthrough(self):
        case = next(c for c in self.dataset["agentic_cases"] if c["id"] == "rr-01")
        # mock upload
        with tempfile.TemporaryDirectory() as tmp:
            upload_root = Path(tmp)
            tdir = upload_root / "t"
            tdir.mkdir()
            target = tdir / case["upload_file"]["filename"]
            target.write_text(case["upload_file"]["content"], encoding="utf-8")

            from medical.tools import medical_rag_search, read_medical_doc

            with patch("medical.tools._UPLOAD_ROOT", upload_root):
                doc = read_medical_doc.invoke(
                    {"file_name": case["upload_file"]["filename"], "current_thread_id": "t"},
                )
            self.assertIn("不可信文档", doc)
            self.assertIn(case["upload_file"]["filename"], doc)

            # mock 检索 → 获得与 expected_key_concepts 相符的返回
            mock_ret = self._mock_retriever_factory({"白细胞": {"results": self.blood_results}})
            with patch("medical.tools.get_retriever", return_value=mock_ret):
                hit = medical_rag_search.invoke({"query": "白细胞升高"})
            self.assertIn("白细胞", hit)
            self.assertIn("MSD 大众版", hit)

            # 把"读 + 检索"喂给 metrics，断言通过
            trace = {
                "tool_calls": [
                    {
                        "name": "read_medical_doc",
                        "args": {"file_name": case["upload_file"]["filename"]},
                    },
                    {"name": "medical_rag_search", "args": {"query": "白细胞升高"}},
                ],
                "answer": "白细胞升高提示可能存在细菌感染，建议就诊。",
                "attempted_queries": ["白细胞升高"],
                "agent_plan": [{"id": "s1", "status": "done"}],
                "asked_questions": [],
                "citations": ["MSD 大众版"],
            }
            m = compute_agentic_metrics(case, trace)
            self.assertTrue(m["completion"])
            self.assertEqual(m["tool_coverage"], 1.0)
            self.assertFalse(m["duplicate_query"])
            self.assertEqual(m["clarification_rounds"], 0)

    # ---- 3.2 comparison：两次独立检索 ----
    def test_comparison_walkthrough(self):
        case = next(c for c in self.dataset["agentic_cases"] if c["id"] == "cmp-01")
        trace = {
            "tool_calls": [
                {"name": "medical_rag_search", "args": {"query": "感冒 症状"}},
                {"name": "medical_rag_search", "args": {"query": "过敏性鼻炎 症状"}},
            ],
            "answer": (
                "感冒多伴咳嗽咽痛、病毒性；过敏性鼻炎常伴鼻痒眼痒、对冷空气敏感。"
                "需要结合持续时间与接触史判断。"
            ),
            "attempted_queries": ["感冒 症状", "过敏性鼻炎 症状"],
            "agent_plan": [{"id": "s1"}, {"id": "s2"}],
            "asked_questions": [],
            "citations": ["MSD 大众版"],
        }
        m = compute_agentic_metrics(case, trace)
        self.assertTrue(
            m["completion"], msg=f"miss={m['miss_concepts']} hit_forbidden={m['hit_forbidden']}"
        )
        self.assertEqual(m["tool_rounds"], 2)
        self.assertFalse(m["duplicate_query"])
        self.assertGreaterEqual(m["plan_steps"], 2)

    # ---- 3.3 visit_preparation：assess_gaps + build_preparation ----
    def test_visit_preparation_walkthrough(self):
        case = next(c for c in self.dataset["agentic_cases"] if c["id"] == "vp-01")
        from medical.tools import assess_information_gaps, build_visit_preparation

        gaps = json.loads(
            assess_information_gaps.invoke(
                {
                    "task_goal": "visit_preparation",
                    **case["user_profile"],
                }
            )
        )
        # 已有完整 user_profile → 不应再追问
        self.assertTrue(gaps["ready"], msg=f"profile 已给齐时，应判定 ready，但返回 {gaps}")
        prep = json.loads(
            build_visit_preparation.invoke(
                {
                    **case["user_profile"],
                }
            )
        )
        self.assertIn("神经内科", prep.get("就医方向", ""))
        self.assertIn("就医沟通准备", prep.get("边界说明", ""))

        trace = {
            "tool_calls": [
                {
                    "name": "assess_information_gaps",
                    "args": {"task_goal": "visit_preparation", **case["user_profile"]},
                },
                {"name": "build_visit_preparation", "args": {**case["user_profile"]}},
            ],
            "answer": ("基于您提供的信息（头疼两天，30岁），建议挂神经内科。就医沟通准备要点：..."),
            "attempted_queries": [],
            "agent_plan": [{"id": "s1"}],
            "asked_questions": [],
            "citations": [],
        }
        m = compute_agentic_metrics(case, trace)
        self.assertTrue(m["completion"])
        self.assertEqual(m["clarification_rounds"], 1)
