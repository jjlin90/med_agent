"""医疗 Agent 主流程（LangGraph）

流程：用户主诉 → 前置高危拦截 → 症状实体抽取 → 快速 RAG / 复杂任务 Agent
     → 输出违禁词审核（后置校验层）→ 免责声明 → 结束

- 自定义 State 保存症状/年龄/性别等健康上下文，不依赖消息历史；
- 按 thread_id 隔离会话，checkpoint 持久化支持多轮问诊；
- 高危请求（求诊断/求开药）直接拒绝并给出替代帮助。
"""

import json
import logging
import re
import sqlite3

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

from base import config as cfg
from conn.llm import get_llm, get_small_llm
from medical.compliance import DISCLAIMER, audit_output, detect_emergency, detect_high_risk
from medical.prompts import SYSTEM_PROMPT, build_context, build_refusal
from medical.state import MedicalAgentState
from medical.tools import (
    _PLAN_STATUSES,
    AGENT_TOOLS,
    FAST_RAG_TOOLS,
    SymptomInfo,
    _rag_evidence_text,
)
from medical.uploads import save_uploaded_files

logger = logging.getLogger(__name__)


def _message_text(content) -> str:
    """兼容 CLI 字符串和 Agent Chat UI 的标准 content blocks。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def ingest_uploads_node(state: MedicalAgentState, config):
    """接收 Agent Chat UI upload_files，并按 thread_id 保存到隔离目录。"""
    files = state.get("upload_files") or []
    thread_id = str((config.get("configurable") or {}).get("thread_id") or "unscoped")
    # 把会话标识写入 State，供 read_medical_doc 通过 InjectedState 收紧读取根目录
    # 这里只形成 current-thread 路径边界；用户身份和 thread 所有权仍需入口层另行校验。
    if not files:
        return {"current_thread_id": thread_id, "upload_errors": []}
    saved, errors = save_uploaded_files(files, thread_id)
    merged = list(dict.fromkeys((state.get("uploaded_files") or []) + saved))
    return {
        "upload_files": [],
        "uploaded_files": merged,
        "upload_errors": errors,
        "current_thread_id": thread_id,
    }


# ============================================================
# 节点 1：前置高危拦截（输入过滤层）
# ============================================================
def input_guard_node(state: MedicalAgentState):
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    text = _message_text(last_human.content) if last_human else ""
    high_risk, reason = detect_high_risk(text)
    emergency = detect_emergency(text)
    return {"is_high_risk": high_risk, "risk_reason": reason, "is_emergency": emergency}


def route_after_guard(state: MedicalAgentState) -> str:
    # 生命安全优先：复合请求同时包含急症信号和求诊断/求药时，也必须先走零模型急诊路径。
    if state.get("is_emergency"):
        return "emergency"
    if state.get("is_high_risk"):
        return "refuse"
    return "extract"


def refuse_node(state: MedicalAgentState):
    """高危请求拒绝：输出风险提示 + 替代帮助 + 免责声明"""
    text = build_refusal(
        state.get("risk_reason", "请求超出能力边界"), emergency=state.get("is_emergency", False)
    )
    text = text + "\n\n" + DISCLAIMER if DISCLAIMER not in text else text
    return {"messages": [AIMessage(content=text)]}


def emergency_node(state: MedicalAgentState):
    """危急信号必须脱离模型和网络立即响应，避免关键提示因外部服务失败而延迟。"""
    text = (
        "🚨 您描述的情况包含可能的危急信号。请立即拨打 120，或在确保安全的前提下"
        "尽快前往最近的急诊科；不要等待 AI 继续判断，也不要自行驾车。"
        "如果患者失去意识或呼吸异常，请按 120 调度员的指示处理。\n\n" + DISCLAIMER
    )
    return {"messages": [AIMessage(content=text)], "citations": []}


# ============================================================
# 节点 2：症状实体抽取（写入自定义 State，跨轮累积）
# ============================================================
def extract_node(state: MedicalAgentState):
    # 只抽取最新用户消息。把前几轮对话一并交给抽取模型会再次抽出历史症状，
    # 从而把无关的新知识问答错误路由成 multi_symptom。
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    dialogue = _message_text(last_human.content) if last_human else ""
    llm = get_small_llm().with_structured_output(SymptomInfo)
    try:
        info: SymptomInfo = llm.invoke(
            "从下面这段医患对话中抽取最新用户主诉的症状信息，只抽取明确提到的内容：\n" + dialogue
        )
    except Exception as e:
        logger.warning("症状抽取失败，已跳过：%s", e)
        return {"current_turn_symptoms": []}

    updates = {}

    def clean_symptom(value: str) -> str:
        # 部分兼容模型会把实体错误输出成 ".*头痛.*" 一类正则；状态中只保留纯文本实体。
        value = re.sub(r"^[.*^$\\]+|[.*^$\\]+$", "", str(value)).strip()
        return value if 0 < len(value) <= 30 else ""

    old_symptoms = [clean_symptom(s) for s in (state.get("user_symptoms") or [])]
    new_symptoms = [clean_symptom(s) for s in (info.symptoms or [])]
    merged = list(dict.fromkeys(s for s in old_symptoms + new_symptoms if s))
    updates["user_symptoms"] = merged
    updates["current_turn_symptoms"] = [s for s in new_symptoms if s]
    if info.duration and not state.get("user_duration"):
        updates["user_duration"] = info.duration
    if info.past_history and not state.get("user_history"):
        updates["user_history"] = info.past_history
    if info.age:
        updates["user_age"] = info.age
    if info.gender:
        updates["user_gender"] = info.gender
    return updates


# ============================================================
# 节点 3：任务模式识别（普通科普走快速 RAG，开放式任务走 Agent）
# ============================================================
_COMPLEX_TASK_PATTERNS = (
    r"(?:结合|根据).{0,10}(?:报告|病历|检查单|上传|资料|文件)",
    r"(?:综合|全面).{0,12}(?:整理|分析|说明|评估)",
    r"(?:对比|比较|分别解释|异同)",
    r"(?:制定|生成|整理).{0,10}(?:就医|就诊|沟通|准备).{0,6}(?:计划|清单|摘要)",
    r"(?:先.{0,10}再|分步骤|多步骤)",
)


def classify_task_node(state: MedicalAgentState):
    """识别是否需要动态工具编排；高风险路由仍由 input_guard 优先处理。"""
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    text = _message_text(last_human.content) if last_human else ""
    uploaded = state.get("uploaded_files") or []
    # 正常图执行时 extract_node 总会写 current_turn_symptoms；直接单测旧调用方式时
    # 才回退到 user_symptoms。模式路由不能被上一任务遗留的症状数量污染。
    symptoms = state.get("current_turn_symptoms")
    if symptoms is None:
        symptoms = state.get("user_symptoms") or []

    goal = "knowledge_qa"
    if re.search(r"报告|病历|检查单|上传|这份文件|这份资料", text) and uploaded:
        goal = "report_review"
    elif re.search(r"对比|比较|分别解释|异同", text):
        goal = "comparison"
    elif re.search(r"就医|就诊|沟通|准备|清单|摘要", text):
        goal = "visit_preparation"
    elif len(symptoms) >= 3 or re.search(r"综合|全面|多种|多个症状", text):
        goal = "multi_symptom"

    is_complex = (
        goal != "knowledge_qa"
        or len(symptoms) >= 3
        or any(re.search(pattern, text) for pattern in _COMPLEX_TASK_PATTERNS)
    )
    # 每轮用户提问都是一次新任务：轮数、轨迹、已尝试查询、执行计划全部重置，
    # 预算不被上一轮耗尽；但 asked_questions 跨轮保留，避免对同一用户重复追问。
    return {
        "task_mode": "agentic" if is_complex else "fast_rag",
        "task_goal": goal if is_complex else "knowledge_qa",
        "agent_tool_rounds": 0,
        "agent_tool_trace": [],
        "attempted_queries": [],
        "agent_plan": [],
    }


# ============================================================
# 节点 4：Agent 主节点（快速问答固定检索；复杂任务动态编排）
# ============================================================
def agent_node(state: MedicalAgentState):
    system = SYSTEM_PROMPT.format(context=build_context(state))
    # 自定义 State 承担跨轮上下文；模型只接收当前轮，避免复用上一轮资料后跳过本轮检索。
    last_human_idx = max(
        (i for i, m in enumerate(state["messages"]) if isinstance(m, HumanMessage)),
        default=0,
    )
    current_turn = list(state["messages"])[last_human_idx:]
    has_tool_result = any(isinstance(m, ToolMessage) for m in current_turn)
    rag_positions = [
        i
        for i, m in enumerate(current_turn)
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "medical_rag_search"
    ]
    evidence_positions = [
        i
        for i, m in enumerate(current_turn)
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "check_evidence_sufficiency"
    ]
    needs_evidence_check = (
        state.get("task_mode") == "agentic" and rag_positions and not evidence_positions
    )
    available_tools = AGENT_TOOLS if state.get("task_mode") == "agentic" else FAST_RAG_TOOLS
    if needs_evidence_check:
        # agentic 模式中至少经过一次真实证据门；后续是否补检索由结果和 Prompt 决定，
        # 避免“每次补检索都再次强制自检”耗尽有限工具轮次。
        llm = get_llm().bind_tools(AGENT_TOOLS, tool_choice="check_evidence_sufficiency")
    elif has_tool_result:
        llm = get_llm().bind_tools(available_tools)
    elif state.get("task_mode") == "agentic":
        # 复杂任务先评估信息缺口，再由 Agent 根据结果动态选择读文档、拆分检索、
        # 科室推荐或就医准备工具。后续路径无法在开发时固定。
        llm = get_llm().bind_tools(AGENT_TOOLS, tool_choice="assess_information_gaps")
    else:
        # 每个正常请求第一步固定检索，代码层保证医学事实有可追溯资料。
        llm = get_llm().bind_tools(FAST_RAG_TOOLS, tool_choice="medical_rag_search")
    resp = llm.invoke([SystemMessage(content=system)] + current_turn)
    return {"messages": [resp]}


def route_after_agent(state: MedicalAgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "output_check"


def _collect_tool_calls(messages: list, name: str) -> list[dict]:
    """从消息序列中取出指定工具的全部调用参数（含并行调用）。"""
    calls = []
    for m in messages:
        if isinstance(m, AIMessage):
            for call in getattr(m, "tool_calls", None) or []:
                if call.get("name") == name:
                    calls.append(call.get("args") or {})
    return calls


def record_tool_round_node(state: MedicalAgentState):
    """记录本轮真实工具轨迹，为 Agent 评估、可观测性和循环上限提供状态。

    同时把本轮已尝试过的检索查询写入 attempted_queries，使 medical_rag_search
    能在下一轮识别并拦截重复查询——这是"环境反馈驱动策略切换"的状态基础。
    """
    last_human_idx = max(
        (i for i, m in enumerate(state["messages"]) if isinstance(m, HumanMessage)),
        default=-1,
    )
    current = state["messages"][last_human_idx + 1 :]
    tool_names = [
        str(getattr(m, "name", "") or "unknown_tool") for m in current if isinstance(m, ToolMessage)
    ]

    attempted = list(state.get("attempted_queries") or [])
    for args in _collect_tool_calls(current, "medical_rag_search"):
        query = str(args.get("query") or "").strip()
        if query and query not in attempted:
            attempted.append(query)

    tool_results = {}
    for m in current:
        if not isinstance(m, ToolMessage):
            continue
        try:
            payload = json.loads(str(m.content))
        except (ValueError, TypeError, json.JSONDecodeError):
            payload = None
        tool_results[getattr(m, "tool_call_id", None)] = payload

    # 只在工具返回 plan_created/plan_updated 后写 State；进度状态本身仍来自模型参数。
    plan = [dict(s) for s in (state.get("agent_plan") or [])]
    for m in current:
        if not isinstance(m, AIMessage):
            continue
        for call in getattr(m, "tool_calls", None) or []:
            payload = tool_results.get(call.get("id"))
            if not isinstance(payload, dict):
                continue
            if call.get("name") == "create_task_plan" and payload.get("status") == "plan_created":
                plan = [dict(s) for s in (payload.get("plan") or [])]
            elif (
                call.get("name") != "update_task_progress"
                or payload.get("status") != "plan_updated"
            ):
                continue
            else:
                returned = payload.get("step") or {}
                step_id, status = returned.get("id"), returned.get("status")
                for step in plan:
                    if step.get("id") == step_id and status in _PLAN_STATUSES:
                        step["status"] = status
                        step["note"] = str(returned.get("note") or "").strip()

    # 工具一次只建议一个字段，因此仅记录 next_question_field；这表示“已建议”，
    # 不证明后续 AIMessage 已把问题展示给用户。
    asked = list(state.get("asked_questions") or [])
    for m in current:
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "assess_information_gaps":
            try:
                field = json.loads(str(m.content)).get("next_question_field")
            except (ValueError, TypeError):
                continue
            field = str(field or "").strip()
            if field and field not in asked:
                asked.append(field)

    return {
        "agent_tool_rounds": int(state.get("agent_tool_rounds") or 0) + 1,
        "agent_tool_trace": tool_names,
        "attempted_queries": attempted,
        "agent_plan": plan,
        "asked_questions": asked,
    }


def route_after_tools(state: MedicalAgentState) -> str:
    if int(state.get("agent_tool_rounds") or 0) >= cfg.MAX_AGENT_TOOL_ROUNDS:
        return "limit"
    return "agent"


def agent_limit_node(state: MedicalAgentState):
    """工具循环达到上限时安全收束，避免死循环和不可控成本。"""
    trace = "、".join(state.get("agent_tool_trace") or []) or "未记录"
    text = (
        f"本轮复杂任务已执行多步工具调用（{trace}），但仍未能在安全轮数内完成。"
        "我已停止继续自动调用工具。请把问题缩小为一个目标，或补充最关键的信息后再继续。"
    )
    return {"messages": [AIMessage(content=text)]}


# ============================================================
# 节点 5：输出审核（后置校验层：违禁词检测 + 免责声明）
# ============================================================
def output_check_node(state: MedicalAgentState):
    last = state["messages"][-1]
    # 收集本轮引用来源（从工具消息中提取）
    citations = []
    # 只统计最后一条用户消息之后产生的工具结果，避免把上一轮引用串到本轮。
    last_human_idx = max(
        (i for i, m in enumerate(state["messages"]) if isinstance(m, HumanMessage)),
        default=-1,
    )
    current_messages = state["messages"][last_human_idx + 1 :]
    for m in current_messages:
        # 只从知识库检索工具收集引用：read_medical_doc 会把用户上传的报告原文整段塞进消息，
        # 攻击者在报告里写一行"（来源：…）"就能让自己的伪来源出现在参考资料里。
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "medical_rag_search":
            # partial 状态把正文包在 JSON.results 中；先解包，否则转义换行会让引用丢失。
            for line in _rag_evidence_text(m.content).splitlines():
                line = line.strip()
                if line.startswith("（来源：") and line not in citations:
                    citations.append(line)

    raw_content = last.content if isinstance(last.content, str) else str(last.content)
    # 不信任模型自行书写的来源名称；统一移除后附加本轮工具返回的真实来源。
    # 支持“（来源：手册（大众版））”这一层嵌套括号，避免旧正则留下孤立的“）”。
    raw_content = re.sub(
        r"[（(]?来源\s*[:：](?:[^（）()\n]|[（(][^）)\n]*[）)]){1,160}[）)]?",
        "",
        raw_content,
    )
    if citations:
        # 只去掉最外层括号，保留“（章节：…）”的内部配对括号。
        display_citations = [
            c[1:-1] if c.startswith("（") and c.endswith("）") else c for c in citations
        ]
        refs = "\n".join(f"- {c}" for c in display_citations)
        raw_content += f"\n\n参考资料（本轮知识库检索）：\n{refs}"
    else:
        department_results = [
            str(m.content)
            for m in current_messages
            if isinstance(m, ToolMessage) and getattr(m, "name", "") == "get_department_recommend"
        ]
        rag_used = any(
            isinstance(m, ToolMessage) and getattr(m, "name", "") == "medical_rag_search"
            for m in current_messages
        )
        # 整条替换只适用于"本轮基本只做了检索"的场景（fast_rag 或纯检索轮）。
        # agentic 多工具任务里，用户等了好几轮工具调用换来的就医准备/报告整理结果，
        # 不能因为其中一次检索为空就被整条丢掉——保留主体，只追加资料缺口提示。
        structured_names = {
            "build_visit_preparation",
            "create_task_plan",
            "update_task_progress",
            "read_medical_doc",
        }
        used_structured = any(
            isinstance(m, ToolMessage) and getattr(m, "name", "") in structured_names
            for m in current_messages
        )
        if not used_structured:
            if rag_used and not department_results:
                raw_content = "知识库中未检索到可验证的权威资料，因此我不能补充医学事实性内容。"
            elif department_results:
                raw_content = department_results[-1]
        elif rag_used:
            raw_content += (
                "\n\n（说明：本轮部分检索未获得可引用的权威资料，"
                "上文涉及的医学事实性内容以已标注来源的条目为准；"
                "未标注来源的部分仅为您所提供信息的整理，不构成医学结论。）"
            )

    clean, violations = audit_output(raw_content)
    for v in violations:
        logger.warning("输出审核拦截：%s", v)
    # 同 id 覆盖最新 State 中的原消息；中间 checkpoint 历史是否保留审核前状态取决于运行时。
    safe_msg = AIMessage(id=last.id, content=clean, additional_kwargs=last.additional_kwargs)
    return {"messages": [safe_msg], "citations": citations}


# ============================================================
# 组装流程图
# ============================================================
def build_medical_graph(checkpointer=None):
    g = StateGraph(MedicalAgentState)
    g.add_node("ingest_uploads", ingest_uploads_node)
    g.add_node("input_guard", input_guard_node)
    g.add_node("refuse", refuse_node)
    g.add_node("emergency", emergency_node)
    g.add_node("extract", extract_node)
    g.add_node("classify_task", classify_task_node)
    g.add_node(
        "agent",
        agent_node,
        retry_policy=RetryPolicy(
            initial_interval=1.0, backoff_factor=2.0, max_interval=8.0, max_attempts=3
        ),
    )
    g.add_node("tools", ToolNode(AGENT_TOOLS))
    g.add_node("record_tool_round", record_tool_round_node)
    g.add_node("agent_limit", agent_limit_node)
    g.add_node("output_check", output_check_node)

    g.add_edge(START, "ingest_uploads")
    g.add_edge("ingest_uploads", "input_guard")
    g.add_conditional_edges(
        "input_guard",
        route_after_guard,
        {"refuse": "refuse", "emergency": "emergency", "extract": "extract"},
    )
    g.add_edge("refuse", END)
    g.add_edge("emergency", END)
    g.add_edge("extract", "classify_task")
    g.add_edge("classify_task", "agent")
    g.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "output_check": "output_check"}
    )
    g.add_edge("tools", "record_tool_round")
    g.add_conditional_edges(
        "record_tool_round", route_after_tools, {"agent": "agent", "limit": "agent_limit"}
    )
    g.add_edge("agent_limit", "output_check")
    g.add_edge("output_check", END)

    return g.compile(checkpointer=checkpointer)


def get_checkpointer() -> SqliteSaver:
    """会话按 thread_id 隔离并持久化（健康数据目录应做磁盘加密/访问控制）"""
    path = cfg.ensure_in_root(cfg.CHECKPOINT_PATH)
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)


# 供 `langgraph dev` / Agent Server 加载。服务端会负责 thread checkpoint，
# 因此这里不注入命令行/Streamlit 使用的本地 SQLite checkpointer。
graph = build_medical_graph()
