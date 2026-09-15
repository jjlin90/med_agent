"""医疗 Agent 自定义状态：不依赖消息历史，显式保存用户健康上下文

按需求要求保存：用户主诉症状、持续时间、年龄性别、高危/危急标记、引用来源。
"""

from typing import Annotated, NotRequired

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class MedicalAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_symptoms: list[str]  # 已提取的用户症状（跨轮累积）
    current_turn_symptoms: NotRequired[
        list[str]
    ]  # 仅本轮明确提到；用于路由，避免历史症状污染新任务
    user_duration: str  # 症状持续时间
    user_history: str  # 既往病史/过敏史/用药史
    user_age: int | None  # 用户年龄
    user_gender: str | None  # 用户性别
    is_high_risk: bool  # 是否识别到高危诉求（求诊断、求开药）
    is_emergency: bool  # 是否包含危急信号（需立即急诊）
    risk_reason: str  # 高危原因说明
    domain_scope: NotRequired[str]  # medical / smalltalk / off_topic / uncertain
    domain_scope_source: NotRequired[str]  # rule / semantic / fallback
    domain_scope_confidence: NotRequired[float | None]  # 语义分类置信度；规则命中为 1.0
    domain_scope_reason: NotRequired[str]  # 限长分类依据或固定失败标记
    safety_intent: NotRequired[
        str
    ]  # emergency / diagnosis_request / medication_request / normal / uncertain
    safety_intent_source: NotRequired[str]  # rule / semantic / pending / fallback
    safety_intent_confidence: NotRequired[float | None]
    safety_intent_reason: NotRequired[str]  # 限长语义安全分类依据或固定失败标记
    citations: list[str]  # 本轮回答引用的知识库来源
    # Agent 原生复杂任务状态：每轮重新分类，控制动态工具编排和循环上限
    task_mode: NotRequired[str]  # fast_rag / agentic
    task_goal: NotRequired[str]  # report_review / visit_preparation / comparison 等
    agent_tool_rounds: NotRequired[int]  # 本轮已完成的工具执行轮数
    agent_tool_trace: NotRequired[list[str]]  # 本轮实际调用过的工具名，便于调试/评估
    # Agent 闭环三件套：让"观察 → 改变策略"真正成立所需的运行时状态
    attempted_queries: NotRequired[list[str]]  # 本轮已尝试过的检索查询，用于拦截重复检索
    agent_plan: NotRequired[list[dict]]  # Plan-and-Execute：[{id, description, status, note}]
    asked_questions: NotRequired[list[str]]  # 工具已选中的待问字段；不证明问题已展示
    # 与 LangGraph Agent Chat UI / 通用智能体输入协议对齐
    upload_files: NotRequired[list[dict]]  # UI 本轮提交的 base64 文件，接收后立即清空
    uploaded_files: NotRequired[list[str]]  # 当前 thread 可访问的隔离文件相对路径
    upload_errors: NotRequired[list[str]]  # 本轮文件接收错误
    # 当前会话标识：由 ingest_uploads 从 config 写入，供 read_medical_doc 收紧
    # current-thread 文件路径；它不构成用户身份或 tenant ACL。
    current_thread_id: NotRequired[str]
    context: NotRequired[dict]  # Agent Chat UI artifact/context 扩展字段
    ui: NotRequired[list]  # LangGraph UI 消息扩展字段
