"""基于 Deep Agents 与 LangChain Middleware 的医疗科普 Agent。"""

import asyncio
from typing import NotRequired

from deepagents import create_deep_agent
from deepagents.graph import DeepAgentState
from langchain.agents.middleware import AgentMiddleware, ModelRetryMiddleware, hook_config
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.config import get_config

from base import config as cfg
from medical.state import MedicalAgentState
from medical.tools import AGENT_TOOLS, FAST_RAG_TOOLS


class DeepMedicalState(DeepAgentState, MedicalAgentState):
    recorded_tool_messages: NotRequired[int]


class MedicalModel(BaseChatModel):
    """延迟构造供应商模型，使前置危急拦截不依赖密钥或网络。"""

    @property
    def _llm_type(self):
        return "medical-provider-adapter"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from medical.graph import get_llm

        model = get_llm()
        tools = kwargs.pop("tools", None)
        choice = kwargs.pop("tool_choice", None)
        if tools is not None:
            model = model.bind_tools(tools, tool_choice=choice)
        response = model.invoke(messages)
        return ChatResult(generations=[ChatGeneration(message=response)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await asyncio.to_thread(self._generate, messages, stop, run_manager, **kwargs)


class MedicalWorkflowMiddleware(AgentMiddleware):
    """负责上传、安全与领域分流、模式选择、工具授权、执行记账和后置审核。"""

    state_schema = DeepMedicalState

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state, runtime):
        from medical import graph as steps

        working = dict(state)
        updates = {
            "recorded_tool_messages": 0,
            "citations": [],
            "agent_tool_rounds": 0,
            "agent_tool_trace": [],
            "attempted_queries": [],
            "agent_plan": [],
        }
        working.update(updates)
        for function in (
            lambda s: steps.ingest_uploads_node(s, get_config()),
            steps.input_guard_node,
        ):
            change = function(working)
            working.update(change)
            updates.update(change)
        route = steps.route_after_guard(working)
        if route == "semantic_safety":
            change = steps.semantic_safety_node(working)
            working.update(change)
            updates.update(change)
            route = steps.route_after_guard(working)
        terminal_handlers = {
            "emergency": steps.emergency_node,
            "refuse": steps.refuse_node,
            "smalltalk": steps.smalltalk_response,
            "off_topic": steps.off_topic_response,
            "uncertain": steps.uncertain_domain_response,
        }
        if route in terminal_handlers:
            updates.update(terminal_handlers[route](working))
            updates["jump_to"] = "end"
            return updates
        for function in (steps.extract_node, steps.classify_task_node):
            change = function(working)
            working.update(change)
            updates.update(change)
        return updates

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(self, state, runtime):
        return await asyncio.to_thread(self.before_agent, state, runtime)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        from medical import graph as steps

        messages = state["messages"]
        start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
        count = sum(isinstance(m, ToolMessage) for m in messages[start + 1 :])
        if count <= state.get("recorded_tool_messages", 0):
            return None
        updates = steps.record_tool_round_node(state)
        updates["recorded_tool_messages"] = count
        working = {**state, **updates}
        if steps.route_after_tools(working) == "limit":
            updates.update(steps.agent_limit_node(working))
            updates["jump_to"] = "end"
        return updates

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)

    def _prepare_request(self, request):
        from medical.graph import SYSTEM_PROMPT, build_context

        state = request.state
        start = max(
            (i for i, m in enumerate(state["messages"]) if isinstance(m, HumanMessage)), default=0
        )
        current = state["messages"][start:]
        results = [m for m in current if isinstance(m, ToolMessage)]
        names = {m.name for m in results}
        complex_task = state.get("task_mode") == "agentic"
        tools = AGENT_TOOLS if complex_task else FAST_RAG_TOOLS
        choice = None
        if (
            complex_task
            and "medical_rag_search" in names
            and "check_evidence_sufficiency" not in names
        ):
            choice = "check_evidence_sufficiency"
        elif not results:
            choice = "assess_information_gaps" if complex_task else "medical_rag_search"
        # 过滤 SDK 通用工具，只开放当前模式的 2/8 个业务工具并注入医疗系统提示。
        return request.override(
            messages=current,
            system_message=SystemMessage(
                content=SYSTEM_PROMPT.format(context=build_context(state))
            ),
            tools=tools,
            tool_choice=choice,
        )

    def wrap_model_call(self, request, handler):
        return handler(self._prepare_request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._prepare_request(request))

    def _allowed(self, request):
        tools = AGENT_TOOLS if request.state.get("task_mode") == "agentic" else FAST_RAG_TOOLS
        return request.tool_call["name"] in {tool.name for tool in tools}

    def wrap_tool_call(self, request, handler):
        if not self._allowed(request):
            return ToolMessage(
                content="该工具不在本模式业务白名单内，执行被拒绝。",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
                status="error",
            )
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        if not self._allowed(request):
            return ToolMessage(
                content="该工具不在本模式业务白名单内，执行被拒绝。",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
                status="error",
            )
        return await handler(request)

    def after_agent(self, state, runtime):
        from medical.graph import output_check_node

        if state.get("is_emergency") or state.get("is_high_risk") or state.get(
            "domain_scope"
        ) in {"smalltalk", "off_topic", "uncertain"}:
            return None
        return output_check_node(state)

    async def aafter_agent(self, state, runtime):
        return self.after_agent(state, runtime)


def create_medical_deep_agent(checkpointer=None):
    """创建医疗科普 Agent，并注入业务 State、工具、Middleware 与 checkpoint。"""
    return create_deep_agent(
        model=MedicalModel(),
        tools=AGENT_TOOLS,
        state_schema=DeepMedicalState,
        middleware=[
            MedicalWorkflowMiddleware(),
            ModelRetryMiddleware(max_retries=2, initial_delay=1, max_delay=8, on_failure="error"),
        ],
        checkpointer=checkpointer,
        name="medical_agent",
    ).with_config({"recursion_limit": max(100, cfg.MAX_AGENT_TOOL_ROUNDS * 12)})
