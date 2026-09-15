"""医疗健康科普 Agent —— Web 前端（Streamlit）

合规要求：
- 免责声明强制展示：首次使用必须勾选知情确认才能开始对话；顶部横幅全程常驻；
- 不允许把 AI 回答当作医疗结果：回答卡片固定附带免责声明；
- 会话按 thread_id 隔离，支持断开续聊（LangGraph checkpoint）。

启动：
    streamlit run app.py
"""

import uuid

import streamlit as st

from base.logging_config import configure_logging
from medical.compliance import DISCLAIMER
from medical.secure_store import log_session

configure_logging()

st.set_page_config(page_title="健康科普助手", page_icon="🏥", layout="wide")


# ============================================================
# 全局资源（每个 Streamlit 会话进程只初始化一次）
# ============================================================
@st.cache_resource
def get_graph():
    from medical.graph import build_medical_agent, get_checkpointer

    return build_medical_agent(get_checkpointer())


# ============================================================
# 会话初始化
# ============================================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex[:8]
if "acknowledged" not in st.session_state:
    st.session_state.acknowledged = False

graph = get_graph()
run_cfg = {"configurable": {"thread_id": st.session_state.thread_id}}


# ============================================================
# 顶部：常驻免责声明横幅（不可关闭）
# ============================================================
st.error(f"⚠️ **免责声明**：{DISCLAIMER} 本助手不能诊断疾病、不能开处方、不能推荐个人用药。")

st.title("🏥 健康科普助手")
st.caption("知识来源：默沙东诊疗手册（大众版）· LangChain + LangGraph 实现")


# ============================================================
# 侧边栏：能力边界 + 会话信息 + 症状档案
# ============================================================
with st.sidebar:
    st.header("📋 能力边界")
    st.success(
        "**✅ 我可以**\n- 医学科普、症状知识解释\n- 整理您描述的症状信息\n- 就诊科室建议\n- 病历/检验报告文本整理"
    )
    st.warning(
        "**❌ 我不能**\n- 判断您得了什么病（诊断）\n- 推荐药物、调整药量、开处方\n- 替代医生做诊疗结论"
    )
    st.divider()

    st.header("💬 会话")
    st.caption("会话 ID（thread_id）")
    st.code(st.session_state.thread_id, language=None)
    if st.button("🔄 开启新会话", use_container_width=True):
        st.session_state.thread_id = uuid.uuid4().hex[:8]
        st.session_state.msgs = []
        st.session_state.history_loaded = False
        st.rerun()

    # 从 checkpoint 恢复症状档案（自定义 State，跨轮累积）
    try:
        snap = graph.get_state(run_cfg)
        vals = snap.values or {}
        if vals.get("user_symptoms"):
            st.divider()
            st.header("🩺 本会话症状档案")
            st.markdown("\n".join(f"- {s}" for s in vals["user_symptoms"]))
            if vals.get("user_duration"):
                st.markdown(f"**持续时间**：{vals['user_duration']}")
            for k, label in [
                ("user_age", "年龄"),
                ("user_gender", "性别"),
                ("user_history", "既往病史"),
            ]:
                if vals.get(k):
                    st.markdown(f"**{label}**：{vals[k]}")
            st.caption("档案仅存于本会话，用于多轮问诊上下文")
    except Exception:
        pass


# ============================================================
# 知情确认门槛：未确认前不展示任何对话入口
# ============================================================
if not st.session_state.acknowledged:
    st.divider()
    st.markdown("### 使用前请阅读并确认")
    st.info(
        "1. 本 AI 仅提供**健康科普参考**，不构成医疗建议；\n"
        "2. 身体不适请前往**正规医疗机构**就诊；\n"
        "3. 紧急情况请立即拨打 **120** 或前往急诊；\n"
        "4. 请勿在此输入他人敏感健康信息。"
    )
    if st.checkbox("我已阅读并理解以上声明，知晓本助手不能替代执业医师", key="ack_box"):
        if st.button("开始使用", type="primary", use_container_width=True):
            st.session_state.acknowledged = True
            st.rerun()
    st.stop()


# ============================================================
# 对话区：从 checkpoint 恢复历史（Human/AI 消息）
# ============================================================
if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False
    st.session_state.msgs = []
    try:
        snap = graph.get_state(run_cfg)
        for m in (snap.values or {}).get("messages", []):
            kind = type(m).__name__
            if kind == "HumanMessage":
                st.session_state.msgs.append(("user", m.content))
            elif kind == "AIMessage" and m.content:
                st.session_state.msgs.append(("assistant", m.content))
    except Exception:
        pass
    st.session_state.history_loaded = True

for role, content in st.session_state.msgs:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)

# 危急信号提示（上一轮若触发急诊标记则常驻红幅）
try:
    snap = graph.get_state(run_cfg)
    if (snap.values or {}).get("is_emergency"):
        st.error("🚨 **本会话中出现过危急信号描述：请立即前往急诊就医或拨打 120！**")
except Exception:
    pass

# ============================================================
# 输入与调用
# ============================================================
if prompt := st.chat_input("请输入您的健康科普问题（如：咳嗽流鼻涕挂什么科）"):
    st.session_state.msgs.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在检索权威医学知识库…"):
            try:
                result = graph.invoke({"messages": [{"role": "user", "content": prompt}]}, run_cfg)
            except Exception as e:
                result = None
                st.error(f"处理出错：{e}")
        if result:
            answer = result["messages"][-1].content
            st.markdown(answer)
            st.session_state.msgs.append(("assistant", answer))
            try:
                log_session(
                    st.session_state.thread_id,
                    prompt,
                    answer,
                    result.get("user_symptoms") or [],
                )
            except Exception:
                # 加密审计日志失败不应向用户暴露内部路径或密钥信息。
                pass

            citations = result.get("citations") or []
            if citations:
                with st.expander(f"📚 本次回答引用的权威资料（{len(citations)} 条）"):
                    for c in citations:
                        st.markdown(f"- {c.strip('（）')}")
            if result.get("is_emergency"):
                st.error("🚨 您描述的情况可能属于急危重症，请**立即前往急诊**或拨打 **120**！")
        else:
            st.session_state.msgs.append(("assistant", "抱歉，处理出现问题，请稍后重试。"))
