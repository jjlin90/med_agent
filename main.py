"""医疗 Agent 交互入口

用法：
    python main.py                      # 交互对话（默认新会话）
    python main.py --session demo1      # 指定会话 ID（按 thread_id 隔离，可续聊）
    python main.py --build              # 构建知识库索引
    python main.py --build --topics 300 # 快速构建 300 个主题（调试用）
    python main.py --eval [--full]      # 运行评估
"""

import argparse
import sys
import uuid

from base.logging_config import configure_logging
from medical.graph import graph as agent  # noqa: F401 -- LangGraph Server 公开入口

configure_logging()

# Windows 控制台中文输出
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def run_chat(session_id: str):
    from langchain_core.messages import HumanMessage

    from medical.compliance import DISCLAIMER
    from medical.graph import build_medical_graph, get_checkpointer
    from medical.secure_store import log_session

    app = build_medical_graph(get_checkpointer())

    print("=" * 60)
    print("健康科普助手（默沙东诊疗手册·大众版知识库）")
    print(f"免责声明：{DISCLAIMER}")
    print("我不能：诊断疾病、推荐药物、开具处方")
    print("我可以：医学科普 / 症状整理 / 就诊科室建议 / 报告资料整理")
    print("输入 exit 退出，输入 new 开启新会话")
    print("=" * 60)

    while True:
        user_input = input("\n你：").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "new":
            session_id = uuid.uuid4().hex[:8]
            print(f"（已开启新会话 {session_id}）")
            continue

        config = {"configurable": {"thread_id": session_id}}
        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)

        answer = result["messages"][-1].content
        print(f"\n助手：{answer}")
        if result.get("user_symptoms"):
            print(f"\n[会话症状档案] {'、'.join(result['user_symptoms'])}")
        if result.get("is_emergency"):
            print("[!] 检测到危急信号描述，已建议立即急诊")

        # 会话日志加密存储（敏感健康数据不明文落盘）
        try:
            log_session(session_id, user_input, answer, result.get("user_symptoms", []))
        except Exception as e:
            print(f"[提示] 会话日志存储失败：{e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=uuid.uuid4().hex[:8], help="会话 ID（thread_id）")
    ap.add_argument("--build", action="store_true", help="构建知识库索引")
    ap.add_argument("--topics", type=int, default=0, help="构建索引时限制主题数量，0=全部")
    ap.add_argument("--rebuild", action="store_true", help="清空向量库重建")
    ap.add_argument("--eval", action="store_true", help="运行评估")
    ap.add_argument("--full", action="store_true", help="评估时追加全流程模型与知识覆盖测试")
    args = ap.parse_args()

    if args.build:
        from medical.rag.build_index import build

        build(max_topics=args.topics, force_rebuild=args.rebuild)
    elif args.eval:
        import evaluate

        evaluate.main(full=args.full)
    else:
        run_chat(args.session)


if __name__ == "__main__":
    main()
