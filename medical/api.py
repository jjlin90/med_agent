"""挂载到 LangGraph Agent Server 的轻量 HTTP 扩展。"""

from fastapi import FastAPI

from medical.compliance import DISCLAIMER

app = FastAPI(title="医疗健康科普 Agent 扩展接口", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "graph_id": "agent"}


@app.get("/medical/capabilities")
def capabilities() -> dict:
    return {
        "can": ["健康科普", "症状整理", "就诊科室建议", "医疗文本整理"],
        "cannot": ["疾病诊断", "开具处方", "个体化用药指导"],
        "disclaimer": DISCLAIMER,
    }
