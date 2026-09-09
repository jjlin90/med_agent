import os
from pathlib import Path

from dotenv import load_dotenv

# ==================== 路径配置 ====================
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(Path(ROOT_PATH) / ".env")

# 默沙东诊疗手册（大众版）离线数据根目录
MSD_DATA_PATH = os.getenv("MSD_DATA_PATH", os.path.join(ROOT_PATH, "data", "msd"))

# 用户上传检验报告的目录（read_medical_doc 工具只允许读这里）
USER_UPLOAD_PATH = os.path.join(ROOT_PATH, "user_upload")


def _resolve_in_root(rel_path: str, default: str) -> str:
    """把可配置路径规范化并限制在项目根目录内，防止路径穿越"""
    root = Path(ROOT_PATH).resolve()
    candidate = (root / rel_path).resolve()
    if not candidate.is_relative_to(root):
        candidate = (root / default).resolve()
    return str(candidate)


def ensure_in_root(path: str) -> str:
    """校验路径位于项目根目录内（防路径穿越），越界抛异常。写文件前必须调用"""
    root = Path(ROOT_PATH).resolve()
    candidate = Path(path).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError(f"路径越界，仅允许写项目目录内: {path}")
    return str(candidate)


# 向量库 / BM25 / checkpoint / 加密日志 路径（均限制在项目根目录内）
VECTORSTORE_PATH = _resolve_in_root(
    os.getenv("VECTORSTORE_PATH", "vectorstore/chroma"), "vectorstore/chroma"
)
BM25_PATH = _resolve_in_root(
    os.getenv("BM25_PATH", "vectorstore/bm25.json"), "vectorstore/bm25.json"
)
CHECKPOINT_PATH = _resolve_in_root(
    os.getenv("CHECKPOINT_PATH", "data/checkpoints.sqlite"), "data/checkpoints.sqlite"
)
SECURE_LOG_PATH = _resolve_in_root(
    os.getenv("SECURE_LOG_PATH", "data/sessions.enc"), "data/sessions.enc"
)
ENABLE_SECURE_SESSION_LOG = os.getenv("ENABLE_SECURE_SESSION_LOG", "false").lower() == "true"

# 普通运行日志：仅记录运行状态和异常，不记录原始用户输入。
ENABLE_FILE_LOG = os.getenv("ENABLE_FILE_LOG", "true").lower() == "true"
LOG_PATH = _resolve_in_root(os.getenv("LOG_PATH", "logs/med_agent.log"), "logs/med_agent.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_MAX_BYTES = max(1024, int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024))))
LOG_BACKUP_COUNT = max(1, int(os.getenv("LOG_BACKUP_COUNT", "5")))

# ==================== 模型配置 ====================
_api_key = os.getenv("OPENAI_API_KEY")  # API Key（内部使用）
MODEL_API_BASE_URL = os.getenv("MODEL_API_BASE_URL")  # OpenAI 兼容接口地址
BASE_LLM = os.getenv("BASE_LLM")  # 主模型
SMALL_LLM = os.getenv("SMALL_LLM", os.getenv("BASE_LLM"))  # 轻量模型（症状抽取等）

# 嵌入模型：BGE-M3；当前项目使用其稠密向量能力，尚未做通用 embedding 对照实验
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# 重排序：SiliconFlow rerank 接口，二次过滤降低幻觉
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", os.getenv("ENABLE_RERANK_TOP_K", "5")))

# ==================== RAG 配置 ====================
# 医学文档切片：按章节切分，目标块大小 400-1024 字符
CHUNK_MIN_SIZE = 400
CHUNK_MAX_SIZE = 1024
CHUNK_OVERLAP = 80
VECTOR_TOP_K = 8  # 向量召回
BM25_TOP_K = 8  # 关键词召回
FINAL_TOP_K = 5  # 融合/重排后最终条数

# Agent 复杂任务最多允许的工具执行轮数，防止模型反复检索或形成死循环
MAX_AGENT_TOOL_ROUNDS = int(os.getenv("MAX_AGENT_TOOL_ROUNDS", "6"))
