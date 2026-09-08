from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from base import config as cfg


def _require_model_config(model_name: str | None) -> None:
    missing = []
    if not cfg.MODEL_API_BASE_URL:
        missing.append("MODEL_API_BASE_URL")
    if not cfg._api_key:
        missing.append("OPENAI_API_KEY")
    if not model_name:
        missing.append("BASE_LLM/SMALL_LLM")
    if missing:
        raise RuntimeError(
            f"模型配置缺失：{', '.join(missing)}。请复制 .env.example 为 .env 后填写。"
        )


def get_llm():
    """主对话模型（Function-call 能力）"""
    _require_model_config(cfg.BASE_LLM)
    return ChatOpenAI(
        base_url=cfg.MODEL_API_BASE_URL,
        model=cfg.BASE_LLM,
        api_key=cfg._api_key,
        temperature=0.3,
    )


def get_small_llm():
    """轻量模型：症状抽取、结构化输出等辅助任务"""
    _require_model_config(cfg.SMALL_LLM)
    return ChatOpenAI(
        base_url=cfg.MODEL_API_BASE_URL,
        model=cfg.SMALL_LLM,
        api_key=cfg._api_key,
        temperature=0,
    )


def get_embeddings():
    """BGE-M3 嵌入模型（SiliconFlow 提供，OpenAI 兼容协议）

    check_embedding_ctx_length=False：非 OpenAI 官方嵌入模型必须关闭
    tiktoken 预切分，否则嵌入调用会异常。
    """
    _require_model_config(cfg.EMBEDDING_MODEL)
    return OpenAIEmbeddings(
        base_url=cfg.MODEL_API_BASE_URL,
        model=cfg.EMBEDDING_MODEL,
        api_key=cfg._api_key,
        check_embedding_ctx_length=False,
        chunk_size=cfg.EMBEDDING_BATCH_SIZE,
    )
