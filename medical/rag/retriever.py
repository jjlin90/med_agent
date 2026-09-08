"""医学知识库混合检索器：向量召回 + BM25 关键词召回 + RRF 融合 + Reranker 重排

医学专有名词对关键词召回敏感，纯向量召回容易漏掉精确术语，
因此采用混合检索；重排序进一步过滤无关文档，降低幻觉。
"""

import json
import logging

import requests
from langchain_chroma import Chroma

from base import config as cfg
from conn.llm import get_embeddings
from medical.rag.build_index import COLLECTION_NAME, _tokenize

_RRF_K = 60  # Reciprocal Rank Fusion 常数
logger = logging.getLogger(__name__)


class MedicalRetriever:
    def __init__(self):
        self._store = None
        self._bm25 = None
        self._corpus = None

    # ---------- 惰性加载 ----------
    def _load_store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=get_embeddings(),
                persist_directory=cfg.VECTORSTORE_PATH,
            )
        return self._store

    def _load_bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            with open(cfg.BM25_PATH, encoding="utf-8") as f:
                data = json.load(f)
            self._corpus = data["corpus"]
            self._bm25 = BM25Okapi(data["tokenized"])
        return self._bm25

    def ready(self) -> bool:
        import os

        return os.path.isfile(cfg.BM25_PATH) and os.path.isdir(cfg.VECTORSTORE_PATH)

    # ---------- 各路召回 ----------
    def _vector_search(self, query: str, k: int) -> list[str]:
        docs = self._load_store().similarity_search(query, k=k)
        return [d.page_content for d in docs]

    def _bm25_search(self, query: str, k: int) -> list[str]:
        bm25 = self._load_bm25()
        scores = bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._corpus[i]["text"] for i in ranked if scores[i] > 0]

    def _rerank(self, query: str, candidates: list[str], top_n: int) -> list[str]:
        """调用 SiliconFlow rerank 接口（BGE-reranker-v2-m3）二次精排"""
        try:
            resp = requests.post(
                cfg.MODEL_API_BASE_URL.rstrip("/") + "/rerank",
                headers={"Authorization": f"Bearer {cfg._api_key}"},
                json={
                    "model": cfg.RERANK_MODEL,
                    "query": query,
                    "documents": candidates,
                    "top_n": top_n,
                },
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json()["results"]
            return [
                candidates[r["index"]] for r in sorted(results, key=lambda x: -x["relevance_score"])
            ]
        except Exception as e:
            logger.warning("rerank 失败，已降级为 RRF 融合：%s", e)
            return candidates[:top_n]

    # ---------- 对外入口 ----------
    def search(self, query: str, top_k: int = 0) -> list[dict]:
        """混合检索，返回 [{text, metadata}]"""
        top_k = max(1, min(top_k or cfg.FINAL_TOP_K, 10))
        vec_hits = self._vector_search(query, cfg.VECTOR_TOP_K)
        kw_hits = self._bm25_search(query, cfg.BM25_TOP_K)

        # RRF 融合两路排名
        fused: dict[str, float] = {}
        for rank, text in enumerate(vec_hits):
            fused[text] = fused.get(text, 0) + 1.0 / (_RRF_K + rank + 1)
        for rank, text in enumerate(kw_hits):
            fused[text] = fused.get(text, 0) + 1.0 / (_RRF_K + rank + 1)
        candidates = [t for t, _ in sorted(fused.items(), key=lambda x: -x[1])]

        # Rerank 精排（不可用时降级为 RRF 顺序）
        if cfg.ENABLE_RERANK and len(candidates) > top_k:
            rerank_n = min(top_k, cfg.RERANK_TOP_K)
            candidates = self._rerank(query, candidates[: max(rerank_n * 2, 8)], rerank_n)
        candidates = candidates[:top_k]

        # 从 BM25 语料取回元数据（向量库与 BM25 同源）
        results, meta_by_text = [], {c["text"]: c["metadata"] for c in self._corpus or []}
        if not meta_by_text:
            self._load_bm25()
            meta_by_text = {c["text"]: c["metadata"] for c in self._corpus}
        for text in candidates:
            results.append({"text": text, "metadata": meta_by_text.get(text, {})})
        return results


# 模块级单例
_retriever: MedicalRetriever | None = None


def get_retriever() -> MedicalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MedicalRetriever()
    return _retriever


def format_results(results: list[dict]) -> str:
    """把检索结果格式化为工具输出，附来源信息供模型引用"""
    if not results:
        return "知识库中未检索到相关权威资料。请如实告知用户暂无相关资料，禁止编造医学知识。"
    lines = []
    for i, r in enumerate(results, 1):
        m = r.get("metadata", {})
        loc = m.get("source", "")
        if m.get("chapter"):
            loc += f"（章节：{m['chapter']}）"
        lines.append(f"[{i}] {r['text']}\n（来源：{loc}）")
    return "\n\n".join(lines)
