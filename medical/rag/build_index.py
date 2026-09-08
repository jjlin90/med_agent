"""知识库索引构建：默沙东 HTML -> 知识切片 -> Chroma 向量库 + BM25 语料

用法：
    python main.py --build                # 全量构建
    python main.py --build --topics 300   # 仅前 300 个主题（快速调试）
    python main.py --build --rebuild      # 重建（清空已有向量库）
"""

import json
import shutil
from pathlib import Path

import jieba
from langchain_chroma import Chroma
from langchain_core.documents import Document

from base import config as cfg
from conn.llm import get_embeddings
from medical.rag.html_parser import iter_all_chunks

COLLECTION_NAME = "msd_medical_home"

# 写路径白名单：仅允许项目根目录下的 vectorstore/ 与 data/
_PROJECT_ROOT = Path(cfg.ROOT_PATH).resolve()
_ALLOWED_DIRS = (_PROJECT_ROOT / "vectorstore", _PROJECT_ROOT / "data")


def _safe_output_path(path: str) -> Path:
    """输出路径必须位于项目根目录白名单内（防路径穿越），越界直接拒绝"""
    resolved = Path(path).resolve()
    if not any(resolved == d or resolved.is_relative_to(d) for d in _ALLOWED_DIRS):
        raise ValueError(f"输出路径越界，仅允许 {[_ALLOWED_DIRS]}: {path}")
    return resolved


def _tokenize(text: str) -> list[str]:
    """中文分词：BM25 关键词召回依赖医学专有名词切分"""
    return [t for t in jieba.cut_for_search(text) if t.strip()]


def _load_existing_corpus(path: Path) -> list[dict]:
    """读取旧 BM25 语料；损坏或非预期格式时拒绝静默覆盖。"""
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"BM25 索引无法读取，请使用 --rebuild 重建：{path}") from exc
    corpus = payload.get("corpus")
    if not isinstance(corpus, list):
        raise RuntimeError(f"BM25 索引格式无效，请使用 --rebuild 重建：{path}")
    return corpus


def build(max_topics: int = 0, force_rebuild: bool = False):
    # 0. 输出路径校验，限制在项目根目录白名单内
    vs_path = _safe_output_path(cfg.VECTORSTORE_PATH)
    bm25_file = _safe_output_path(cfg.BM25_PATH)

    # 1. 解析全部知识切片
    print("开始解析默沙东诊疗手册（大众版）...")
    docs = iter_all_chunks(cfg.MSD_DATA_PATH, max_topics=max_topics)
    print(f"解析完成，共 {len(docs)} 个知识切片")
    if not docs:
        raise RuntimeError("未解析到知识切片，已中止构建以保护现有索引。")

    # 2. 重建则清空向量库
    if force_rebuild and vs_path.exists():
        shutil.rmtree(vs_path)

    # 3. 同步 Chroma：新增、更新内容变化的切片，并删除已处理主题的过期切片。
    vs_path.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(vs_path),
    )
    collection = store._collection

    current_ids = {doc["id"] for doc in docs}
    current_files = {doc["metadata"]["file"] for doc in docs}
    all_existing = collection.get(include=["metadatas"])
    stale_ids = [
        doc_id
        for doc_id, metadata in zip(all_existing["ids"], all_existing["metadatas"], strict=True)
        if metadata.get("file") in current_files and doc_id not in current_ids
    ]
    if not max_topics:
        stale_ids.extend(doc_id for doc_id in all_existing["ids"] if doc_id not in current_ids)
    stale_ids = sorted(set(stale_ids))
    if stale_ids:
        collection.delete(ids=stale_ids)

    batch_size = max(cfg.EMBEDDING_BATCH_SIZE * 4, 64)
    total_new = 0
    total_updated = 0
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        ids = [d["id"] for d in batch]
        existing_payload = collection.get(ids=ids, include=["documents", "metadatas"])
        existing = {
            doc_id: (text, metadata)
            for doc_id, text, metadata in zip(
                existing_payload["ids"],
                existing_payload["documents"],
                existing_payload["metadatas"],
                strict=True,
            )
        }
        new_docs: list[Document] = []
        new_ids: list[str] = []
        updated_docs: list[Document] = []
        updated_ids: list[str] = []
        for doc, doc_id in zip(batch, ids, strict=True):
            document = Document(page_content=doc["text"], metadata=doc["metadata"])
            if doc_id not in existing:
                new_docs.append(document)
                new_ids.append(doc_id)
            elif existing[doc_id] != (doc["text"], doc["metadata"]):
                updated_docs.append(document)
                updated_ids.append(doc_id)
        if new_docs:
            store.add_documents(new_docs, ids=new_ids)
            total_new += len(new_docs)
        if updated_docs:
            store.update_documents(ids=updated_ids, documents=updated_docs)
            total_updated += len(updated_docs)
        print(
            f"  向量化进度 {min(start + batch_size, len(docs))}/{len(docs)}"
            f"（累计新增 {total_new}，更新 {total_updated}）"
        )
    print(f"向量库已删除 {len(stale_ids)} 条过期切片")
    print(f"向量库完成：{collection.count()} 条切片位于 {vs_path}")

    # 4. 构建 BM25 语料。小批次构建会合并旧语料，不再覆盖未处理主题。
    corpus = [{"id": d["id"], "text": d["text"], "metadata": d["metadata"]} for d in docs]
    if max_topics and not force_rebuild:
        retained = [
            doc
            for doc in _load_existing_corpus(bm25_file)
            if doc.get("metadata", {}).get("file") not in current_files
        ]
        corpus = retained + corpus
    corpus.sort(key=lambda doc: doc["id"])
    bm25_data = {"corpus": corpus, "tokenized": [_tokenize(d["text"]) for d in corpus]}
    bm25_file.parent.mkdir(parents=True, exist_ok=True)
    bm25_file.write_text(json.dumps(bm25_data, ensure_ascii=False), encoding="utf-8")
    print(f"BM25 索引完成：{len(corpus)} 条语料位于 {bm25_file}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=int, default=0, help="限制主题页数量，0=全部")
    ap.add_argument("--rebuild", action="store_true", help="清空向量库重建")
    args = ap.parse_args()
    build(max_topics=args.topics, force_rebuild=args.rebuild)
