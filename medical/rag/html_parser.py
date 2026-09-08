"""默沙东诊疗手册（大众版）离线 HTML 解析器

数据结构：
- 根目录 {GUID}.html 为主题详情页（正文容器 class 含 Topic_topic__）
- h1 = 主题名，h2/h3 = 章节/小节层级
- Json/allchapterstopics.json 提供 GUID -> (主题名, 所属章节) 映射

切片原则：按章节/小节切分，块大小控制在 512-1024 字符（见 base/config.py），
每个切片正文前拼接主题上下文标题，提升向量召回质量。
"""

import json
import os
import re

from bs4 import BeautifulSoup

from base import config as cfg

# 除主题页外一并收录的通用科普页（检查项目、医学术语、健康生活）
GENERAL_PAGES = {
    "commonmedicaltests.html": "常见医学检查科普",
    "bloodtest.html": "血液检查科普",
    "medicalterms.html": "医学术语表",
    "Healthyliving.html": "健康生活方式",
}

_SENT_SPLIT = re.compile(r"(?<=[。！？；])")


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def load_topic_meta(data_root: str) -> dict[str, dict]:
    """读取 allchapterstopics.json，返回 {GUID: {name, chapter}}"""
    path = os.path.join(data_root, "Json", "allchapterstopics.json")
    meta = {}
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    for item in data:
        if item.get("MetaInfo") == "Topics" and item.get("Id"):
            meta[item["Id"]] = {
                "name": item.get("Name", ""),
                "chapter": item.get("SectionName", ""),
            }
    return meta


def _iter_text_blocks(container):
    """按文档顺序遍历标题与段落，产出 (类型, 文本)；过滤连续重复（离线页面常见双份渲染）"""
    prev = None
    for el in container.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = _clean_text(el.get_text(" ", strip=True))
        if not text or len(text) < 2:
            continue
        if text == prev:  # 连续重复项去重
            continue
        prev = text
        # 导航/按钮噪声过滤
        if text in ("展开全部", "收起全部", "打印", "分享", "下载", "听此页"):
            continue
        yield el.name, text


def _make_chunks(title: str, chapter: str, blocks, source: str) -> list[dict]:
    """把 (h2,h3,p) 序列按小节聚合，再按目标块大小切片"""
    # 1. 聚合到小节
    sections = []  # [[h2, h3, [paras]]]
    h2, h3 = "", ""
    for tag, text in blocks:
        if tag == "h2":
            h2, h3 = text, ""
            sections.append([h2, h3, []])
        elif tag == "h3":
            h3 = text
            sections.append([h2, h3, []])
        else:  # p / li / h4
            if not sections:
                sections.append(["", "", []])
            sections[-1][2].append(text)

    # 2. 小节 -> (小节标题, 正文) 切片，块大小控制在 400-1024
    pieces: list[list[str]] = []  # [[header, body, sec2, sec3]]
    for sec2, sec3, paras in sections:
        body = "".join(paras)
        if len(body) < 30:  # 过滤导航、目录等无效内容
            continue
        header = f"《{title}》" + (sec2 or "") + (f"·{sec3}" if sec3 else "")

        if len(body) <= cfg.CHUNK_MAX_SIZE:
            # 过短切片并入同小节的上一片，避免碎片
            if (
                len(body) < cfg.CHUNK_MIN_SIZE
                and pieces
                and pieces[-1][0] == header
                and len(pieces[-1][1]) + len(body) + 1 <= cfg.CHUNK_MAX_SIZE
            ):
                pieces[-1][1] = pieces[-1][1] + "\n" + body
            else:
                pieces.append([header, body, sec2, sec3])
        else:
            # 超长小节按句子切分为多片，带少量重叠
            sents = _SENT_SPLIT.split(body)
            cur = ""
            for s in sents:
                if cur and len(cur) + len(s) > cfg.CHUNK_MAX_SIZE:
                    pieces.append([header, cur, sec2, sec3])
                    cur = cur[-cfg.CHUNK_OVERLAP :] + s
                else:
                    cur += s
            if cur.strip():
                pieces.append([header, cur, sec2, sec3])

    # 3. 生成带元数据的文档块
    docs = []
    for i, (header, body, sec2, sec3) in enumerate(pieces):
        text = header + "：" + body
        docs.append(
            {
                "id": f"{os.path.splitext(source)[0]}_{i}",
                "text": text,
                "metadata": {
                    "title": title,
                    "chapter": chapter,
                    "section": sec2,
                    "subsection": sec3,
                    "source": f"默沙东诊疗手册（大众版）·{title}",
                    "file": source,
                },
            }
        )
    return docs


def parse_topic_page(path: str) -> list[dict]:
    """解析单个主题 HTML 为知识切片列表"""
    source = os.path.basename(path)
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    # 定位正文容器（class 名带构建哈希，用前缀匹配）
    container = soup.find("div", class_=re.compile(r"Topic_topic__"))
    if container is None:
        main = soup.find("div", class_=re.compile(r"mainContainer"))
        container = main if main is not None else soup.body
    if container is None:
        return []

    # 剔除脚本样式与页脚导航
    for tag in container.find_all(["script", "style", "footer", "nav"]):
        tag.decompose()

    title = (
        _clean_text(container.h1.get_text())
        if container.h1
        else _clean_text(soup.title.get_text())
        if soup.title
        else source
    )
    blocks = list(_iter_text_blocks(container))
    return _make_chunks(title, "", blocks, source)


def parse_general_page(path: str, chapter: str) -> list[dict]:
    """解析通用科普页（检查项目/术语表等）为知识切片列表"""
    source = os.path.basename(path)
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    for tag in soup.find_all(["script", "style", "footer", "nav", "header"]):
        tag.decompose()
    container = soup.body if soup.body else soup

    title = _clean_text(soup.title.get_text()) if soup.title else chapter
    blocks = list(_iter_text_blocks(container))
    return _make_chunks(title, chapter, blocks, source)


def iter_all_chunks(data_root: str, max_topics: int = 0) -> list[dict]:
    """遍历全部主题页与通用页，产出全部知识切片

    max_topics: 调试用限制主题数量，0 表示全部
    """
    meta = load_topic_meta(data_root)
    all_docs: list[dict] = []

    # 主题页
    topic_files = sorted(
        os.path.join(data_root, gid + ".html")
        for gid in meta
        if os.path.exists(os.path.join(data_root, gid + ".html"))
    )
    if max_topics:
        topic_files = topic_files[:max_topics]

    for i, path in enumerate(topic_files):
        gid = os.path.splitext(os.path.basename(path))[0]
        m = meta.get(gid, {})
        docs = parse_topic_page(path)
        for d in docs:
            d["metadata"]["title"] = m.get("name") or d["metadata"]["title"]
            d["metadata"]["chapter"] = m.get("chapter", "")
            d["metadata"]["source"] = f"默沙东诊疗手册（大众版）·{d['metadata']['title']}"
        all_docs.extend(docs)
        if (i + 1) % 200 == 0:
            print(f"  已解析 {i + 1}/{len(topic_files)} 个主题页，累计切片 {len(all_docs)}")

    # 通用科普页
    for fname, chapter in GENERAL_PAGES.items():
        path = os.path.join(data_root, fname)
        if os.path.exists(path):
            docs = parse_general_page(path, chapter)
            for d in docs:
                d["metadata"]["source"] = f"默沙东诊疗手册（大众版）·{chapter}"
            all_docs.extend(docs)
            print(f"  通用页 {fname} 解析切片 {len(docs)}")

    return all_docs
