"""Agent Chat UI 上传文件接收与 thread 隔离。"""

import base64
import binascii
import re
from pathlib import Path

from base import config as cfg
from medical.privacy import redact_sensitive_text

ALLOWED_UPLOAD_SUFFIXES = {".txt", ".md", ".csv"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _safe_segment(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", value).strip("._")
    return cleaned[:80] or fallback


def save_uploaded_files(files: list[dict], thread_id: str) -> tuple[list[str], list[str]]:
    """保存 UI 的 base64 文本文件，返回相对 user_upload 的路径和错误。"""
    if not files:
        return [], []

    upload_root = Path(cfg.USER_UPLOAD_PATH).resolve()
    safe_thread = _safe_segment(thread_id, "unscoped")
    thread_dir = (upload_root / safe_thread).resolve()
    if not thread_dir.is_relative_to(upload_root):
        raise ValueError("上传目录越界")
    thread_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    errors: list[str] = []
    total_size = 0
    for index, item in enumerate(files):
        metadata = item.get("metadata") if isinstance(item, dict) else None
        raw_name = metadata.get("filename", "") if isinstance(metadata, dict) else ""
        original_name = _safe_segment(Path(str(raw_name)).name, f"upload_{index + 1}.txt")
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            errors.append(f"文件 {index + 1}：仅支持 txt/md/csv 文本文件")
            continue
        # 文件名常含患者姓名、日期或病案号。落盘时一律改为会话内通用名。
        filename = f"document_{index + 1}{suffix}"

        encoded = item.get("data", "") if isinstance(item, dict) else ""
        if not isinstance(encoded, str):
            errors.append(f"文件 {index + 1}：文件数据格式错误")
            continue
        if encoded.startswith("data:"):
            _, separator, encoded = encoded.partition(",")
            if not separator:
                errors.append(f"文件 {index + 1}：Data URL 格式错误")
                continue
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            errors.append(f"文件 {index + 1}：Base64 解码失败")
            continue

        total_size += len(payload)
        if len(payload) > MAX_UPLOAD_BYTES or total_size > MAX_UPLOAD_BYTES:
            errors.append(f"文件 {index + 1}：上传文件总量不能超过 10 MB")
            continue

        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            errors.append(f"文件 {index + 1}：仅支持 UTF-8 文本")
            continue

        target = (thread_dir / filename).resolve()
        if not target.is_relative_to(thread_dir):
            errors.append(f"文件 {index + 1}：文件路径非法")
            continue
        target.write_text(redact_sensitive_text(text), encoding="utf-8")
        saved.append(target.relative_to(upload_root).as_posix())
    return saved, errors
