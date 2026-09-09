"""本地隐私保护工具。

这里只处理可稳定识别的直接标识符。医学内容本身仍属于敏感数据：普通运行日志默认开启，
但不得记录原始用户输入；会话审计日志默认关闭。上传目录和持久化文件必须保持在版本控制之外。
"""

from __future__ import annotations

import hashlib
import hmac
import re

_REDACTION_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<![\w.-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])"), "[已脱敏邮箱]"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[已脱敏手机号]"),
    (re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"), "[已脱敏身份证号]"),
    (
        re.compile(r"((?:患者)?姓名|住址|地址|病案号|住院号|门诊号)\s*[:：]\s*[^,，;；\n]{1,60}"),
        r"\1：[已脱敏]",
    ),
)


def redact_sensitive_text(value: str) -> str:
    """遮盖文本中的常见直接标识符，不修改普通医学指标与数值。"""
    redacted = str(value or "")
    for pattern, replacement in _REDACTION_RULES:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def pseudonymize_identifier(value: str, secret: bytes) -> str:
    """用本地密钥生成不可逆、稳定的短标识，避免日志保存原始 thread_id。"""
    digest = hmac.new(secret, str(value).encode("utf-8"), hashlib.sha256).hexdigest()
    return f"thread-{digest[:16]}"
