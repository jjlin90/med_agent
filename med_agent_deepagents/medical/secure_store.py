"""用户健康数据加密存储

用户输入的症状、病史属于敏感个人健康数据：
- 会话日志（症状主诉/回答摘要）使用 Fernet 对称加密后落盘，禁止明文存储；
- 加密密钥保存在项目 data 目录下的 .fernet_key（生产环境应接入 KMS）；
- 禁止把用户真实病历用于模型训练（本项目不做任何训练数据落盘）。
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from cryptography.fernet import Fernet

from base import config as cfg
from medical.privacy import pseudonymize_identifier, redact_sensitive_text

_KEY_FILE = Path(cfg.ROOT_PATH) / "data" / ".fernet_key"


def _get_fernet() -> Fernet:
    _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not _KEY_FILE.exists():
        _KEY_FILE.write_bytes(Fernet.generate_key())
    return Fernet(_KEY_FILE.read_bytes())


def log_session(
    thread_id: str,
    user_text: str,
    assistant_text: str,
    symptoms: list[str],
) -> bool:
    """按显式配置记录最小化、脱敏且加密的审计日志。

    默认不落盘。返回值表示本次是否真的写入日志。
    """
    if not cfg.ENABLE_SECURE_SESSION_LOG:
        return False

    f = _get_fernet()
    secret = _KEY_FILE.read_bytes()
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "thread_id": pseudonymize_identifier(thread_id, secret),
        "user_text": redact_sensitive_text(user_text)[:2000],
        "assistant_text": redact_sensitive_text(assistant_text)[:2000],
        "symptom_count": len(symptoms or []),
    }
    encrypted = f.encrypt(json.dumps(record, ensure_ascii=False).encode("utf-8"))
    log_path = cfg.ensure_in_root(cfg.SECURE_LOG_PATH)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as fp:
        fp.write(encrypted + b"\n")
    return True


def read_sessions() -> list[dict]:
    """解密读取全部会话日志（仅供运维审计使用）"""
    f = _get_fernet()
    log_path = cfg.ensure_in_root(cfg.SECURE_LOG_PATH)
    if not Path(log_path).exists():
        return []
    records = []
    for line in Path(log_path).read_bytes().splitlines():
        if not line.strip():
            continue
        records.append(json.loads(f.decrypt(line).decode("utf-8")))
    return records
