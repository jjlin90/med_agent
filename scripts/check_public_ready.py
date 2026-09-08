"""Fail when files intended for Git contain common secret or privacy leaks."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 2 * 1024 * 1024
SKIP_SCAN = {"scripts/check_public_ready.py"}
SENSITIVE_PATHS = (
    ".env",
    "data/.fernet_key",
    "data/checkpoints.sqlite",
    "data/sessions.enc",
    "vectorstore",
    "user_upload",
    "output",
    "logs",
)
ALLOWED_PLACEHOLDERS = {"user_upload/.gitkeep"}
RULES = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "API token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "email address": re.compile(r"(?<![\w.-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])"),
    "Chinese mobile number": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "Chinese ID number": re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
    "absolute Windows path": re.compile(r"\b[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]"),
}


def publishable_files() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item for item in result.stdout.decode("utf-8").split("\0") if item]


def main() -> int:
    failures: list[str] = []
    paths = publishable_files()
    relative_paths = {path.relative_to(ROOT).as_posix() for path in paths}

    for sensitive in SENSITIVE_PATHS:
        if any(
            (item == sensitive or item.startswith(f"{sensitive}/"))
            and item not in ALLOWED_PLACEHOLDERS
            for item in relative_paths
        ):
            failures.append(f"sensitive path is publishable: {sensitive}")

    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if relative in SKIP_SCAN or not path.is_file() or path.stat().st_size > MAX_TEXT_BYTES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in RULES.items():
            if pattern.search(content):
                failures.append(f"{label}: {relative}")

    if failures:
        print("Public-readiness check failed:")
        for failure in sorted(set(failures)):
            print(f"- {failure}")
        return 1

    print(f"Public-readiness check passed ({len(paths)} publishable files scanned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
