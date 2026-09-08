"""统一日志配置的离线测试。"""

import logging
import unittest
from pathlib import Path

from base import config as cfg
from base.logging_config import RedactingFormatter


class LoggingConfigTests(unittest.TestCase):
    def test_default_log_path_is_inside_project(self):
        root = Path(cfg.ROOT_PATH).resolve()
        self.assertTrue(Path(cfg.LOG_PATH).resolve().is_relative_to(root))

    def test_formatter_redacts_direct_identifiers(self):
        formatter = RedactingFormatter("%(message)s")
        mobile = "".join(("138", "0013", "8000"))
        email = "@".join(("demo", "example.com"))
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg=f"姓名：张三，手机号 {mobile}，邮箱 {email}",
            args=(),
            exc_info=None,
        )
        text = formatter.format(record)
        self.assertNotIn("张三", text)
        self.assertNotIn(mobile, text)
        self.assertNotIn(email, text)


if __name__ == "__main__":
    unittest.main()
