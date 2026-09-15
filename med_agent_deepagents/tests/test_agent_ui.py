"""保留原页面的知情确认、输入和会话重置。"""

import unittest
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver
from streamlit.testing.v1 import AppTest


class AgentUiContractTests(unittest.TestCase):
    def test_acknowledgement_emergency_and_reset(self):
        with patch("medical.graph.get_checkpointer", return_value=MemorySaver()):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(
                timeout=25
            )
            self.assertEqual(len(app.exception), 0)
            app.checkbox[0].set_value(True).run(timeout=25)
            next(button for button in app.button if button.label == "开始使用").click().run(
                timeout=25
            )
            self.assertEqual(len(app.chat_input), 1)
            app.chat_input[0].set_value("家人剧烈胸痛").run(timeout=25)
            self.assertTrue(
                any(
                    "120" in text for role, text in app.session_state["msgs"] if role == "assistant"
                )
            )
            old_thread = app.session_state["thread_id"]
            next(button for button in app.button if "开启新会话" in button.label).click().run(
                timeout=25
            )
            self.assertNotEqual(app.session_state["thread_id"], old_thread)
            self.assertEqual(app.session_state["msgs"], [])
            self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
