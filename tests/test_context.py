from __future__ import annotations

import unittest

from tests.support import fresh_dir
from context import ContextManager, estimate_messages


class ContextTests(unittest.TestCase):
    def _messages(self, rounds: int = 20) -> list[dict]:
        messages = [{"role": "system", "content": "系统提示"}]
        for i in range(rounds):
            call_id = f"call_{i}"
            messages.append(
                {"role": "user", "content": f"第{i}个问题：" + "很长的内容 " * 20}
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "a.txt"}',
                            },
                        }
                    ],
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": "结果 " * 30}
            )
        return messages

    def test_budget_and_pairing(self) -> None:
        messages = self._messages()
        manager = ContextManager(
            budget=700, summarizer=lambda chunk, prev: "摘要" + str(len(chunk))
        )
        visible = manager.build(messages)
        self.assertLessEqual(estimate_messages(visible), 700)
        self.assertEqual(visible[0]["content"], "系统提示")
        self.assertTrue(
            any("早期对话摘要" in str(m.get("content")) for m in visible)
        )
        pending: set[str] = set()
        for message in visible:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    pending.add(call["id"])
            elif message.get("role") == "tool":
                self.assertIn(message["tool_call_id"], pending)
        self.assertEqual(manager._cut % 3, 0)

    def test_no_compression_when_fits(self) -> None:
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
        manager = ContextManager(budget=1000)
        self.assertIs(manager.build(messages), messages)

    def test_mechanical_fallback_when_summarizer_empty(self) -> None:
        messages = self._messages(rounds=10)
        manager = ContextManager(budget=400, summarizer=lambda chunk, prev: "")
        visible = manager.build(messages)
        self.assertLessEqual(estimate_messages(visible), 400)
        self.assertTrue(manager.summary_text)

    def test_reset(self) -> None:
        manager = ContextManager(budget=10, summarizer=lambda chunk, prev: "x")
        manager.build(self._messages(rounds=3))
        manager.reset()
        self.assertEqual(manager.summary_text, "")
        self.assertEqual(manager._cut, 0)


if __name__ == "__main__":
    unittest.main()
