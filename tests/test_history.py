from __future__ import annotations

import json
import unittest

from history import HistoryStore
from tests.support import fresh_dir


class HistoryTests(unittest.TestCase):
    def test_roundtrip_and_latest(self) -> None:
        store = HistoryStore(fresh_dir("history"))
        sid = store.new_session({"model": "m"})
        store.append(sid, {"role": "system", "content": "s"})
        store.append(sid, {"role": "user", "content": "u"})
        self.assertEqual([m["role"] for m in store.load(sid)], ["system", "user"])
        self.assertEqual(store.latest(), sid)
        self.assertEqual(store.meta(sid)["model"], "m")

    def test_corrupt_line_skipped(self) -> None:
        store = HistoryStore(fresh_dir("history_corrupt"))
        sid = store.new_session({})
        store.append(sid, {"role": "user", "content": "ok"})
        path = store.root / f"{sid}.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "{broken\n", encoding="utf-8")
        self.assertEqual(store.load(sid), [{"role": "user", "content": "ok"}])

    def test_archive(self) -> None:
        store = HistoryStore(fresh_dir("history_archive"))
        sid = store.new_session({})
        store.append(sid, {"role": "user", "content": "u"})
        target = store.archive(sid)
        self.assertIsNotNone(target)
        assert target is not None
        self.assertTrue(target.exists())
        self.assertFalse(store.exists(sid))

    def test_invalid_id_rejected(self) -> None:
        store = HistoryStore(fresh_dir("history_invalid"))
        with self.assertRaises(ValueError):
            store.load("../../etc/passwd")

    def test_message_json_shape(self) -> None:
        store = HistoryStore(fresh_dir("history_shape"))
        sid = store.new_session({})
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression":"1+1"}'},
                }
            ],
        }
        store.append(sid, message)
        loaded = store.load(sid)
        self.assertEqual(loaded[0]["tool_calls"][0]["function"]["name"], "calculator")
        self.assertEqual(json.loads(loaded[0]["tool_calls"][0]["function"]["arguments"]), {"expression": "1+1"})


if __name__ == "__main__":
    unittest.main()
