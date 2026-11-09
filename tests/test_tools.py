from __future__ import annotations

import unittest

import tools
from tests.support import ROOT, fresh_dir


class ToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = fresh_dir("tools")
        tools.configure_workspace(self.ws)

    def tearDown(self) -> None:
        tools.configure_workspace(ROOT / "workspace")

    def test_calculator(self) -> None:
        result = tools.execute_tool("calculator", {"expression": "2**10"})
        self.assertTrue(result.ok)
        self.assertEqual(result.output, "1024")

    def test_write_read_edit_search(self) -> None:
        write = tools.execute_tool(
            "write_file", {"path": "a/b.txt", "content": "hello\nworld\nworld\n"}
        )
        self.assertTrue(write.ok, write.error)
        self.assertTrue((self.ws / "a" / "b.txt").is_file())

        read = tools.execute_tool("read_file", {"path": "a/b.txt"})
        self.assertIn("hello", read.output)

        multiple = tools.execute_tool(
            "edit_file", {"path": "a/b.txt", "old": "world", "new": "there"}
        )
        self.assertFalse(multiple.ok)
        self.assertIn("匹配到", multiple.error)

        edit = tools.execute_tool(
            "edit_file", {"path": "a/b.txt", "old": "hello", "new": "hi"}
        )
        self.assertTrue(edit.ok, edit.error)
        self.assertIn("hi", (self.ws / "a" / "b.txt").read_text(encoding="utf-8"))

        search = tools.execute_tool("search_files", {"pattern": "hi"})
        self.assertIn("a/b.txt", search.output)

    def test_todo(self) -> None:
        plan = tools.execute_tool("todo", {"op": "set", "items": ["一", "二"]})
        self.assertIn("[ ] 1.", plan.output)
        plan = tools.execute_tool("todo", {"op": "done", "index": 2})
        self.assertIn("[x] 2.", plan.output)
        plan = tools.execute_tool("todo", {"op": "list"})
        self.assertIn("二", plan.output)

    def test_path_escape(self) -> None:
        read = tools.execute_tool("read_file", {"path": "../secret.txt"})
        self.assertFalse(read.ok)
        self.assertIn("越界", read.error)

        write = tools.execute_tool(
            "write_file", {"path": "../evil.txt", "content": "x"}
        )
        self.assertFalse(write.ok)
        self.assertFalse((self.ws.parent / "evil.txt").exists())

    def test_arg_validation(self) -> None:
        bad = tools.execute_tool("calculator", {"expression": "1+1", "extra": 1})
        self.assertFalse(bad.ok)
        self.assertIn("未知参数", bad.error)

        missing = tools.execute_tool("read_file", {})
        self.assertFalse(missing.ok)
        self.assertIn("缺少必填参数", missing.error)

        wrong_type = tools.execute_tool("calculator", {"expression": 11})
        self.assertFalse(wrong_type.ok)
        self.assertIn("应为 string", wrong_type.error)

    def test_approval_for_overwrite(self) -> None:
        first = tools.execute_tool(
            "write_file", {"path": "note.txt", "content": "one"}
        )
        self.assertTrue(first.ok)

        denied = tools.execute_tool(
            "write_file", {"path": "note.txt", "content": "two"}
        )
        self.assertFalse(denied.ok)
        self.assertIn("人工确认", denied.error)

        allowed = tools.execute_tool(
            "write_file",
            {"path": "note.txt", "content": "two"},
            approver=lambda name, args: True,
        )
        self.assertTrue(allowed.ok)
        self.assertEqual((self.ws / "note.txt").read_text(encoding="utf-8"), "two")


if __name__ == "__main__":
    unittest.main()
