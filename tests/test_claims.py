from __future__ import annotations

import unittest
from dataclasses import dataclass

from claims import verify_claims
from tests.support import fresh_dir


@dataclass
class Rec:
    name: str
    args: dict
    ok: bool = True
    output: str = ""


class ClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = fresh_dir("claims")
        (self.ws / "result.txt").write_text("56088", encoding="utf-8")

    def test_write_claim_with_record(self) -> None:
        records = [Rec("write_file", {"path": "result.txt"}, output="已写入 result.txt")]
        self.assertEqual(
            verify_claims("已经写进 result.txt 了。", records, self.ws, 0), []
        )

    def test_write_claim_without_record(self) -> None:
        issues = verify_claims("已经写进 result.txt 了。", [], self.ws, 0)
        self.assertTrue(issues)

    def test_write_claim_missing_file(self) -> None:
        issues = verify_claims("已经创建 ghost.txt。", [], self.ws, 0)
        self.assertTrue(any("不存在" in issue for issue in issues))

    def test_failed_record_does_not_count(self) -> None:
        records = [
            Rec(
                "write_file",
                {"path": "result.txt"},
                ok=False,
                output="",
            )
        ]
        self.assertTrue(verify_claims("已经写进 result.txt。", records, self.ws, 0))

    def test_read_claim(self) -> None:
        records = [Rec("read_file", {"path": "result.txt"}, output="56088")]
        self.assertEqual(verify_claims("已读取 result.txt。", records, self.ws, 0), [])

    def test_read_claim_missing(self) -> None:
        self.assertTrue(verify_claims("已读取 ghost.txt。", [], self.ws, 0))

    def test_path_outside_workspace_ignored(self) -> None:
        self.assertEqual(
            verify_claims("已经写进 ../agent.py。", [], self.ws, 0), []
        )

    def test_no_claim_sentence(self) -> None:
        self.assertEqual(verify_claims("result.txt 里有 56088。", [], self.ws, 0), [])


if __name__ == "__main__":
    unittest.main()
