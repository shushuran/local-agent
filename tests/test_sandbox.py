from __future__ import annotations

import unittest

import tools
from tests.support import ROOT, fresh_dir


class SandboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = fresh_dir("sandbox")
        self.outside = self.ws.parent / "evil.txt"
        if self.outside.exists():
            self.outside.unlink()
        tools.configure_workspace(self.ws)

    def tearDown(self) -> None:
        tools.configure_workspace(ROOT / "workspace")

    def run_code(self, code: str):
        return tools.execute_tool("python_exec", {"code": code})

    def test_normal_code(self) -> None:
        result = self.run_code("print(sum(range(1, 11)))")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output, "55")

    def test_numpy_import(self) -> None:
        result = self.run_code("import numpy as np; print(int(np.arange(10).sum()))")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output, "45")

    def test_write_escape_blocked(self) -> None:
        result = self.run_code("open(r'..\\evil.txt', 'w').write('x')")
        self.assertFalse(result.ok)
        self.assertIn("沙箱", result.error)
        self.assertFalse(self.outside.exists())

    def test_network_blocked(self) -> None:
        result = self.run_code(
            "import socket; socket.create_connection(('example.com', 80), 2)"
        )
        self.assertFalse(result.ok)
        self.assertIn("沙箱", result.error)

    def test_subprocess_blocked(self) -> None:
        result = self.run_code(
            "import subprocess; subprocess.run(['cmd', '/c', 'dir'])"
        )
        self.assertFalse(result.ok)
        self.assertIn("沙箱", result.error)


if __name__ == "__main__":
    unittest.main()
