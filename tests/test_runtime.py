from __future__ import annotations

import unittest

from history import HistoryStore
from runtime import AgentRuntime
from tests.support import (
    FakeClient,
    assert_tool_protocol,
    fresh_dir,
    make_config,
    response,
)


class RuntimeTests(unittest.TestCase):
    def run_agent(self, responses, **overrides):
        cfg = make_config(fresh_dir("runtime"), **overrides)
        client = FakeClient(responses)
        runtime = AgentRuntime(cfg, client=client)
        runtime.start()
        return runtime, client

    def test_final_answer(self) -> None:
        runtime, client = self.run_agent([response(content="完成")])
        result = runtime.run_turn("你好")
        self.assertEqual(result.text, "完成")
        self.assertEqual(result.stop_reason, "final")
        self.assertEqual(result.steps, 1)
        self.assertEqual(client.calls, 1)
        assert_tool_protocol(runtime.messages)
        runtime.close()

    def test_tool_then_final(self) -> None:
        runtime, _ = self.run_agent(
            [
                response(calls=[("calculator", {"expression": "1+1"})]),
                response(content="2"),
            ]
        )
        result = runtime.run_turn("1+1")
        self.assertEqual(result.stop_reason, "final")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.records[0].output, "2")
        self.assertEqual(runtime.messages[-1]["content"], "2")
        assert_tool_protocol(runtime.messages)
        runtime.close()

    def test_repeated_call_stops_and_keeps_protocol(self) -> None:
        duplicate = response(calls=[("calculator", {"expression": "1+1"})])
        runtime, _ = self.run_agent([duplicate, duplicate])
        result = runtime.run_turn("重复算")
        self.assertEqual(result.stop_reason, "repeated_call")
        self.assertEqual(result.steps, 2)
        self.assertTrue(result.records[0].ok)
        self.assertFalse(result.records[1].ok)
        self.assertIn("未执行", result.records[1].error)
        assert_tool_protocol(runtime.messages)
        runtime.close()

    def test_max_steps(self) -> None:
        runtime, _ = self.run_agent(
            [
                response(calls=[("calculator", {"expression": "1+1"})]),
                response(calls=[("calculator", {"expression": "2+2"})]),
            ],
            max_steps=2,
        )
        result = runtime.run_turn("一直算")
        self.assertEqual(result.stop_reason, "max_steps")
        self.assertEqual(result.steps, 2)
        assert_tool_protocol(runtime.messages)
        runtime.close()

    def test_different_args_are_not_repeats(self) -> None:
        runtime, _ = self.run_agent(
            [
                response(calls=[("calculator", {"expression": "1+1"})]),
                response(calls=[("calculator", {"expression": "2+2"})]),
                response(content="done"),
            ]
        )
        result = runtime.run_turn("两次不同计算")
        self.assertEqual(result.stop_reason, "final")
        self.assertEqual(result.steps, 3)
        self.assertTrue(all(record.ok for record in result.records))
        runtime.close()

    def test_empty_answer_retry(self) -> None:
        runtime, _ = self.run_agent([response(content=""), response(content="好的")])
        result = runtime.run_turn("说话")
        self.assertEqual(result.stop_reason, "final")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.text, "好的")
        runtime.close()

    def test_claim_mismatch_retry(self) -> None:
        runtime, _ = self.run_agent(
            [
                response(content="已经创建 ghost.txt。"),
                response(content="抱歉，我没有创建。"),
            ]
        )
        result = runtime.run_turn("建文件")
        self.assertEqual(result.stop_reason, "final")
        self.assertEqual(result.steps, 2)
        self.assertTrue(result.warnings)
        self.assertIn("核验", runtime.messages[3]["content"])
        runtime.close()

    def test_context_limit_stops(self) -> None:
        runtime, _ = self.run_agent(
            [response(content="hi", prompt_tokens=9000)],
            hard_context_tokens=8192,
        )
        result = runtime.run_turn("长上下文")
        self.assertEqual(result.stop_reason, "context_limit")
        self.assertIn("上下文", result.text)
        runtime.close()

    def test_time_budget_stops_before_model_call(self) -> None:
        runtime, client = self.run_agent(
            [response(content="不应该被调用")], max_turn_seconds=-1
        )
        result = runtime.run_turn("超时")
        self.assertEqual(result.stop_reason, "time_budget")
        self.assertEqual(result.steps, 0)
        self.assertEqual(client.calls, 0)
        runtime.close()

    def test_history_resume_repairs_dangling_tool_calls(self) -> None:
        cfg = make_config(fresh_dir("runtime_repair"), persist_history=True)
        history = HistoryStore(cfg.history_dir)
        session_id = history.new_session({})
        history.append(session_id, {"role": "system", "content": "s"})
        history.append(session_id, {"role": "user", "content": "算一下"})
        history.append(
            session_id,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_broken",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression":"1+1"}',
                        },
                    }
                ],
            },
        )

        runtime = AgentRuntime(cfg, client=FakeClient([response(content="ok")]))
        runtime.start(session_id)
        self.assertEqual(runtime.messages[-1]["role"], "tool")
        self.assertEqual(runtime.messages[-1]["tool_call_id"], "call_broken")
        self.assertIn("中断", runtime.messages[-1]["content"])
        runtime.close()

    def test_history_resume(self) -> None:
        cfg = make_config(fresh_dir("runtime_history"), persist_history=True)
        first = AgentRuntime(cfg, client=FakeClient([response(content="第一轮完成")]))
        session_id = first.start()
        first.run_turn("记住暗号是 Falcon")
        first.close()

        second = AgentRuntime(cfg, client=FakeClient([response(content="第二轮完成")]))
        second.start(session_id)
        self.assertEqual(second.turn_count, 1)
        self.assertTrue(
            any("Falcon" in str(m.get("content")) for m in second.messages)
        )
        result = second.run_turn("继续")
        self.assertEqual(result.text, "第二轮完成")
        loaded = second.history.load(session_id)
        self.assertGreaterEqual(len(loaded), 5)
        second.close()


if __name__ == "__main__":
    unittest.main()
