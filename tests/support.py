"""共享测试工具。"""

from __future__ import annotations

import dataclasses
import itertools
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402

TEST_ROOT = ROOT / ".test_tmp" / "unittest"
_CALL_IDS = itertools.count(1)


def fresh_dir(name: str) -> Path:
    path = TEST_ROOT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_config(tmp: Path, **overrides):
    cfg = dataclasses.replace(
        Config(),
        workspace=tmp / "workspace",
        history_dir=tmp / "history",
        trace_dir=tmp / "traces",
        persist_history=False,
        trace=False,
        summarize=False,
        auto_approve=True,
        temperature=0.0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.ensure_dirs()
    return cfg


class FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name: str, args: dict, call_id: str) -> None:
        self.id = call_id
        self.type = "function"
        self.function = FakeFunction(name, json.dumps(args, ensure_ascii=False))


class FakeMessage:
    def __init__(self, content: str | None, tool_calls: list[FakeToolCall] | None) -> None:
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none: bool = True) -> dict:
        data: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return data


class FakeClient:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls += 1
        if not self._responses:
            raise AssertionError("FakeClient 没有更多预设响应")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def response(
    content: str | None = None,
    calls: list[tuple[str, dict]] | None = None,
    prompt_tokens: int = 100,
) -> SimpleNamespace:
    tool_calls = None
    if calls:
        tool_calls = [
            FakeToolCall(name, args, f"call_{next(_CALL_IDS)}")
            for name, args in calls
        ]
    message = FakeMessage(content, tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=10,
            total_tokens=prompt_tokens + 10,
        ),
    )


def assert_tool_protocol(messages: list[dict]) -> None:
    """每条 assistant 的 tool_calls 都必须有对应的 tool 结果。"""
    pending: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                pending.add(call["id"])
        elif role == "tool":
            pending.discard(message.get("tool_call_id"))
        elif role == "user":
            assert not pending, f"user 之前有未回应的 tool_calls: {pending}"
    assert not pending, f"结尾有未回应的 tool_calls: {pending}"
