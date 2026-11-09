"""Agent 运行时：模型调用、工具循环、停止条件、失败处理和上下文压缩。

agent.py 只负责 CLI；所有可测试的逻辑都在这里。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import openai
from openai import OpenAI

from claims import verify_claims
from config import Config
from context import ContextManager
from history import HistoryStore
from tools import TOOLS, ToolResult, configure_workspace, execute_tool
from trace import Tracer

EventCallback = Callable[[str, dict[str, Any]], None]
Approver = Callable[[str, dict[str, Any]], bool]


class AgentAPIError(RuntimeError):
    """模型服务不可用、模型不存在等致命错误。"""


def build_system_prompt(react: bool = False) -> str:
    prompt = (
        "你是一个本地助手，可以调用工具完成计算、读写文件和执行代码。\n"
        "规则：\n"
        "1) 涉及计算必须调用 calculator 或 python_exec，不要心算；\n"
        "2) 声称完成某件事之前，必须真的调用过对应工具；\n"
        "3) 工具返回错误时，向用户说明，不要编造结果；\n"
        "4) 只能操作 workspace 目录内的文件，路径越界会被拒绝；\n"
        "5) 只有确实需要多步执行（多次工具调用）的任务才用 todo 建立计划；"
        "简单问答、单次读写不要用 todo；\n"
        "6) 不确定文件内容时先 read_file，不要猜；\n"
        "7) 没有合适工具时，直接说明做不到；不要编造数据，"
        "也不要用代码块假装调用工具。"
    )
    if react:
        prompt += (
            "\n8) 每次调用工具前，先用一句话写出你的判断和下一步计划，"
            "这段文字会展示给用户；然后再发起工具调用。"
        )
    return prompt


@dataclass
class ToolRecord:
    step: int
    name: str
    args: dict[str, Any]
    ok: bool
    output: str
    error: str
    elapsed: float
    started_at: float


@dataclass
class TurnResult:
    text: str
    stop_reason: str
    steps: int
    records: list[ToolRecord] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0
    warnings: list[str] = field(default_factory=list)
    session_id: str = ""


class AgentRuntime:
    def __init__(
        self,
        cfg: Config,
        client: Any | None = None,
        on_event: EventCallback | None = None,
        approver: Approver | None = None,
    ) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        configure_workspace(cfg.workspace)

        self.client = client or OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=120.0,
            max_retries=0,
        )
        self.history = HistoryStore(cfg.history_dir) if cfg.persist_history else None
        self.context = ContextManager(
            budget=cfg.context_tokens,
            summarizer=self._summarize if cfg.summarize else None,
        )
        self.on_event: EventCallback = on_event or (lambda kind, data: None)
        self.approver = approver
        self.tracer: Tracer | None = None
        self.session_id = "local"
        self.messages: list[dict[str, Any]] = []
        self.turn_count = 0

    # -- 生命周期 -----------------------------------------------------------
    def start(self, resume: str | None = None) -> str:
        system = {"role": "system", "content": build_system_prompt(self.cfg.react)}
        if resume:
            if self.history is None:
                raise ValueError("未启用历史持久化，无法恢复会话")
            session_id = self.history.latest() if resume == "latest" else resume
            if not session_id or not self.history.exists(session_id):
                raise FileNotFoundError(f"会话不存在: {resume}")
            loaded = self.history.load(session_id)
            self.session_id = session_id
            loaded = self._repair_loaded(loaded)
            if loaded and loaded[0].get("role") == "system":
                loaded[0] = system
            else:
                loaded.insert(0, system)
            self.messages = loaded
        elif self.history is not None:
            self.session_id = self.history.new_session(
                {"model": self.cfg.model, "workspace": str(self.cfg.workspace)}
            )
            self.messages = [system]
            self._persist(system)
        else:
            self.session_id = "local-" + time.strftime("%H%M%S")
            self.messages = [system]

        self.turn_count = sum(1 for m in self.messages if m.get("role") == "user")
        self.context.reset()
        self._init_tracer()
        return self.session_id

    @staticmethod
    def _repair_loaded(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """上次进程如果在「assistant 已返回 tool_calls、工具还没执行」时被杀，
        历史里会留下没有 tool 结果的消息，直接恢复会被 API 拒绝。
        这里截断到最早缺口，并补一条说明性的 tool 结果。"""
        pending: dict[str, tuple[int, str]] = {}
        for index, message in enumerate(messages):
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    call_id = call.get("id")
                    if call_id:
                        pending[call_id] = (index, call.get("function", {}).get("name", ""))
            elif message.get("role") == "tool":
                pending.pop(message.get("tool_call_id"), None)
        if not pending:
            return messages

        first_gap = min(index for index, _ in pending.values())
        repaired = list(messages[: first_gap + 1])
        for call_id, (_, name) in pending.items():
            repaired.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": f"（上次运行在调用 {name} 时中断，工具没有返回结果）",
                }
            )
        return repaired

    def reset(self) -> str:
        if self.history is not None and self.history.exists(self.session_id):
            self.history.archive(self.session_id)
        if self.tracer is not None:
            self.tracer.close()
        self.context.reset()
        self.messages = [
            {"role": "system", "content": build_system_prompt(self.cfg.react)}
        ]
        self.turn_count = 0
        if self.history is not None:
            self.session_id = self.history.new_session(
                {"model": self.cfg.model, "workspace": str(self.cfg.workspace)}
            )
        else:
            self.session_id = "local-" + time.strftime("%H%M%S")
        if self.history is not None:
            self._persist(self.messages[0])
        self._init_tracer()
        return self.session_id

    def close(self) -> None:
        if self.tracer is not None:
            self.tracer.close()
            self.tracer = None

    def _init_tracer(self) -> None:
        path = (
            self.cfg.trace_dir / f"{self.session_id}.jsonl"
            if self.cfg.trace
            else None
        )
        self.tracer = Tracer(
            path,
            {
                "session": self.session_id,
                "model": self.cfg.model,
                "workspace": str(self.cfg.workspace),
                "base_url": self.cfg.base_url,
                "react": self.cfg.react,
            },
        )

    # -- 主循环 -------------------------------------------------------------
    def run_turn(self, user_input: str) -> TurnResult:
        cfg = self.cfg
        turn_started = time.monotonic()
        turn_started_wall = time.time()
        records: list[ToolRecord] = []
        warnings: list[str] = []
        repeat_counts: Counter[tuple[str, str]] = Counter()
        consecutive_errors = 0
        empty_retry_used = False
        claim_retry_used = False
        stop_reason = "final"
        final_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        step_used = 0

        self.turn_count += 1
        if self.tracer is not None:
            self.tracer.event("turn_start", turn=self.turn_count, user=user_input)
        self.messages.append({"role": "user", "content": user_input})
        self._persist(self.messages[-1])

        deadline = turn_started + cfg.max_turn_seconds
        while step_used < cfg.max_steps:
            if time.monotonic() >= deadline:
                stop_reason = "time_budget"
                final_text = f"（单轮耗时超过 {cfg.max_turn_seconds:.0f}s，已停止）"
                break
            step_used += 1

            prepared = self.context.build(self.messages)
            if self.context.did_compress:
                if self.tracer is not None:
                    self.tracer.event(
                        "context_compressed",
                        dropped=self.context._cut,
                        summary_chars=len(self.context.summary_text),
                    )
                self.on_event(
                    "compressed",
                    {
                        "summary": self.context.summary_text,
                        "sent": len(prepared),
                        "total": len(self.messages),
                    },
                )

            try:
                response = self._chat(prepared)
            except AgentAPIError as exc:
                stop_reason = "api_error"
                final_text = str(exc)
                self.on_event("error", {"message": str(exc)})
                break

            usage = getattr(response, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens += getattr(usage, "completion_tokens", 0) or 0

            message = response.choices[0].message
            assistant = message.model_dump(exclude_none=True)
            self.messages.append(assistant)
            self._persist(assistant)
            if self.tracer is not None:
                self.tracer.event(
                    "model_response",
                    step=step_used,
                    content=(message.content or "")[:1000],
                    tool_calls=[
                        {
                            "name": call.function.name,
                            "args": (call.function.arguments or "")[:1000],
                        }
                        for call in (message.tool_calls or [])
                    ],
                    prompt_tokens=prompt_tokens,
                    completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                )

            if cfg.react and message.content:
                self.on_event("thought", {"text": message.content})
            if prompt_tokens > cfg.hard_context_tokens:
                stop_reason = "context_limit"
                final_text = (
                    f"（上下文达到 {prompt_tokens} tokens，超过 {cfg.hard_context_tokens}，"
                    "已停止。请 reset 或缩短问题。）"
                )
                break

            if not message.tool_calls:
                text = (message.content or "").strip()
                if not text:
                    if not empty_retry_used:
                        empty_retry_used = True
                        self._system_note("你上一条回答是空的。请直接给出最终回答，或调用工具。")
                        continue
                    stop_reason = "empty_answer"
                    final_text = "(模型返回空回答)"
                    break

                issues = verify_claims(
                    text, records, cfg.workspace, turn_started_wall
                )
                if issues and not claim_retry_used:
                    claim_retry_used = True
                    warnings.extend(issues)
                    if self.tracer is not None:
                        self.tracer.event("claim_mismatch", issues=issues)
                    self.on_event("claim_mismatch", {"issues": issues})
                    self._system_note(
                        "系统核验发现你的回答与工具记录不一致："
                        + "；".join(issues)
                        + "。请核实；如果没做成就如实说明，或者现在补做。"
                    )
                    continue
                if issues:
                    warnings.extend(issues)
                    self.on_event(
                        "claim_mismatch", {"issues": issues, "unresolved": True}
                    )
                final_text = text
                stop_reason = "final"
                break

            repeated = False
            responded: set[str] = set()
            for index, call in enumerate(message.tool_calls):
                call_id = call.id or f"call_{step_used}_{index}"
                name = call.function.name
                raw_args = call.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError as exc:
                    record = ToolRecord(
                        step_used,
                        name,
                        {},
                        False,
                        "",
                        f"参数不是合法 JSON: {exc}",
                        0.0,
                        time.time(),
                    )
                    records.append(record)
                    self._append_tool(
                        call_id,
                        ToolResult(False, error=record.error),
                        record,
                    )
                    responded.add(call_id)
                    continue
                if not isinstance(args, dict):
                    args = {"value": args}

                signature = (
                    name,
                    json.dumps(args, sort_keys=True, ensure_ascii=False),
                )
                repeat_counts[signature] += 1
                if repeat_counts[signature] >= cfg.repeat_limit:
                    repeated = True
                    stop_reason = "repeated_call"
                    final_text = (
                        f"检测到你重复调用 {name} 且参数相同，为避免死循环已停止。"
                        "请换一种方法，或直接说明当前进展和限制。"
                    )
                    self.on_event(
                        "stop", {"reason": stop_reason, "detail": final_text}
                    )
                    # 这条 assistant 消息里可能还有没执行的 tool_call，
                    # 必须全部补上 tool 结果，否则历史/上下文会留下坏消息。
                    for rest_index, rest in enumerate(message.tool_calls):
                        rest_id = rest.id or f"call_{step_used}_{rest_index}"
                        if rest_id in responded:
                            continue
                        rest_record = ToolRecord(
                            step_used,
                            rest.function.name,
                            {},
                            False,
                            "",
                            "未执行：检测到重复调用，本轮已停止",
                            0.0,
                            time.time(),
                        )
                        records.append(rest_record)
                        self._append_tool(
                            rest_id,
                            ToolResult(False, error=rest_record.error),
                            rest_record,
                        )
                        responded.add(rest_id)
                    break

                self.on_event(
                    "tool_call",
                    {"step": step_used, "name": name, "args": args},
                )
                result = execute_tool(
                    name,
                    args,
                    approver=self.approver,
                    auto_approve=cfg.auto_approve,
                )
                record = ToolRecord(
                    step_used,
                    name,
                    args,
                    result.ok,
                    result.output,
                    result.error,
                    result.elapsed,
                    time.time(),
                )
                records.append(record)
                self.on_event(
                    "tool_result",
                    {
                        "step": step_used,
                        "name": name,
                        "ok": result.ok,
                        "text": result.text()[:400],
                        "elapsed": result.elapsed,
                    },
                )
                self._append_tool(call_id, result, record)
                responded.add(call_id)
                if result.ok:
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1

            if repeated:
                break

            if consecutive_errors >= 3:
                self._system_note(
                    "连续 3 次工具调用失败。请先停下来分析错误原因，不要重复同样的调用；"
                    "必要时向用户说明当前限制。"
                )
                consecutive_errors = 0
        else:
            stop_reason = "max_steps"
            final_text = (
                f"（已达到最大步数 {cfg.max_steps}，强制停止。"
                "通常是模型在重复调用工具，或任务需要拆分。）"
            )

        elapsed = time.monotonic() - turn_started
        if self.tracer is not None:
            self.tracer.event(
                "turn_end",
                turn=self.turn_count,
                stop=stop_reason,
                steps=step_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                elapsed=round(elapsed, 3),
                warnings=warnings,
            )
        return TurnResult(
            text=final_text,
            stop_reason=stop_reason,
            steps=step_used,
            records=records,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed=elapsed,
            warnings=warnings,
            session_id=self.session_id,
        )

    # -- 模型调用 -----------------------------------------------------------
    def _chat(self, messages: list[dict[str, Any]]) -> Any:
        cfg = self.cfg
        delay = 1.0
        attempt = 0
        while True:
            attempt += 1
            try:
                return self.client.chat.completions.create(
                    model=cfg.model,
                    messages=messages,
                    tools=TOOLS,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    timeout=120,
                )
            except openai.NotFoundError as exc:
                raise AgentAPIError(
                    f"模型 '{cfg.model}' 不存在。请检查 `ollama list`，"
                    f"或用 --model 指定已有模型。原始错误: {exc}"
                ) from exc
            except (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.InternalServerError,
            ) as exc:
                error: Exception = exc
            except openai.APIStatusError as exc:
                if exc.status_code < 500:
                    raise AgentAPIError(
                        f"模型服务返回 {exc.status_code}: {exc}"
                    ) from exc
                error = exc

            if attempt >= cfg.api_retries:
                raise AgentAPIError(
                    f"连接模型服务失败（已重试 {attempt} 次）：{error}"
                ) from error
            self.on_event(
                "retry", {"attempt": attempt, "error": str(error)[:300]}
            )
            if self.tracer is not None:
                self.tracer.event(
                    "api_retry", attempt=attempt, error=str(error)[:500]
                )
            time.sleep(delay)
            delay = min(delay * 2, 8.0)

    # -- 上下文摘要 ---------------------------------------------------------
    def _summarize(self, chunk: list[dict[str, Any]], previous: str) -> str:
        rendered = self._render_messages(chunk)[-7000:]
        prompt = (
            "把下面的对话压缩成不超过 200 字的事实摘要。\n"
            "必须保留：用户目标、已确认的结论、涉及的文件路径、未完成事项。\n"
            "不要编造没发生的事；不要写客套话；只输出摘要正文。\n"
        )
        if previous:
            prompt += f"\n已有摘要：{previous}\n"
        prompt += f"\n新增对话：\n{rendered}"
        try:
            response = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=350,
                timeout=90,
            )
            text = (response.choices[0].message.content or "").strip()
        except Exception as exc:  # 摘要失败不能拖垮主循环
            if self.tracer is not None:
                self.tracer.event("summary_failed", error=str(exc)[:500])
            return ""
        return text[:900]

    @staticmethod
    def _render_messages(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.get("role")
            content = str(message.get("content") or "")
            if role == "user":
                parts.append(f"用户: {content[:600]}")
            elif role == "assistant":
                calls = []
                for call in message.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    calls.append(f"{fn.get('name')}({str(fn.get('arguments'))[:200]})")
                suffix = f" 调用: {', '.join(calls)}" if calls else ""
                parts.append(f"助手: {content[:400]}{suffix}")
            elif role == "tool":
                parts.append(f"工具结果: {content[:300]}")
            elif role == "system":
                parts.append(f"系统: {content[:300]}")
        return "\n".join(parts)

    # -- 工具与消息 ---------------------------------------------------------
    def _append_tool(
        self, tool_call_id: str, result: ToolResult, record: ToolRecord
    ) -> None:
        message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result.text()[: self.cfg.max_tool_output],
        }
        self.messages.append(message)
        self._persist(message)
        if self.tracer is not None:
            self.tracer.event(
                "tool_result",
                step=record.step,
                name=record.name,
                ok=record.ok,
                error=record.error,
                elapsed=round(record.elapsed, 3),
            )

    def _system_note(self, text: str) -> None:
        message = {"role": "system", "content": text}
        self.messages.append(message)
        self._persist(message)
        if self.tracer is not None:
            self.tracer.event("system_note", text=text)

    def _persist(self, message: dict[str, Any]) -> None:
        if self.history is not None:
            self.history.append(self.session_id, message)
