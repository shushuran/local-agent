"""上下文预算：估算 token、按整组裁剪、滚动摘要。

三条硬规则：
1. 裁剪点必须落在 user 消息边界，绝不把 assistant(tool_calls) 和它的 tool 结果拆开。
2. 最近一组（最后一轮 user 及其后续）永远保留。
3. 摘要失败时退化为机械摘要，不能让主循环崩掉。
"""

from __future__ import annotations

from typing import Any, Callable

SUMMARY_MARK = "【早期对话摘要】"


def estimate_text(text: str) -> int:
    """粗略 token 估算：中文约 1.2 字/token，其他约 3.6 字符/token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk / 1.2 + other / 3.6) + 1


def estimate_message(message: dict[str, Any]) -> int:
    total = 8  # role / 分隔符等固定开销
    total += estimate_text(str(message.get("content") or ""))
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        total += estimate_text(str(fn.get("name") or ""))
        total += estimate_text(str(fn.get("arguments") or ""))
    for key in ("name", "tool_call_id"):
        total += estimate_text(str(message.get(key) or ""))
    return total


def estimate_messages(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_message(m) for m in messages)


def _split_head(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    head: list[dict[str, Any]] = []
    index = 0
    while index < len(messages) and messages[index].get("role") == "system":
        head.append(messages[index])
        index += 1
    return head, messages[index:]


class ContextManager:
    def __init__(
        self,
        budget: int,
        summarizer: Callable[[list[dict[str, Any]], str], str] | None = None,
        summary_chars: int = 900,
    ) -> None:
        self.budget = budget
        self.summarizer = summarizer
        self.summary_chars = max(120, min(summary_chars, max(120, budget // 3)))
        self._summary = ""
        self._cut = 0  # body 中已裁掉的条数
        self._summarized_upto = 0  # body 中已摘要到的位置
        self.did_compress = False

    # -- 对外 ---------------------------------------------------------------
    def reset(self) -> None:
        self._summary = ""
        self._cut = 0
        self._summarized_upto = 0
        self.did_compress = False

    @property
    def summary_text(self) -> str:
        return self._summary

    def build(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.did_compress = False
        head, body = _split_head(messages)
        if not body:
            return messages

        fits = estimate_messages(head + body) <= self.budget
        if not self._summary and self._cut == 0 and fits:
            return messages

        cut = min(self._cut, len(body))
        # 先按「不含摘要」的预算裁到位，给摘要预留空间。
        reserve = (self.summary_chars // 2 + 40) if self.summarizer else 0
        while estimate_messages(head + body[cut:]) + reserve > self.budget:
            new_cut = self._advance(body, cut)
            if new_cut <= cut:
                break
            cut = new_cut
            self.did_compress = True

        if cut > self._summarized_upto:
            self._summarize_into(body, cut)

        visible = self._compose(head, body, cut)
        # 摘要本身也占 token；仍超预算就继续裁，并补摘要。
        while estimate_messages(visible) > self.budget:
            new_cut = self._advance(body, cut)
            if new_cut <= cut:
                break
            cut = new_cut
            self.did_compress = True
            if cut > self._summarized_upto:
                self._summarize_into(body, cut)
            visible = self._compose(head, body, cut)

        self._cut = cut
        return visible

    def _summarize_into(self, body: list[dict[str, Any]], cut: int) -> None:
        chunk = body[self._summarized_upto:cut]
        if not chunk:
            return
        new_summary = self._summarize(chunk)
        if new_summary:
            self._summary = new_summary
            self.did_compress = True
        self._summarized_upto = cut

    # -- 内部 ---------------------------------------------------------------
    def _compose(
        self,
        head: list[dict[str, Any]],
        body: list[dict[str, Any]],
        cut: int,
    ) -> list[dict[str, Any]]:
        result = list(head)
        if cut > 0 and self._summary:
            result.append(
                {
                    "role": "system",
                    "content": f"{SUMMARY_MARK}（更早的 {cut} 条消息已压缩；"
                    f"原始记录仍在 history/ 中）\n{self._summary}",
                }
            )
        result.extend(body[cut:])
        return result

    @staticmethod
    def _advance(body: list[dict[str, Any]], cut: int) -> int:
        starts = [i for i, m in enumerate(body) if m.get("role") == "user"]
        if not starts:
            return len(body)
        later = [s for s in starts if s > cut]
        if later:
            return later[0]
        return cut  # 已到最后一组，保留

    def _summarize(self, chunk: list[dict[str, Any]]) -> str:
        if self.summarizer is None:
            return ""
        try:
            text = self.summarizer(chunk, self._summary)
        except Exception:
            return self._mechanical(chunk)
        text = (text or "").strip()
        if not text:
            text = self._mechanical(chunk)
        return text[: self.summary_chars]

    @staticmethod
    def _mechanical(chunk: list[dict[str, Any]]) -> str:
        asks = []
        tools = []
        for message in chunk:
            if message.get("role") == "user":
                asks.append(str(message.get("content") or "").strip().replace("\n", " ")[:80])
            for call in message.get("tool_calls") or []:
                name = (call.get("function") or {}).get("name")
                if name:
                    tools.append(str(name))
        parts = []
        if asks:
            parts.append("用户此前的请求：" + "；".join(asks[-6:]))
        if tools:
            parts.append("调用过的工具：" + "、".join(dict.fromkeys(tools)))
        return "；".join(parts) or "（更早的对话内容已省略）"
