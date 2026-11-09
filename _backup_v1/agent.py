"""最小可用的本地 Agent：Ollama + 原生 function calling + 手写 ReAct 循环。

核心只有一件事：
    把 [系统提示 + 历史消息 + 用户问题 + 工具清单] 发给模型
    -> 模型要么直接回答，要么返回 tool_calls
    -> 执行工具，把结果作为 role="tool" 追加进历史
    -> 再发给模型，直到它不再调工具
理解了这个循环，所有 Agent 框架都只是它的包装。
"""

import json
import sys
import time


from openai import OpenAI

from tools import FUNCS, TOOLS

MODEL = "qwen-agent"
MAX_STEPS = 8  # 最多几轮工具调用，防止死循环

client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")

SYSTEM_PROMPT = (
    "你是一个本地助手，可以调用工具完成计算、读写文件和执行代码。"
    "规则：1) 涉及计算必须调用 calculator 或 python_exec，不要心算；"
    "2) 声称完成某件事之前，必须真的调用过对应工具；"
    "3) 工具返回错误时，向用户说明，不要编造结果。"
)


def run_turn(messages: list, user_input: str) -> str:
    """执行一轮完整的「思考-调用-再思考」，返回最终回答。"""
    messages.append({"role": "user", "content": user_input})

    for step in range(1, MAX_STEPS + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.2,
            max_tokens=1024,
        )
        msg = resp.choices[0].message

        # 必须把 assistant 这条消息（含 tool_calls）原样放回历史，
        # 否则下一轮模型会认为工具结果没有来源。
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content or "(空回答)"

        for i, tc in enumerate(msg.tool_calls):
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            print(f"  [step {step}] -> {name}({args})")
            fn = FUNCS.get(name)
            try:
                result = fn(**args) if fn else f"未知工具: {name}"
            except Exception as e:  # 工具报错也要回灌给模型，而不是直接崩
                result = f"工具执行失败: {e}"

            text = str(result)
            print(f"      {text[:200]}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id or f"call_{step}_{i}",
                    "content": text[:4000],
                }
            )

    return "（已达到最大步数，强制停止。通常是模型在重复调用同一个工具。）"


def main() -> None:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("本地 Agent 已启动。输入 exit 退出，输入 reset 清空上下文。")

    while True:
        try:
            question = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("上下文已清空。")
            continue
        if not question:
            continue

        t0 = time.time()
        try:
            answer = run_turn(messages, question)
        except Exception as e:
            print(f"\n[请求失败] {e}")
            print("检查：Ollama 是否在运行？模型名是否和 Modelfile 里一致？")
            continue

        elapsed = time.time() - t0
        print(f"\nAgent> {answer}")
        print(f"[耗时 {elapsed:.1f}s | 上下文消息数 {len(messages)}]")


if __name__ == "__main__":
    sys.exit(main())
