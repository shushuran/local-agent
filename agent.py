"""本地 Agent 的命令行入口。

用法：
    python agent.py                    # 交互模式
    python agent.py --ask "3.5 的 12 次方"   # 单轮
    python agent.py --resume latest     # 恢复最近会话
    python agent.py --list-sessions
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

from config import Config
from history import HistoryStore
from runtime import AgentRuntime, TurnResult

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except (OSError, ValueError):
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="本地 Agent：Ollama + function calling + 手写循环"
    )
    parser.add_argument("--model", help="模型名（默认读 AGENT_MODEL 或 qwen-agent）")
    parser.add_argument("--base-url", help="OpenAI 兼容地址")
    parser.add_argument("--max-steps", type=int, help="单轮最多几步")
    parser.add_argument("--temperature", type=float, help="采样温度")
    parser.add_argument("--context-tokens", type=int, help="消息预算（token 估算）")
    parser.add_argument("--workspace", help="工作区目录（默认 ./workspace）")
    parser.add_argument(
        "--react",
        action="store_true",
        default=None,
        help="显式 Thought 模式：调用工具前先说明计划",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="恢复会话：--resume <id> 或 --resume（最近一个）",
    )
    parser.add_argument(
        "--list-sessions", action="store_true", help="列出历史会话后退出"
    )
    parser.add_argument(
        "--status", action="store_true", help="检查模型服务与可用模型后退出"
    )
    parser.add_argument("--ask", help="只问一个问题，打印结果后退出")
    parser.add_argument("--yes", action="store_true", help="自动确认覆盖写等敏感操作")
    parser.add_argument("--no-history", action="store_true", help="不写 history/")
    parser.add_argument("--no-trace", action="store_true", help="不写 traces/")
    return parser


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    changes: dict[str, object] = {}
    for field_name, value in (
        ("model", args.model),
        ("base_url", args.base_url),
        ("max_steps", args.max_steps),
        ("temperature", args.temperature),
        ("context_tokens", args.context_tokens),
    ):
        if value is not None:
            changes[field_name] = value
    if args.react is not None:
        changes["react"] = True
    if args.workspace:
        changes["workspace"] = Path(args.workspace).expanduser().resolve()
    if args.yes:
        changes["auto_approve"] = True
    if args.no_history:
        changes["persist_history"] = False
    if args.no_trace:
        changes["trace"] = False
    return dataclasses.replace(cfg, **changes)


def make_printer() -> callable:
    def printer(kind: str, data: dict) -> None:
        if kind == "thought":
            print(f"  思考：{data['text']}")
        elif kind == "tool_call":
            args = json.dumps(data["args"], ensure_ascii=False)
            print(f"  [step {data['step']}] -> {data['name']}({args[:300]})")
        elif kind == "tool_result":
            mark = "ok" if data["ok"] else "fail"
            text = str(data["text"]).replace("\n", " ")[:200]
            print(f"      [{mark}] {text}  ({data['elapsed']:.2f}s)")
        elif kind == "retry":
            print(f"  [重试 {data['attempt']}] {data['error'][:200]}")
        elif kind == "compressed":
            print(
                f"  [上下文压缩] 发送 {data['sent']} 条 / 共 {data['total']} 条，"
                f"摘要 {len(data['summary'])} 字"
            )
        elif kind == "claim_mismatch":
            suffix = "（仍未解决）" if data.get("unresolved") else "，要求模型核实"
            print("  [!] 声明核验：" + "；".join(data["issues"]) + suffix)
        elif kind == "error":
            print(f"  [错误] {data['message']}")
        elif kind == "stop":
            print(f"  [停止] {data['detail']}")

    return printer


def make_approver() -> callable:
    def approver(name: str, args: dict) -> bool:
        print(f"\n[需要确认] {name} {json.dumps(args, ensure_ascii=False)[:400]}")
        try:
            answer = input("允许执行？[y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return answer in {"y", "yes", "是"}

    return approver


def print_turn(result: TurnResult, runtime: AgentRuntime) -> None:
    print(f"\nAgent> {result.text}")
    extra = ""
    if result.stop_reason != "final":
        extra = f" | 停止原因 {result.stop_reason}"
    print(
        f"[耗时 {result.elapsed:.1f}s | 步数 {result.steps} | "
        f"上下文消息数 {len(runtime.messages)}{extra}]"
    )
    if result.prompt_tokens:
        print(
            f"[tokens 输入 {result.prompt_tokens} / 输出 {result.completion_tokens}]"
        )
    if result.warnings:
        print("[核验警告] " + "；".join(result.warnings))


def print_service_status(cfg: Config) -> int:
    print(f"api      : {cfg.base_url}")
    print(f"model    : {cfg.model}")
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=10)
    try:
        listing = client.models.list()
        names = [model.id for model in listing.data]
    except Exception as exc:
        print(f"[错误] 无法访问模型服务：{type(exc).__name__}: {exc}")
        print("检查：Ollama 是否启动？base_url 是否正确？")
        return 1
    if not names:
        print("[错误] 服务可达，但没有可用模型")
        return 1
    print("available: " + ", ".join(names))
    matched = cfg.model in names or any(
        name.startswith(cfg.model + ":") for name in names
    )
    if not matched:
        print(f"[警告] 配置的模型 {cfg.model} 不在列表里；用 --model 指定")
        return 1
    print("[OK] 模型服务可用")
    return 0


def print_sessions(cfg: Config) -> None:
    store = HistoryStore(cfg.history_dir)
    sessions = store.list_sessions()
    if not sessions:
        print("(没有历史会话)")
        return
    print(f"{'session':<26} {'创建时间':<20} {'模型':<14} 行数")
    for item in sessions:
        print(
            f"{item.get('session', ''):<26} "
            f"{item.get('created', ''):<20} "
            f"{str(item.get('model', '')):<14} "
            f"{item.get('lines', 0)}"
        )


HELP = """命令：
  /reset   归档当前会话并清空上下文
  /session 显示当前会话 id 和消息数
  /status  检查模型服务与可用模型
  /help    显示本帮助
  exit     退出（也可以 Ctrl+C）
"""


def main() -> int:
    args = build_parser().parse_args()
    cfg = build_config(args)

    if args.list_sessions:
        print_sessions(cfg)
        return 0

    if args.status:
        return print_service_status(cfg)

    runtime = AgentRuntime(
        cfg,
        on_event=make_printer(),
        approver=None if cfg.auto_approve else make_approver(),
    )
    try:
        try:
            session_id = runtime.start(args.resume)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[启动失败] {exc}")
            return 2

        print("本地 Agent v2（Ollama + 原生 function calling）")
        print(f"python   : {sys.executable}")
        print(f"model    : {cfg.model}")
        print(f"api      : {cfg.base_url}  (接口根路径，浏览器打开会 404，属正常)")
        print(f"workspace: {cfg.workspace}")
        print(f"session  : {session_id}")
        if runtime.turn_count:
            print(f"已恢复 {runtime.turn_count} 轮历史对话")
        print("输入 /help 查看命令，exit 退出。")

        if args.ask is not None:
            result = runtime.run_turn(args.ask)
            print_turn(result, runtime)
            return 0 if result.stop_reason != "api_error" else 1

        if not sys.stdin.isatty():
            print("[错误] 当前标准输入不是交互终端，无法读取命令。")
            print("       Code Runner 的 Output 面板不支持 stdin；请改用集成终端")
            print("       （本项目已设置 code-runner.runInTerminal=true），")
            print('       或使用：python agent.py --ask "你的问题"')
            return 2

        while True:
            print("\n你> ", end="", flush=True)
            try:
                line = input().strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            lowered = line.lower()
            if lowered in {"exit", "quit", "/exit", "/quit"}:
                break
            if lowered in {"/reset", "reset"}:
                session_id = runtime.reset()
                print(f"上下文已清空，新会话: {session_id}")
                continue
            if lowered in {"/help", "help"}:
                print(HELP)
                continue
            if lowered == "/status":
                print_service_status(cfg)
                continue
            if lowered == "/session":
                print(f"session: {runtime.session_id} | 消息数: {len(runtime.messages)}")
                continue

            started = time.time()
            try:
                result = runtime.run_turn(line)
            except Exception as exc:  # 兜底：主循环不应该把 CLI 打崩
                print(f"\n[未捕获错误] {type(exc).__name__}: {exc}")
                continue
            if result.elapsed == 0:
                result.elapsed = time.time() - started
            print_turn(result, runtime)
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    sys.exit(main())
