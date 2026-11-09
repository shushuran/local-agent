"""脚本化评测：跑固定用例，按断言打分，输出 report.md。

用法：
    python evals/run_evals.py
    python evals/run_evals.py --repeats 2 --only calc,chain
    python evals/run_evals.py --model qwen2.5:7b
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools  # noqa: E402
from config import Config  # noqa: E402
from runtime import AgentRuntime  # noqa: E402

CASES_PATH = Path(__file__).resolve().parent / "cases.json"
EVALS_DIR = Path(__file__).resolve().parent


def load_cases() -> list[dict[str, Any]]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def evaluate(
    case: dict[str, Any],
    result: Any,
    workspace: Path,
    case_dir: Path,
) -> list[str]:
    checks = case.get("checks") or {}
    failures: list[str] = []
    records = result.records
    ok_names = {record.name for record in records if record.ok}

    any_of = checks.get("any_of")
    if any_of:
        groups = [
            evaluate({"checks": sub}, result, workspace, case_dir) for sub in any_of
        ]
        if all(group for group in groups):
            failures.append("any_of 的所有条件都未通过")

    for name in checks.get("tool_called", []):
        if name not in ok_names:
            failures.append(f"未成功调用工具 {name}")

    for group in checks.get("tool_called_any", []):
        if not any(name in ok_names for name in group):
            failures.append(f"以下工具一个都没成功调用: {'/'.join(group)}")

    for name, needle in (checks.get("tool_error_contains") or {}).items():
        hit = any(
            record.name == name
            and not record.ok
            and needle in ((record.error or "") + "\n" + (record.output or ""))
            for record in records
        )
        if not hit:
            failures.append(f"{name} 没有产生包含「{needle}」的失败")

    for rel in checks.get("file_exists", []):
        if not (workspace / rel).is_file():
            failures.append(f"文件不存在: {rel}")

    for rel, needle in (checks.get("file_contains") or {}).items():
        path = workspace / rel
        if not path.is_file():
            failures.append(f"文件不存在: {rel}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        needles = [needle] if isinstance(needle, str) else list(needle)
        if not any(item in text for item in needles):
            failures.append(f"{rel} 不包含 {needles}")

    for rel in checks.get("no_outside_file", []):
        if (case_dir / rel).exists():
            failures.append(f"workspace 外出现了文件: {rel}")

    answer = result.text or ""
    contains = checks.get("answer_contains") or []
    if contains and not any(item in answer for item in contains):
        failures.append(f"回答不包含 {contains}")
    contains_all = checks.get("answer_contains_all") or []
    for item in contains_all:
        if item not in answer:
            failures.append(f"回答缺少「{item}」")
    for item in checks.get("answer_not_contains") or []:
        if item in answer:
            failures.append(f"回答不该包含「{item}」")

    allowed = checks.get("stop_reason")
    if allowed:
        allowed_list = [allowed] if isinstance(allowed, str) else list(allowed)
        if result.stop_reason not in allowed_list:
            failures.append(f"停止原因 {result.stop_reason} 不在 {allowed_list}")

    if "max_steps" in checks and result.steps > checks["max_steps"]:
        failures.append(f"步数 {result.steps} 超过 {checks['max_steps']}")

    if checks.get("warnings_not_empty") and not result.warnings:
        failures.append("期望出现声明核验警告，但没有")

    return failures


def run_case(
    case: dict[str, Any],
    base_cfg: Config,
    run_dir: Path,
    repeat: int,
    verbose: bool,
) -> dict[str, Any]:
    case_dir = run_dir / case["id"] / f"r{repeat}"
    workspace = case_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    for rel, content in (case.get("setup_files") or {}).items():
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    tools.configure_workspace(workspace)
    cfg = dataclasses.replace(
        base_cfg,
        workspace=workspace,
        history_dir=case_dir / "history",
        trace_dir=case_dir / "traces",
        persist_history=False,
        trace=True,
        summarize=False,
        auto_approve=True,
        max_steps=case.get("max_steps", 6),
    )
    cfg.ensure_dirs()

    def on_event(kind: str, data: dict) -> None:
        if verbose and kind in {"tool_call", "tool_result", "retry", "error", "stop"}:
            print(f"    [{kind}] {str(data)[:200]}")

    started = time.time()
    runtime = AgentRuntime(cfg, on_event=on_event)
    result = None
    error = None
    try:
        runtime.start()
        result = runtime.run_turn(case["prompt"])
        failures = evaluate(case, result, workspace, case_dir)
    except Exception as exc:  # 单个用例失败不能中断整轮评测
        error = f"{type(exc).__name__}: {exc}"
        failures = [error]
    finally:
        runtime.close()

    record = {
        "id": case["id"],
        "repeat": repeat,
        "passed": not failures,
        "failures": failures,
        "error": error,
        "elapsed": round(time.time() - started, 2),
        "steps": result.steps if result else 0,
        "prompt_tokens": result.prompt_tokens if result else 0,
        "stop_reason": result.stop_reason if result else "error",
        "answer": (result.text if result else "")[:500],
        "workspace": str(workspace),
    }
    return record


def build_report(
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    model: str,
    repeats: int,
    run_dir: Path,
) -> str:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_case.setdefault(record["id"], []).append(record)

    total = len(records)
    passed = sum(1 for record in records if record["passed"])
    steps = [record["steps"] for record in records if record["steps"]]
    elapsed = [record["elapsed"] for record in records]
    prompts = [record["prompt_tokens"] for record in records if record["prompt_tokens"]]
    p95 = sorted(elapsed)[max(0, int(len(elapsed) * 0.95) - 1)] if elapsed else 0

    lines = [
        "# local-agent 评测报告",
        "",
        f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型：`{model}`",
        f"- 用例：{len(cases)} 个 × {repeats} 次",
        f"- 通过：{passed}/{total}（{passed / total * 100:.1f}%）",
        f"- 平均步数：{statistics.mean(steps):.1f}" if steps else "- 平均步数：0",
        f"- P95 耗时：{p95:.1f}s",
        f"- 最大 prompt tokens：{max(prompts) if prompts else 0}",
        f"- 运行目录：`{run_dir}`",
        "",
        "| 用例 | 通过 | 平均步数 | 平均耗时 | 停止原因 | 失败原因 |",
        "|---|---|---|---|---|---|",
    ]
    for case in cases:
        case_records = by_case.get(case["id"], [])
        if not case_records:
            continue
        pass_count = sum(1 for record in case_records if record["passed"])
        case_steps = statistics.mean([r["steps"] for r in case_records]) if case_records else 0
        case_elapsed = statistics.mean([r["elapsed"] for r in case_records]) if case_records else 0
        reasons = sorted({r["stop_reason"] for r in case_records})
        failure_texts = []
        for record in case_records:
            for failure in record["failures"]:
                if failure not in failure_texts:
                    failure_texts.append(failure)
        reason = "；".join(failure_texts)[:160].replace("|", "/")
        lines.append(
            f"| {case['id']} | {pass_count}/{len(case_records)} | "
            f"{case_steps:.1f} | {case_elapsed:.1f}s | {','.join(reasons)} | {reason} |"
        )
    lines.append("")
    lines.append("> 说明：评测只判断工具是否真的被调用、文件是否真的写入、回答是否包含关键事实；")
    lines.append("> 不判断措辞是否漂亮。失败详情见各用例目录下的 traces/。")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="local-agent 脚本化评测")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--only", help="只跑指定 id，逗号分隔")
    parser.add_argument("--model", help="覆盖模型名")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--react", action="store_true")
    args = parser.parse_args()

    cases = load_cases()
    if args.only:
        wanted = {item.strip() for item in args.only.split(",")}
        cases = [case for case in cases if case["id"] in wanted]
        if not cases:
            print(f"没有匹配的用例: {args.only}")
            return 2

    base_cfg = Config()
    if args.model:
        base_cfg.model = args.model
    base_cfg.react = args.react
    base_cfg.temperature = 0.0
    base_cfg.max_tokens = 512
    base_cfg.max_turn_seconds = 90

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = EVALS_DIR / ".runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"模型: {base_cfg.model} | 用例: {len(cases)} | 重复: {args.repeats}")
    records: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        for index, case in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] {case['id']} (r{repeat}) ...", end="", flush=True)
            record = run_case(case, base_cfg, run_dir, repeat, args.verbose)
            records.append(record)
            mark = "PASS" if record["passed"] else "FAIL"
            print(
                f" {mark}  {record['steps']} steps, {record['elapsed']}s"
                + ("" if record["passed"] else f"  <- {'；'.join(record['failures'])[:160]}")
            )

    report = build_report(records, cases, base_cfg.model, args.repeats, run_dir)
    report_path = EVALS_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8")
    (EVALS_DIR / "last_run.json").write_text(
        json.dumps(
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": base_cfg.model,
                "repeats": args.repeats,
                "run_dir": str(run_dir),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(report)
    print(f"报告已写入: {report_path}")
    return 0 if all(record["passed"] for record in records) else 1


if __name__ == "__main__":
    sys.exit(main())
