"""Agent 可调用的工具集合 + 工具注册表。

每个工具 = 一个普通 Python 函数 + 一段 JSON Schema + 权限元数据。
模型只能看到 schema；真正的执行、参数校验、路径限制、超时和人工确认都在这里。
"""

from __future__ import annotations

import ast
import fnmatch
import json
import operator
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
BASE = (ROOT / "workspace").resolve()
CHILD = ROOT / "sandbox_child.py"
MAX_TOOL_OUTPUT = 4000
MAX_WRITE_BYTES = 200_000
MAX_READ_BYTES = 1_000_000

BASE.mkdir(parents=True, exist_ok=True)


def configure_workspace(path: str | Path) -> Path:
    """切换工作区（评测时为每个用例准备独立目录）。"""
    global BASE
    BASE = Path(path).expanduser().resolve()
    BASE.mkdir(parents=True, exist_ok=True)
    return BASE


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("指数过大，拒绝计算")
        return _BIN_OPS[type(node.op)](left, right)
    raise ValueError("表达式里只允许数字和 + - * / // % ** 和括号")


def calculator(expression: str) -> str:
    """计算四则运算/幂/取模表达式。"""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("表达式不能为空")
    tree = ast.parse(expression, mode="eval")
    return str(_eval_node(tree.body))


def _sandbox_env() -> dict[str, str]:
    keep = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT")
    env = {key: os.environ[key] for key in keep if key in os.environ}
    tmp = BASE / ".tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = str(tmp)
    env["TMP"] = str(tmp)
    env["HOME"] = str(BASE)
    env["USERPROFILE"] = str(BASE)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def python_exec(code: str) -> str:
    """在软沙箱里执行一段 Python 代码，返回 stdout/stderr（限时 20 秒）。"""
    if not isinstance(code, str) or not code.strip():
        raise ValueError("代码不能为空")
    try:
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", "-I", str(CHILD), str(BASE)],
            input=code,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            cwd=str(BASE),
            env=_sandbox_env(),
        )
    except subprocess.TimeoutExpired:
        return "执行超时（>20s），请简化代码"
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        raise RuntimeError(out[:2000] or f"代码以退出码 {proc.returncode} 结束")
    return out[:MAX_TOOL_OUTPUT] or "(执行成功，无输出)"


def _safe_path(path: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("路径不能为空")
    if "\x00" in path:
        raise ValueError("路径包含非法字符")
    base = BASE.resolve()
    target = (base / path).resolve()
    if target != base and base not in target.parents:
        raise ValueError("路径越界：只允许操作 workspace/ 目录下的文件")
    return target


def read_file(path: str) -> str:
    """读取 workspace 下的文本文件。"""
    target = _safe_path(path)
    if not target.exists():
        return f"文件不存在: {path}"
    if not target.is_file():
        return f"不是文件: {path}"
    data = target.read_bytes()
    if len(data) > MAX_READ_BYTES:
        data = data[:MAX_READ_BYTES]
    text = data.decode("utf-8", errors="replace")
    return text[:MAX_TOOL_OUTPUT]


def write_file(path: str, content: str) -> str:
    """在 workspace 下写入（覆盖）文本文件。"""
    target = _safe_path(path)
    if not isinstance(content, str):
        content = str(content)
    raw = content.encode("utf-8")
    if len(raw) > MAX_WRITE_BYTES:
        raise ValueError(f"内容 {len(raw)} 字节，超过上限 {MAX_WRITE_BYTES}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return f"已写入 {target.relative_to(BASE)}，共 {len(content)} 字符"


def edit_file(path: str, old: str, new: str, count: int = 1) -> str:
    """把文件里的 old 精确替换为 new。"""
    target = _safe_path(path)
    if not target.is_file():
        return f"文件不存在: {path}"
    if not old:
        raise ValueError("old 不能为空")
    text = target.read_text(encoding="utf-8", errors="replace")
    found = text.count(old)
    if found == 0:
        raise ValueError(f"未找到要替换的内容（{len(old)} 字符），请先 read_file 确认原文")
    if count == 1 and found > 1:
        raise ValueError(f"匹配到 {found} 处，请提供更长的唯一片段，或显式指定 count")
    replaced = text.replace(old, new, 0 if count <= 0 else count)
    raw = replaced.encode("utf-8")
    if len(raw) > MAX_WRITE_BYTES:
        raise ValueError(f"结果 {len(raw)} 字节，超过上限 {MAX_WRITE_BYTES}")
    target.write_bytes(raw)
    return f"已更新 {target.relative_to(BASE)}（替换 {min(found, count) if count > 0 else found} 处）"


def list_files(path: str = ".", recursive: bool = False) -> str:
    """列出 workspace 下某个目录的内容。"""
    target = _safe_path(path)
    if not target.is_dir():
        return f"不是目录: {path}"
    if not recursive:
        items = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        if not items:
            return "(空目录)"
        return "\n".join(
            f"{'[D]' if item.is_dir() else '[F]'} {item.name}" for item in items
        )

    lines: list[str] = []
    for dirpath, dirnames, filenames in os.walk(target):
        dirnames[:] = sorted(
            d for d in dirnames if d not in {"__pycache__", ".git", ".tmp", "node_modules"}
        )
        current = Path(dirpath)
        depth = len(current.relative_to(target).parts)
        if depth >= 3:
            dirnames[:] = []
        for name in sorted(filenames)[:200]:
            if ".tmp" in current.parts:
                continue
            lines.append(f"[F] {(current / name).relative_to(BASE)}")
            if len(lines) >= 200:
                return "\n".join(lines) + "\n(已截断)"
    return "\n".join(lines) or "(空目录)"


def search_files(pattern: str, path: str = ".", glob: str = "*") -> str:
    """按正则搜索 workspace 下的文本文件。"""
    target = _safe_path(path)
    if not target.is_dir():
        return f"不是目录: {path}"
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"正则有误: {exc}"

    matches: list[str] = []
    scanned = 0
    skip_dirs = {"__pycache__", ".git", ".tmp", "node_modules", ".venv"}
    for dirpath, dirnames, filenames in os.walk(target):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            if not fnmatch.fnmatch(name, glob):
                continue
            scanned += 1
            if scanned > 400:
                matches.append("(已扫描 400 个文件，停止)")
                return "\n".join(matches)
            file_path = Path(dirpath) / name
            try:
                data = file_path.read_bytes()
            except OSError:
                continue
            if len(data) > MAX_READ_BYTES:
                continue
            text = data.decode("utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    snippet = line.strip()
                    if len(snippet) > 300:
                        snippet = snippet[:300] + "…"
                    matches.append(
                        f"{str(file_path.relative_to(BASE)).replace(chr(92), '/')}:{line_no}: {snippet}"
                    )
                    if len(matches) >= 80:
                        matches.append("(匹配过多，已截断)")
                        return "\n".join(matches)
    return "\n".join(matches) or f"没有匹配: {pattern}"


def current_time() -> str:
    """获取当前本地时间。"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


_PLAN_FILE = ".plan.json"


def _load_plan() -> list[dict[str, Any]]:
    path = BASE / _PLAN_FILE
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = data.get("items") if isinstance(data, dict) else None
    return items if isinstance(items, list) else []


def _render_plan(items: list[dict[str, Any]]) -> str:
    lines = ["# 计划"]
    for index, item in enumerate(items, 1):
        mark = "x" if item.get("done") else " "
        lines.append(f"- [{mark}] {index}. {item.get('text', '')}")
    return "\n".join(lines)


def todo(
    op: str,
    items: list[str] | None = None,
    index: int | None = None,
) -> str:
    """维护 workspace/.plan.json 里的任务计划。"""
    current = _load_plan()
    if op == "set":
        if not items:
            raise ValueError("op=set 时必须提供 items")
        current = [{"text": str(text), "done": False} for text in items]
    elif op in {"done", "undone"}:
        if index is None or not (1 <= index <= len(current)):
            raise ValueError(f"index 必须在 1..{len(current)} 之间")
        current[index - 1]["done"] = op == "done"
    elif op != "list":
        raise ValueError("op 只能是 set / done / undone / list")

    (BASE / _PLAN_FILE).write_text(
        json.dumps({"items": current}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return _render_plan(current)


# ---------------------------------------------------------------------------
# 工具注册表
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolSpec:
    name: str
    func: Callable[..., object]
    schema: dict[str, Any]
    permission: str  # read / write / exec
    timeout: float = 30.0
    requires_approval: bool = False

    def as_openai(self) -> dict[str, Any]:
        return {"type": "function", "function": self.schema}


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: str = ""
    elapsed: float = 0.0

    def text(self) -> str:
        if self.ok:
            return self.output
        return f"工具执行失败: {self.error}"


def _schema(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _build_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="calculator",
            func=calculator,
            permission="read",
            schema=_schema(
                "calculator",
                "计算数学表达式。凡是涉及数字运算，都必须用这个工具，不要心算。"
                "支持 + - * / // % ** 和括号。",
                {
                    "expression": {
                        "type": "string",
                        "description": "要计算的表达式，例如 (3.5)**12 或 123*456",
                    }
                },
                ["expression"],
            ),
        ),
        ToolSpec(
            name="python_exec",
            func=python_exec,
            permission="exec",
            timeout=25.0,
            schema=_schema(
                "python_exec",
                "在 workspace 内的软沙箱里执行 Python 代码（20 秒限制）。"
                "适合数据处理、多步计算、调用标准库。禁止写 workspace 外文件、联网、起子进程。",
                {
                    "code": {
                        "type": "string",
                        "description": "完整可执行的 Python 代码；用 print 输出结果",
                    }
                },
                ["code"],
            ),
        ),
        ToolSpec(
            name="list_files",
            func=list_files,
            permission="read",
            schema=_schema(
                "list_files",
                "列出 workspace 目录下的文件和子目录。不确定有什么文件时先调这个。",
                {
                    "path": {
                        "type": "string",
                        "description": "相对路径，默认为当前目录 .",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "是否递归列出子目录，默认 false",
                    },
                },
                [],
            ),
        ),
        ToolSpec(
            name="read_file",
            func=read_file,
            permission="read",
            schema=_schema(
                "read_file",
                "读取 workspace 下的文本文件内容。",
                {"path": {"type": "string", "description": "相对路径，例如 todo.md"}},
                ["path"],
            ),
        ),
        ToolSpec(
            name="write_file",
            func=write_file,
            permission="write",
            requires_approval=True,
            schema=_schema(
                "write_file",
                "把内容写入 workspace 下的文本文件（会覆盖同名文件，覆盖前需用户确认）。"
                "新建文件不需要确认。",
                {
                    "path": {
                        "type": "string",
                        "description": "相对路径，例如 notes/todo.md",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的完整内容",
                    },
                },
                ["path", "content"],
            ),
        ),
        ToolSpec(
            name="edit_file",
            func=edit_file,
            permission="write",
            schema=_schema(
                "edit_file",
                "把文件中的一段原文精确替换成新内容，比 write_file 更适合局部修改。"
                "old 必须和文件内容完全一致；若出现多处会拒绝，请提供更长的唯一片段。",
                {
                    "path": {"type": "string", "description": "相对路径"},
                    "old": {"type": "string", "description": "要被替换的原文"},
                    "new": {"type": "string", "description": "替换后的内容，可为空"},
                    "count": {
                        "type": "integer",
                        "description": "最多替换几处，默认 1",
                    },
                },
                ["path", "old", "new"],
            ),
        ),
        ToolSpec(
            name="search_files",
            func=search_files,
            permission="read",
            schema=_schema(
                "search_files",
                "在 workspace 里按正则搜索文件内容，返回 路径:行号: 内容。",
                {
                    "pattern": {"type": "string", "description": "Python 正则表达式"},
                    "path": {"type": "string", "description": "搜索起点，默认 ."},
                    "glob": {
                        "type": "string",
                        "description": "文件名通配符，默认 *（例如 *.py）",
                    },
                },
                ["pattern"],
            ),
        ),
        ToolSpec(
            name="todo",
            func=todo,
            permission="write",
            schema=_schema(
                "todo",
                "维护任务计划。只有确实需要多步执行（多次工具调用）的任务才用；"
                "简单问答、单次读写不要用。op=set 建立计划，op=done 标记完成，"
                "op=list 查看。",
                {
                    "op": {
                        "type": "string",
                        "enum": ["set", "done", "undone", "list"],
                        "description": "操作类型",
                    },
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "op=set 时的任务列表",
                    },
                    "index": {
                        "type": "integer",
                        "description": "op=done/undone 时的任务序号，从 1 开始",
                    },
                },
                ["op"],
            ),
        ),
        ToolSpec(
            name="current_time",
            func=current_time,
            permission="read",
            schema=_schema(
                "current_time",
                "获取当前日期和时间。涉及“今天”“现在”的问题必须先调用。",
                {},
                [],
            ),
        ),
    ]


SPECS = _build_specs()
REGISTRY: dict[str, ToolSpec] = {spec.name: spec for spec in SPECS}
TOOLS = [spec.as_openai() for spec in SPECS]
FUNCS = {spec.name: spec.func for spec in SPECS}

_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def _validate_args(spec: ToolSpec, args: dict[str, Any]) -> None:
    params = spec.schema.get("parameters") or {}
    properties = params.get("properties") or {}
    for name in params.get("required") or []:
        if name not in args:
            raise ValueError(f"缺少必填参数: {name}")
    for name, value in args.items():
        if name not in properties:
            raise ValueError(f"未知参数: {name}")
        prop = properties[name] or {}
        expected = prop.get("type")
        if expected in _TYPE_MAP:
            if expected in {"integer", "number"} and isinstance(value, bool):
                raise ValueError(f"参数 {name} 应为 {expected}，收到 bool")
            if not isinstance(value, _TYPE_MAP[expected]):
                raise ValueError(
                    f"参数 {name} 应为 {expected}，收到 {type(value).__name__}"
                )
        choices = prop.get("enum")
        if choices and value not in choices:
            raise ValueError(f"参数 {name} 必须是 {choices} 之一")


def _requires_approval(name: str, args: dict[str, Any]) -> bool:
    if name == "write_file":
        # 新建文件不需要确认；覆盖已有文件才需要。
        try:
            return _safe_path(str(args.get("path", ""))).exists()
        except Exception:
            return True
    spec = REGISTRY.get(name)
    if spec is None:
        return False
    return spec.requires_approval


def execute_tool(
    name: str,
    args: dict[str, Any],
    approver: Callable[[str, dict[str, Any]], bool] | None = None,
    auto_approve: bool = False,
) -> ToolResult:
    spec = REGISTRY.get(name)
    if spec is None:
        return ToolResult(False, error=f"未知工具: {name}")
    if not isinstance(args, dict):
        return ToolResult(False, error="参数必须是 JSON 对象")

    try:
        _validate_args(spec, args)
    except ValueError as exc:
        return ToolResult(False, error=str(exc))

    if _requires_approval(name, args) and not auto_approve:
        if approver is None:
            return ToolResult(False, error="该操作需要人工确认，但当前没有确认通道")
        try:
            allowed = approver(name, args)
        except Exception as exc:
            return ToolResult(False, error=f"确认过程出错: {exc}")
        if not allowed:
            return ToolResult(False, error="用户拒绝执行该工具调用")

    started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(spec.func, **args)
    try:
        output = future.result(timeout=spec.timeout)
        text = "" if output is None else str(output)
        return ToolResult(True, output=text, elapsed=time.perf_counter() - started)
    except FutureTimeout:
        return ToolResult(
            False,
            error=f"工具执行超时（>{spec.timeout:.0f}s）",
            elapsed=time.perf_counter() - started,
        )
    except Exception as exc:
        return ToolResult(
            False,
            error=f"{type(exc).__name__}: {exc}",
            elapsed=time.perf_counter() - started,
        )
    finally:
        executor.shutdown(wait=False)
