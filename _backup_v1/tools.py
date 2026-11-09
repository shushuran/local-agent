"""Agent 可调用的工具集合。

每个工具 = 一个普通 Python 函数 + 一段 JSON Schema 描述。
注意：模型只能看到 schema 里的 description，看不到函数实现。
描述写得清不清楚，直接决定它会不会用、用得对不对。
"""

import ast
import operator
import subprocess
import sys
import time
from pathlib import Path

# 所有文件操作都被限制在这个目录里
BASE = Path(__file__).resolve().parent / "workspace"
BASE.mkdir(exist_ok=True)

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


def _eval_node(node):
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
    tree = ast.parse(expression, mode="eval")
    return str(_eval_node(tree.body))


def python_exec(code: str) -> str:
    """在本机执行一段 Python 代码，返回 stdout/stderr（限时 20 秒）。"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(BASE),
        )
    except subprocess.TimeoutExpired:
        return "执行超时（>20s），请简化代码"
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return out[:4000] or "(执行成功，无输出)"


def _safe_path(path: str) -> Path:
    p = (BASE / path).resolve()
    if p != BASE and BASE not in p.parents:
        raise ValueError("路径越界：只允许操作 workspace/ 目录下的文件")
    return p


def read_file(path: str) -> str:
    """读取 workspace 下的文本文件。"""
    p = _safe_path(path)
    if not p.exists():
        return f"文件不存在: {path}"
    if not p.is_file():
        return f"不是文件: {path}"
    return p.read_text(encoding="utf-8", errors="replace")[:4000]


def write_file(path: str, content: str) -> str:
    """在 workspace 下写入（覆盖）文本文件。"""
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"已写入 {p.relative_to(BASE)}，共 {len(content)} 字符"


def list_files(path: str = ".") -> str:
    """列出 workspace 下某个目录的内容。"""
    p = _safe_path(path)
    if not p.is_dir():
        return f"不是目录: {path}"
    items = sorted(p.iterdir())
    if not items:
        return "(空目录)"
    return "\n".join(f"{'[D]' if i.is_dir() else '[F]'} {i.name}" for i in items)


def current_time() -> str:
    """获取当前本地时间。"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# 给模型看的工具声明。description 就是提示词，改它等于改行为。
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式。凡是涉及数字运算，都必须用这个工具，不要心算。支持 + - * / // % ** 和括号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的表达式，例如 (3.5)**12 或 123*456",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": "执行一段 Python 代码。适合数据处理、字符串处理、调用标准库。需要多步计算或循环时用它。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "完整可执行的 Python 代码"}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出 workspace 目录下的文件和子目录。不确定有什么文件时先调这个。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对路径，默认为当前目录 .",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取 workspace 下的文本文件内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对路径，例如 todo.md"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "把内容写入 workspace 下的文本文件（会覆盖同名文件）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对路径，例如 notes/todo.md"},
                    "content": {"type": "string", "description": "要写入的完整内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "获取当前日期和时间。涉及“今天”“现在”的问题必须先调用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

FUNCS = {
    "calculator": calculator,
    "python_exec": python_exec,
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "current_time": current_time,
}
