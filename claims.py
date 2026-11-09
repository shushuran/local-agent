"""声明核验：模型说「已写入/已读取」时，用工具记录和文件系统对一遍。

只做保守检查，避免把正常表述误判成撒谎：
- 只看含文件名的句子；
- 只核验 workspace 内的路径；
- 写入必须来自成功的 write_file / edit_file，或本轮 python_exec 确实改动了该文件。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

PATH_RE = re.compile(
    r"((?:\.\.?[\\/])?(?:[A-Za-z0-9_\u4e00-\u9fff.\-]+[\\/])*"
    r"[A-Za-z0-9_\u4e00-\u9fff.\-]+\.(?:md|txt|py|json|csv|log|ya?ml|html|pdf))",
    re.IGNORECASE,
)

WRITE_VERBS = (
    "已写入",
    "写入",
    "已写",
    "已保存",
    "保存",
    "已创建",
    "创建",
    "已生成",
    "生成",
    "写进",
    "写到了",
)
READ_VERBS = ("已读取", "读取", "读出来", "读了", "已读")


def _norm_rel(path: str) -> str:
    value = path.replace("\\", "/").strip().strip("`'\"")
    while value.startswith("./"):
        value = value[2:]
    return value


def _inside_workspace(rel: str, workspace: Path) -> Path | None:
    try:
        target = (workspace / rel).resolve()
    except (OSError, ValueError):
        return None
    if target == workspace or workspace in target.parents:
        return target
    return None


def _same_path(candidate: str, rel: str) -> bool:
    return os.path.normcase(_norm_rel(candidate)) == os.path.normcase(rel)


def _has_write(
    records: Iterable[Any], rel: str, target: Path, turn_started: float
) -> bool:
    for record in records:
        if not getattr(record, "ok", False):
            continue
        name = getattr(record, "name", "")
        output = str(getattr(record, "output", "") or "")
        args = getattr(record, "args", {}) or {}
        if name == "write_file" and output.startswith("已写入"):
            if _same_path(str(args.get("path", "")), rel):
                return True
        if name == "edit_file" and output.startswith("已更新"):
            if _same_path(str(args.get("path", "")), rel):
                return True
        if name == "python_exec":
            try:
                if target.exists() and target.stat().st_mtime >= turn_started - 2:
                    return True
            except OSError:
                continue
    return False


def _has_read(records: Iterable[Any], rel: str) -> bool:
    for record in records:
        if not getattr(record, "ok", False):
            continue
        name = getattr(record, "name", "")
        args = getattr(record, "args", {}) or {}
        output = str(getattr(record, "output", "") or "")
        if name == "read_file" and "文件不存在" not in output:
            if _same_path(str(args.get("path", "")), rel):
                return True
        if name == "python_exec" and output:
            return True
    return False


def verify_claims(
    text: str,
    records: Iterable[Any],
    workspace: Path,
    turn_started: float,
) -> list[str]:
    if not text:
        return []
    workspace = Path(workspace).resolve()
    issues: list[str] = []
    sentences = [s for s in re.split(r"[。！？!?\n；;]", text) if s.strip()]
    for sentence in sentences:
        rels = {_norm_rel(p) for p in PATH_RE.findall(sentence)}
        if not rels:
            continue
        write_claim = any(verb in sentence for verb in WRITE_VERBS)
        read_claim = any(verb in sentence for verb in READ_VERBS)
        if not write_claim and not read_claim:
            continue
        for rel in rels:
            target = _inside_workspace(rel, workspace)
            if target is None:
                continue
            if write_claim:
                if not target.exists():
                    issues.append(f"回答声称写入 {rel}，但该文件不存在")
                elif not _has_write(records, rel, target, turn_started):
                    issues.append(f"回答声称写入 {rel}，但本轮没有成功的写入调用")
            elif read_claim and not _has_read(records, rel):
                issues.append(f"回答声称读取 {rel}，但本轮没有成功的读取调用")
    return list(dict.fromkeys(issues))
