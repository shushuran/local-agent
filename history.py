"""会话历史：一行一条消息存 JSONL，支持恢复和归档。

文件格式：
    {"type": "meta", "session": "...", "model": "...", ...}
    {"type": "message", "message": {"role": "user", "content": "..."}}
    ...

不依赖数据库；一行坏了只跳过那一行，不拖垮整个会话。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

_SESSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


class HistoryStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    # -- 路径 ---------------------------------------------------------------
    def _path(self, session_id: str) -> Path:
        if not _SESSION_RE.match(session_id):
            raise ValueError(f"非法会话 id: {session_id!r}")
        return self.root / f"{session_id}.jsonl"

    def exists(self, session_id: str) -> bool:
        try:
            return self._path(session_id).is_file()
        except ValueError:
            return False

    # -- 写 -----------------------------------------------------------------
    def new_session(self, meta: dict[str, Any] | None = None) -> str:
        session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        record = {
            "type": "meta",
            "session": session_id,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        record.update(meta or {})
        with self._path(session_id).open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return session_id

    def append(self, session_id: str, message: dict[str, Any]) -> None:
        path = self._path(session_id)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"type": "message", "message": message}, ensure_ascii=False)
                + "\n"
            )

    # -- 读 -----------------------------------------------------------------
    def load(self, session_id: str) -> list[dict[str, Any]]:
        path = self._path(session_id)
        if not path.is_file():
            raise FileNotFoundError(f"会话不存在: {session_id}")
        messages: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("type") == "message" and isinstance(
                    record.get("message"), dict
                ):
                    messages.append(record["message"])
        return messages

    def meta(self, session_id: str) -> dict[str, Any] | None:
        path = self._path(session_id)
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("type") == "meta":
                    return record
        return None

    def list_sessions(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.jsonl"), reverse=True):
            session_id = path.stem
            meta = self.meta(session_id) or {"session": session_id}
            count = 0
            try:
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    for _ in fh:
                        count += 1
            except OSError:
                continue
            meta = dict(meta)
            meta["lines"] = count
            meta["path"] = str(path)
            items.append(meta)
        return items

    def latest(self) -> str | None:
        sessions = self.list_sessions()
        return sessions[0]["session"] if sessions else None

    def archive(self, session_id: str) -> Path | None:
        path = self._path(session_id)
        if not path.is_file():
            return None
        archive_dir = self.root / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        target = archive_dir / f"{session_id}.jsonl"
        path.replace(target)
        return target
