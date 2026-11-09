"""结构化追踪：每个事件一行 JSONL，方便事后回放和统计。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, path: Path | None, meta: dict[str, Any] | None = None) -> None:
        self.path = path
        self._fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")
            self.event("run_meta", **(meta or {}))

    def event(self, kind: str, **data: Any) -> None:
        if self._fh is None:
            return
        record = {"ts": round(time.time(), 4), "event": kind}
        record.update(data)
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
