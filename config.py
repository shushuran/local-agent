"""运行配置。

优先级：命令行参数 > 环境变量 > 默认值。
所有路径都基于本文件所在目录，避免从别处启动时工作区漂移。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class Config:
    # 模型服务
    base_url: str = _env("AGENT_BASE_URL", "http://127.0.0.1:11434/v1")
    api_key: str = _env("AGENT_API_KEY", "ollama")
    model: str = _env("AGENT_MODEL", "qwen-agent")
    temperature: float = _env_float("AGENT_TEMPERATURE", 0.2)
    max_tokens: int = _env_int("AGENT_MAX_TOKENS", 1024)

    # 循环控制
    max_steps: int = _env_int("AGENT_MAX_STEPS", 8)
    max_turn_seconds: float = _env_float("AGENT_MAX_TURN_SECONDS", 180.0)
    repeat_limit: int = _env_int("AGENT_REPEAT_LIMIT", 2)
    api_retries: int = _env_int("AGENT_API_RETRIES", 3)

    # 上下文
    context_tokens: int = _env_int("AGENT_CONTEXT_TOKENS", 6200)
    hard_context_tokens: int = _env_int("AGENT_HARD_CONTEXT_TOKENS", 8192)
    max_tool_output: int = _env_int("AGENT_MAX_TOOL_OUTPUT", 4000)
    summarize: bool = True

    # 交互
    react: bool = False
    auto_approve: bool = False

    # 存储
    workspace: Path = ROOT / "workspace"
    history_dir: Path = ROOT / "history"
    trace_dir: Path = ROOT / "traces"
    persist_history: bool = True
    trace: bool = True

    def ensure_dirs(self) -> None:
        for path in (self.workspace, self.history_dir, self.trace_dir):
            path.mkdir(parents=True, exist_ok=True)
