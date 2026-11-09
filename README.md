# local-agent v2

一个不依赖 Agent 框架的本地智能体：Ollama + 原生 function calling + 手写循环。
v1 只够演示「模型会调工具」；v2 补的是让它在真实使用中**不卡死、不失忆、不越界、能被评测**的工程部分。

## v2 相对 v1 做了什么

| 能力 | v1 | v2 |
|---|---|---|
| 循环 | 全部塞在 `agent.py` | `runtime.py` 里的 `AgentRuntime`，可测试 |
| 停止条件 | 只有 `MAX_STEPS` | 步数、重复调用、token 预算、单轮耗时、空回答 |
| 失败处理 | 异常打印后继续 | API 退避重试、工具错误回灌、连续失败提示 |
| 历史 | 进程退出即丢 | `history/*.jsonl`，支持 `--resume` |
| 上下文 | 只增不减 | token 预算、按轮裁剪、滚动摘要 |
| 工具 | 6 个函数 + 手写 schema | 注册表 + 参数校验 + 权限/超时/人工确认 |
| 代码执行 | 直接起子进程 | `python_exec` 软沙箱（写/网/子进程拦截） |
| 完成度 | 模型说完成就算 | 声明核验：说写了就查文件，对不上要求重做 |
| 规划 | 无 | `todo` 工具 + `workspace/.plan.json` |
| 观测 | 屏幕输出 | `traces/*.jsonl` 每步记录 |
| 评测 | 手动看 | `evals/` 12 条用例 + 报告 |

## 目录

```
local-agent/
├── agent.py            # CLI：参数解析、REPL、打印
├── runtime.py          # 核心循环、停止条件、重试、声明核验
├── config.py           # 配置（CLI / 环境变量 / 默认值）
├── context.py          # token 估算、按轮裁剪、滚动摘要
├── history.py          # JSONL 会话持久化
├── claims.py           # 「已写入/已读取」的声明核验
├── trace.py            # JSONL 追踪
├── tools.py            # 工具函数 + 注册表 + 参数校验 + 审批
├── sandbox_child.py    # python_exec 子进程的审计钩子沙箱
├── Modelfile           # qwen-agent = qwen2.5:7b + 8192 上下文
├── workspace/          # Agent 唯一可写目录
├── history/            # 会话记录（运行后生成）
├── traces/             # 每步追踪（运行后生成）
├── tests/              # 39 个 unittest
├── evals/              # 12 条脚本化评测 + report.md
└── _backup_v1/         # 改造前的四个文件，方便对照
```

## 快速开始

先确认用的是 pytorch 环境（Code Runner 已在 `.vscode/settings.json` 里固定）：

```powershell
cd D:\DSHworkspace\测试1\local-agent
python agent.py
```

常用命令：

```powershell
python agent.py                                  # 交互
python agent.py --ask "3.5 的 12 次方是多少"     # 单轮
python agent.py --resume latest                  # 恢复最近会话
python agent.py --list-sessions                  # 看历史会话
python agent.py --status                         # 检查模型服务与可用模型
python agent.py --react                          # 显式 Thought 模式
python agent.py --yes                            # 自动确认覆盖写
```

交互里支持 `/reset`、`/session`、`/status`、`/help`、`exit`。

> **Code Runner 注意**：它默认把输出放在只读的 Output 面板里，`input()` 等不到输入，看起来就像 `/help` 没反应。本项目已设置 `code-runner.runInTerminal: true`，Run Code 会在集成终端里运行，输入才可用。如果还停在旧进程，先按 `Ctrl+Alt+M` 停止，或重载窗口。

模型侧依赖 Ollama 里的 `qwen-agent`（由 `Modelfile` 创建，和 `qwen2.5:7b` 共用权重）：

```powershell
ollama list
# 若没有 qwen-agent：
ollama create qwen-agent -f Modelfile
```

## 一次对话发生了什么

```
用户输入
  -> ContextManager.build()       # 超预算就先裁剪 + 摘要，保持 tool_call 配对
  -> chat.completions.create()    # 带 9 个工具的 schema
  -> 有 tool_calls?
      是 -> 参数校验 -> 权限/审批 -> 执行 -> 结果作为 role=tool 回灌
             -> 重复检测 / 连续失败提示 / 单轮耗时检查
             -> 回到模型
      否 -> 声明核验 -> 最终回答
```

停止原因（`TurnResult.stop_reason`）：

| 值 | 含义 |
|---|---|
| `final` | 正常结束 |
| `max_steps` | 达到步数上限 |
| `repeated_call` | 同一工具 + 同一参数重复出现 |
| `time_budget` | 单轮超时 |
| `context_limit` | prompt tokens 超过硬上限 |
| `empty_answer` | 连续空回答 |
| `api_error` | 模型服务不可用、模型不存在等 |

## 配置

命令行优先，其次环境变量，最后默认值。

| CLI | 环境变量 | 默认 |
|---|---|---|
| `--model` | `AGENT_MODEL` | `qwen-agent` |
| `--base-url` | `AGENT_BASE_URL` | `http://127.0.0.1:11434/v1` |
| `--max-steps` | `AGENT_MAX_STEPS` | `8` |
| `--temperature` | `AGENT_TEMPERATURE` | `0.2` |
| `--context-tokens` | `AGENT_CONTEXT_TOKENS` | `6200` |
| `--workspace` | — | `./workspace` |
| `--react` | — | 关闭 |
| `--yes` | — | 关闭（覆盖写会询问） |
| `--no-history` | — | 关闭（默认写 history） |
| `--no-trace` | — | 关闭（默认写 traces） |

## 安全模型

**路径限制**：`read_file` / `write_file` / `edit_file` / `list_files` / `search_files` 都走 `_safe_path()`，`resolve()` 之后必须仍在 `workspace/` 内，符号链接逃逸也会被拦。

**python_exec 软沙箱**：子进程安装 `sys.addaudithook`，拦截：

- workspace 之外的写入（相对路径、绝对路径都算）
- workspace / Python 安装目录 / 系统目录之外的读取
- socket 连接、DNS
- subprocess / os.system / exec / spawn
- `os.remove`、`os.rename`、`os.mkdir` 等写操作

它挡的是**模型误操作**，不是恶意攻击。刻意用底层手段（例如 `ctypes`）可以绕过；numpy/pandas 导入也依赖 ctypes，所以没有封死。见 [sandbox_child.py](sandbox_child.py)。

**人工确认**：覆盖已有文件时，CLI 会停下来问 `[y/N]`；`--yes` 或评测里的 `auto_approve` 可跳过。新建文件不需要确认。

**声明核验**：回答里出现「已写入/已创建/已读取 + 文件名」时，`claims.py` 会对照本轮工具记录和文件系统；对不上就插入系统提示要求模型核实或补做，最多一次。仍对不上时在 CLI 打印 `[核验警告]`，并在 `TurnResult.warnings` 里保留。

## 记忆与上下文

- 每条消息（含 tool 结果）追加写入 `history/<session>.jsonl`，退出后可以 `--resume <id>` 继续。
- `--resume latest` 取最近一个会话；`/reset` 把当前会话移到 `history/archive/` 再开新会话。
- token 估算按中文 1.2 字/token、其他 3.6 字符/token；预算默认 6200，给 8192 的窗口留出工具 schema 和输出空间。
- 超预算时按 **user 消息边界**裁剪，保留最后一轮；被裁掉的部分交给模型生成滚动摘要，失败则退化为机械摘要。原始消息始终在 history 里，不会真的丢。

## 评测

```powershell
python evals/run_evals.py                    # 12 条用例
python evals/run_evals.py --repeats 2        # 每条跑两次
python evals/run_evals.py --only calc,chain  # 只跑指定用例
python evals/run_evals.py --verbose          # 打印每步工具调用
```

当前基线（`qwen-agent`，2026-11-09）：

- **12/12 通过**
- 平均 1.7 步，P95 11.0s
- 最大 prompt tokens 1511
- 报告：[evals/report.md](evals/report.md)

用例覆盖：计算、写文件、多步串联、不存在文件、无工具任务、路径越界读写、python 正常执行、网络拦截、子进程拦截、编辑、搜索。失败详情看 `evals/.runs/<run>/<case>/traces/`。

## 单元测试

```powershell
python -m unittest discover -s tests -t . -v
```

39 个测试，覆盖上下文裁剪与配对、历史损坏行、声明核验、路径逃逸、参数校验、覆盖审批、软沙箱、重复调用、空回答、claim 重试、超时/上下文上限、会话恢复与中断修复。

## 常见问题

**浏览器打开 `http://127.0.0.1:11434/v1` 显示 404**

这是正常的。`/v1` 是 OpenAI 兼容接口的根路径，不是网页；Ollama 只在这个前缀下提供 `/v1/models`、`/v1/chat/completions` 等接口。验证服务：

```powershell
python agent.py --status
# 或
curl http://127.0.0.1:11434/v1/models
curl http://127.0.0.1:11434/api/version
```

- 返回模型列表：服务正常。
- 连接被拒绝：Ollama 没启动。
- 返回 `model 'xxx' not found`：模型不存在，`ollama list` 里换一个，或改 `--model`。

**Code Runner 里输入 `/help` 没有任何反应**

见上面的 Code Runner 说明：Output 面板不支持 stdin。用集成终端或 `--ask`；本项目已开启 `runInTerminal`，重载窗口后重新 Run Code。

**回答很慢或一直转圈**

7B 模型在 CPU 上会很慢；`ollama ps` 看是不是 100% GPU。单轮默认限时 180 秒，可用 `--max-steps` 和上下文预算一起调。

## 已知限制

1. **7B 模型本身不稳定**：会过度使用 `todo`、偶尔编造内容。运行时能拦住一部分（声明核验、路径限制），但拦不住所有幻觉；评测通过率只能说明这批用例。
2. **摘要是有损的**：`history/` 保留原文，但模型看到的是摘要；长任务里早期细节可能丢失。
3. **软沙箱不是安全边界**：见上，ctypes 等底层路径可绕过；不要用它执行不可信代码。
4. **单机单进程**：没有并发控制，同一 workspace 不要同时跑两个 Agent。
5. **没有 RAG / 多智能体 / 网页搜索**：这些是下一步的可选扩展，不是当前缺口。

## v1 对照

`_backup_v1/` 保留了改造前的 `agent.py`、`tools.py`、`README.md`、`Modelfile`。想对照最小实现时直接打开；不要删除，它是这个项目最好的「减法」参考。
