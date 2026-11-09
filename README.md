# local-agent：最小本地 Agent（Ollama + function calling）

不依赖任何 Agent 框架，整个循环 40 行，目的是让你看清「模型每一轮到底收到了什么」。

## 目录

```
local-agent/
├── agent.py        # 主循环 + CLI
├── tools.py        # 6 个工具 + 给模型看的 JSON Schema
├── Modelfile       # 固定 num_ctx=8192 的模型配置
└── workspace/      # Agent 唯一能读写的目录（自动创建）
```

## 快速开始

```powershell
# 1. 安装 Ollama（已装跳过）
winget install -e --id Ollama.Ollama
# 安装后重开一个终端，确认：
ollama --version

# 2. 拉模型（约 4.7G）
ollama pull qwen2.5:7b-instruct

# 3. 在本目录创建带 8k 上下文的模型
ollama create qwen-agent -f Modelfile
ollama run qwen-agent "你好"      # 先确认能跑

# 4. Python 环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install openai

# 5. 启动
python agent.py
```

## 验收任务（按顺序做）

| # | 输入 | 期望 |
|---|---|---|
| 1 | `3.5 的 12 次方是多少` | 调 `calculator` |
| 2 | `在 workspace 下建 todo.md，写三条今天要做的事` | 调 `write_file` |
| 3 | `先算 123*456，把结果写进 result.txt，再读出来确认` | 连续 3 次工具调用 |
| 4 | `读一下 not_exist.txt` | 明确说不存在，不编造 |
| 5 | `帮我查一下今天北京天气` | 没有搜索工具，看它是承认做不到还是瞎编 |

第 4、5 条是关键：它们测的不是代码，是模型行为。

## 跑起来后重点观察

1. **上下文膨胀**：每轮工具结果都进 messages，右下角会打印消息数。聊十轮以上就会顶到 8192。
2. **工具描述的影响**：把 `calculator` 的 description 改模糊，再跑任务 1，看调用准确率怎么变。
3. **死循环**：同一工具连续调用、参数不变——这是小模型最常见的故障。
4. **假装完成**：没调 `write_file` 却说“已经写好了”。
5. **语言差异**：同样任务用英文问一遍，成功率往往更高。

## 常见问题

**`ollama create` 报错 model not found**
先 `ollama pull qwen2.5:7b-instruct`，Modelfile 的 FROM 必须和本地已有名字完全一致。

**模型不调工具，只回一段文字**
Ollama 版本太老（需 ≥ 0.4）。升级后 `ollama rm qwen-agent` 重建。
也可能是模型不支持 tools，换 `qwen2.5:7b-instruct` 或 `llama3.1:8b`。

**响应很慢 / 显存爆了**
`ollama ps` 看是不是 100% GPU。出现 CPU offload 会慢 5–10 倍。
12G 显存跑 7B q4 + 8k 上下文大约占 6–7G；如果只有 6G 显存，把 Modelfile 改成
`FROM qwen2.5:3b-instruct` + `num_ctx 4096`。

**`tool_call_id` 相关报错**
个别 Ollama 版本返回的 id 为空。代码里已用 `call_{step}_{i}` 兜底；
若仍报错，改用原生接口 `POST /api/chat`（tools 格式相同，arguments 是对象而非字符串）。

**中文路径乱码**
workspace 路径不要带中文和空格。

## 下一步练习（按难度）

1. 加一个 `web_search` 工具（Tavily 或博查 API，国内可直连）。
2. 实现滑动窗口：消息超过 20 条时丢弃最早的一批（注意别把 tool 消息和它的 assistant 拆开）。
3. 把 `MAX_STEPS` 改成动态的：连续两次相同工具+相同参数就中断。
4. 加一个 `todo` 工具，让 Agent 自己维护任务列表——这是 planning 的雏形。
5. 换成 LangGraph 重写，对比两者代码量和你对状态流转的掌控感。
