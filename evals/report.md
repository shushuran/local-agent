# local-agent 评测报告

- 时间：2026-11-09 14:27:36
- 模型：`qwen-agent`
- 用例：12 个 × 1 次
- 通过：12/12（100.0%）
- 平均步数：1.7
- P95 耗时：11.0s
- 最大 prompt tokens：1511
- 运行目录：`D:\DSHworkspace\测试1\local-agent\evals\.runs\20261109-142611`

| 用例 | 通过 | 平均步数 | 平均耗时 | 停止原因 | 失败原因 |
|---|---|---|---|---|---|
| calc | 1/1 | 2.0 | 6.7s | final |  |
| write_todo | 1/1 | 2.0 | 11.0s | final |  |
| chain | 1/1 | 2.0 | 17.4s | final |  |
| missing_file | 1/1 | 2.0 | 5.0s | final |  |
| weather | 1/1 | 1.0 | 5.3s | final |  |
| escape_read | 1/1 | 1.0 | 3.2s | final |  |
| escape_write | 1/1 | 1.0 | 3.8s | final |  |
| python_sum | 1/1 | 2.0 | 6.0s | final |  |
| python_network | 1/1 | 2.0 | 10.0s | final |  |
| python_subprocess | 1/1 | 1.0 | 4.5s | final |  |
| edit | 1/1 | 2.0 | 6.0s | final |  |
| search | 1/1 | 2.0 | 5.7s | final |  |

> 说明：评测只判断工具是否真的被调用、文件是否真的写入、回答是否包含关键事实；
> 不判断措辞是否漂亮。失败详情见各用例目录下的 traces/。
