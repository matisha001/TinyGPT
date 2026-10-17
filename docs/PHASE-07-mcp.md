# PHASE-07：MCP 客户端（stdio JSON-RPC）

> 分支：`p7-mcp` → 合并入 `main`

## 1. 本阶段目标

实现通用 MCP（Model Context Protocol）客户端：stdio 传输 + JSON-RPC 2.0，
按 `initialize → tools/list → tools/call` 连接任意 MCP Server。

## 2. 实现（tinygpt/mcp/client.py）

| 组件 | 说明 |
|---|---|
| `MCPClient(cmd, timeout)` | 以子进程拉起任意 server（`python xxx_server.py` 等） |
| `connect()` | `initialize`（协议版本 2024-11-05）→ `notifications/initialized`（单程通知，**不读响应**） |
| `list_tools()` | `tools/list` → 工具描述列表 |
| `call_tool(name, args)` | `tools/call` → 聚合 text 内容，`isError` 时抛 `MCPError` |
| `close()` | `shutdown` → terminate（超时则 kill） |

**踩坑记录**：`notifications/initialized` 是 JSON-RPC 通知（服务端不回响应），
早期实现按请求处理导致客户端 `readline` 永久阻塞——修复为 `notify=True` 写而不读。

## 3. Demo Server（scripts/demo_mcp_server.py）

最小 stdio JSON-RPC 服务端，提供 `echo` / `add` / `get_time` 三个工具，
用于验证客户端协议交互。

## 4. 运行

```bash
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, '.')
from tinygpt.mcp import MCPClient
c = MCPClient(["python", "scripts/demo_mcp_server.py"])
c.connect()
print(c.list_tools())
print(c.call_tool("add", {"a": 20, "b": 22}))
c.close()
EOF
```

## 5. 验收

- [x] initialize / tools/list / tools/call / 通知 全协议走通
- [x] 未知工具返回错误不阻塞；close 正常收尾
- [x] 单测 3 passed（tests/test_mcp.py）
- [x] 分支 `p7-mcp` 推送 gitee，合并 `main`

## 6. 下一步

P8：RAG（字符 bigram + TF-IDF）；P9：OpenAI 兼容 API（含 agent 工具循环，MCP 工具接入）。
