"""demo MCP Server：stdio JSON-RPC 2.0，供 MCPClient 测试与演示。

按 MCP 协议实现 initialize → tools/list → tools/call 最小子集。
运行：python scripts/demo_mcp_server.py
"""
from __future__ import annotations

import json
import sys
import time


def _echo(args): return args
def _add(args): return {"sum": args.get("a", 0) + args.get("b", 0)}
def _now(args): return {"time": time.strftime("%Y-%m-%d %H:%M:%S")}

TOOLS = {
    "echo": _echo,
    "add": _add,
    "get_time": _now,
}
TOOL_DESCS = [
    {"name": "echo", "description": "原样返回输入", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "add", "description": "两个整数相加", "inputSchema": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}},
    {"name": "get_time", "description": "返回服务器当前时间", "inputSchema": {"type": "object", "properties": {}}},
]


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = msg.get("method")
        params = msg.get("params", {})
        rid = msg.get("id")
        if method == "initialize":
            out = {"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "demo-mcp-server", "version": "0.1.0"},
            }}
        elif method == "notifications/initialized":
            continue                                    # 单程通知，无响应
        elif method == "tools/list":
            out = {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOL_DESCS}}
        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments", {})
            if name in TOOLS:
                result = TOOLS[name](args)
                out = {"jsonrpc": "2.0", "id": rid, "result": {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}}
            else:
                out = {"jsonrpc": "2.0", "id": rid, "result": {
                    "isError": True, "content": [{"type": "text", "text": f"unknown tool: {name}"}]}}
        elif method == "shutdown":
            out = {"jsonrpc": "2.0", "id": rid, "result": {}}
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()
            break
        else:
            out = {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {method}"}}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
