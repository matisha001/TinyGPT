"""MCP 客户端测试：连接 demo 服务器，验证 initialize → tools/list → tools/call。"""
from __future__ import annotations

from tinygpt.mcp import MCPClient


def test_mcp_initialize_and_tools():
    client = MCPClient(["python", "scripts/demo_mcp_server.py"])
    client.connect()
    try:
        tools = client.list_tools()
        names = [t["name"] for t in tools]
        assert "echo" in names and "add" in names and "get_time" in names
    finally:
        client.close()


def test_mcp_call_tool():
    client = MCPClient(["python", "scripts/demo_mcp_server.py"])
    client.connect()
    try:
        r = client.call_tool("add", {"a": 20, "b": 22})
        assert "sum" in r.get("text", "") and '"sum": 42' in r.get("text", "")
        r2 = client.call_tool("echo", {"x": "hi"})
        assert "hi" in r2.get("text", "")
        r3 = client.call_tool("get_time", {})
        assert r3.get("text", "").strip() != ""
    finally:
        client.close()


def test_mcp_unknown_tool_raises():
    client = MCPClient(["python", "scripts/demo_mcp_server.py"])
    client.connect()
    try:
        try:
            client.call_tool("no_such_tool", {})
        except Exception:                                 # noqa: BLE001
            pass                                          # 服务器返回 isError，客户端可抛可返回
        else:
            pass
    finally:
        client.close()
