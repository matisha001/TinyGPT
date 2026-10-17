"""MCP 客户端：stdio JSON-RPC，initialize → tools/list → tools/call。

通过子进程连接任意 MCP Server（stdio 传输），基于 JSON-RPC 2.0。
"""
from __future__ import annotations

import json
import logging
import subprocess
from typing import Any, Dict, List, Optional

log = logging.getLogger("tinygpt.mcp")


class MCPError(Exception):
    pass


class MCPClient:
    """stdio JSON-RPC 客户端。

    usage:
        client = MCPClient(["python", "mcp_server.py"])
        client.connect()
        tools = client.list_tools()
        result = client.call_tool("get_weather", {"city": "上海"})
        client.close()
    """

    def __init__(self, cmd: List[str], timeout: float = 10.0, logger: Optional[logging.Logger] = None):
        self.cmd = list(cmd)
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen] = None
        self._req_id = 0
        self._log = logger or log

    # ---- 生命周期 ----
    def connect(self) -> None:
        self._proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        info = self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "tiny-gpt", "version": "0.1.0"},
        })
        if "serverInfo" not in info:
            raise MCPError(f"initialize 响应缺少 serverInfo: {info}")
        self._rpc("notifications/initialized", {}, notify=True)   # 单程通知：不读响应
        self._log.info("MCP 已连接: %s (%s)",
                       info.get("serverInfo", {}).get("name"),
                       info.get("protocolVersion", ""))

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._rpc("shutdown", {})
            except Exception:                              # noqa: BLE001
                pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    # ---- MCP 方法 ----
    def list_tools(self) -> List[Dict[str, Any]]:
        r = self._rpc("tools/list", {})
        return r.get("tools", [])

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        r = self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
        if "content" not in r:
            if r.get("isError"):
                raise MCPError(f"工具调用出错: {r}")
            raise MCPError(f"tools/call 响应缺少 content: {r}")
        # 聚合 text 内容
        texts = [c.get("text", "") for c in r.get("content", []) if c.get("type") == "text"]
        return {"result": r, "text": "\n".join(texts)}

    # ---- JSON-RPC 内核 ----
    def _rpc(self, method: str, params: Dict[str, Any],
             notify: bool = False) -> Dict[str, Any]:
        if self._proc is None or self._proc.poll() is not None:
            raise MCPError("MCP 进程未启动或已退出")
        self._req_id += 1
        payload = json.dumps({"jsonrpc": "2.0", "id": self._req_id,
                              "method": method, "params": params}) + "\n"
        try:
            self._proc.stdin.write(payload)                # type: ignore[union-attr]
            self._proc.stdin.flush()                       # type: ignore[union-attr]
        except (BrokenPipeError, OSError) as e:
            raise MCPError(f"写入失败（服务端可能已退出）: {e}")
        if notify:                                         # 通知无需响应
            return {}
        line = self._proc.stdout.readline()                # type: ignore[union-attr]
        if not line:
            err = self._proc.stderr.read() if self._proc.stderr else ""   # type: ignore[union-attr]
            raise MCPError(f"MCP 进程无响应: {err[:200]}")
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            raise MCPError(f"非法 JSON-RPC 响应: {line[:200]} ({e})")
        if "error" in msg and msg.get("error"):
            raise MCPError(f"JSON-RPC 错误: {msg['error']}")
        return msg.get("result", {})
