"""安全执行器与工具调用单元测试。"""
from __future__ import annotations

import pytest

from tinygpt.tools.safe_exec import (parse_tool_call, register_tool, run_tool,
                                     safe_exec)


def test_safe_exec_arithmetic():
    r = safe_exec("add(multiply(2, 3), 4)")
    assert r["ok"] and r["result"] == 10


def test_safe_exec_string_helpers():
    r = safe_exec("upper(concat('ab', 'cd'))")
    assert r["ok"] and r["result"] == "ABCD"


def test_safe_exec_denies_import():
    r = safe_exec("__import__('os').system('echo hi')")
    assert not r["ok"]
    assert "不允许" in r["error"] or "白名单" in r["error"]


def test_safe_exec_denies_attributes():
    r = safe_exec("'abc'.upper()")
    assert not r["ok"]           # 属性访问被禁
    r2 = safe_exec("open('/etc/passwd')")
    assert not r2["ok"]


def test_safe_exec_denies_comprehension_and_lambda():
    assert not safe_exec("[x for x in range(3)]")["ok"]
    assert not safe_exec("(lambda x: x)(1)")["ok"]


def test_safe_exec_timeout():
    # spawn 子进程有固定启动开销，超时给足余量验证正常路径
    r = safe_exec("add(1, 1)", timeout=5.0)
    assert r["ok"] and r["result"] == 2


def test_parse_tool_call_json_block():
    text = "让我算一下：\n```json\n{\"name\": \"add\", \"arguments\": {\"a\": 1, \"b\": 2}}\n```"
    call = parse_tool_call(text)
    assert call == {"name": "add", "arguments": {"a": 1, "b": 2}}


def test_parse_tool_call_bare():
    call = parse_tool_call('{"name": "upper", "arguments": "{\\"s\\": \\"hi\\"}"}')
    assert call["name"] == "upper"
    assert call["arguments"] == {"s": "hi"}


def test_parse_tool_call_none():
    assert parse_tool_call("直接回答就好") is None


def test_run_tool_whitelist():
    r = run_tool("multiply", {"a": 6, "b": 7})
    assert r["ok"] and r["result"] == 42


def test_run_tool_unknown():
    r = run_tool("rm_rf", {})
    assert not r["ok"]


def test_register_tool():
    register_tool("double", lambda x: x * 2)
    r = run_tool("double", {"x": 21})
    assert r["ok"] and r["result"] == 42
