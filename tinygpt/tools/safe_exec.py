"""工具函数安全执行器：OpenAI 风格 tools 解析 + AST 白名单执行。

安全模型：
1. 只允许白名单 AST 节点类型（无 import / 无 attribute 访问 / 无 comprehension / 无 subscript 复杂访问）。
2. 只允许调用白名单函数（数学与字符串助手），内置 name 空间与 __builtins__ 隔离。
3. 在子进程中执行并设置超时，杜绝死循环/恶意构造拖垮主进程。
"""
from __future__ import annotations

import ast
import json
import multiprocessing
import re
from typing import Any, Callable, Dict, Optional


# ---- 白名单函数 ----
def add(a, b): return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def divide(a, b): return a / b
def floor_div(a, b): return a // b
def modulo(a, b): return a % b
def power(a, b): return a ** b
def concat(a, b): return str(a) + str(b)
def upper(s): return str(s).upper()
def lower(s): return str(s).lower()
def strip(s): return str(s).strip()
def replace(s, old, new): return str(s).replace(str(old), str(new))
def split(s, sep=" "): return str(s).split(str(sep))
def join(sep, items): return str(sep).join(str(x) for x in items)
def length(x): return len(x)
def sum_(x): return sum(x)
def abs_(x): return abs(x)
def min_(x): return min(x)
def max_(x): return max(x)
def round_(x, ndigits=0): return round(x, ndigits)
def int_(x): return int(x)
def float_(x): return float(x)
def str_(x): return str(x)
def contains(s, sub): return str(sub) in str(s)
def startswith(s, sub): return str(s).startswith(str(sub))
def endswith(s, sub): return str(s).endswith(str(sub))

WHITELIST_FUNCS: Dict[str, Callable] = {
    "add": add, "subtract": subtract, "multiply": multiply, "divide": divide,
    "floor_div": floor_div, "modulo": modulo, "power": power,
    "concat": concat, "upper": upper, "lower": lower, "strip": strip,
    "replace": replace, "split": split, "join": join,
    "length": length, "sum": sum_, "abs": abs_, "min": min_, "max": max_,
    "round": round_, "int": int_, "float": float_, "str": str_,
    "contains": contains, "startswith": startswith, "endswith": endswith,
}

ALLOWED_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Call, ast.Tuple, ast.List, ast.Dict, ast.Set,
    ast.keyword,
)
DENIED_MESSAGE = "该表达式使用了不允许的语法（如 import / 属性访问 / 列表推导），已拒绝执行。"


class SafeExecError(Exception):
    pass


def validate_ast(code: str) -> ast.AST:
    """AST 白名单校验：不合法直接抛 SafeExecError。"""
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as e:
        raise SafeExecError(f"语法错误: {e}")
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise SafeExecError(DENIED_MESSAGE)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise SafeExecError(DENIED_MESSAGE)      # 禁止 x.f() 属性调用
            if node.func.id not in WHITELIST_FUNCS:
                raise SafeExecError(f"函数 {node.func.id} 不在白名单内")
    return tree


def _run_in_process(code: str, funcs: Dict[str, Callable], q: Any) -> None:
    try:
        tree = validate_ast(code)
        globs = {"__builtins__": {}, **{k: v for k, v in funcs.items()}}
        result = eval(compile(tree, "<safe_exec>", "eval"), globs, {})
        q.put(("ok", result))
    except SafeExecError as e:
        q.put(("error", str(e)))
    except Exception as e:                                   # noqa: BLE001
        q.put(("error", f"执行错误: {type(e).__name__}: {e}"))


def safe_exec(code: str, timeout: float = 3.0,
              funcs: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
    """执行白名单表达式，返回 {'ok': bool, 'result'|'error', 'time_ms'}。"""
    funcs = funcs or WHITELIST_FUNCS
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run_in_process, args=(code, funcs, q), daemon=True)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"ok": False, "error": f"执行超时（>{timeout}s），已终止", "time_ms": -1}
    try:
        status, payload = q.get_nowait()
    except Exception:                                      # noqa: BLE001
        return {"ok": False, "error": "执行器异常退出", "time_ms": -1}
    if status == "ok":
        return {"ok": True, "result": payload}
    return {"ok": False, "error": payload}


# ---- OpenAI 风格工具调用解析 ----
_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_GREEDY_RE = re.compile(r"\{.*\}", re.DOTALL)
_NONGREEDY_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _coerce_args(args) -> dict:
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return args if isinstance(args, dict) else {}


def _try_dict(raw: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and "name" in data:
        return {"name": data["name"], "arguments": _coerce_args(data.get("arguments", {}))}
    return None


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出文本中提取工具调用 JSON。

    依次尝试：整体 JSON → ```json 代码块 → 文本中贪心/非贪心 JSON 片段。
    返回 {"name": str, "arguments": dict}；找不到返回 None。
    """
    t = text.strip()
    if t.startswith("{"):
        hit = _try_dict(t)
        if hit:
            return hit
    for m in _BLOCK_RE.finditer(text):
        hit = _try_dict(m.group(1))
        if hit:
            return hit
    for pattern in (_GREEDY_RE, _NONGREEDY_RE):
        for m in pattern.finditer(text):
            hit = _try_dict(m.group(0))
            if hit:
                return hit
    return None


def run_tool(name: str, arguments: dict, timeout: float = 3.0) -> Dict[str, Any]:
    """根据注册表执行工具调用；注册表含自定义工具（如 search_kb）。"""
    global _EXTRA_TOOLS
    func = _EXTRA_TOOLS.get(name) or WHITELIST_FUNCS.get(name)
    if func is None:
        return {"ok": False, "error": f"未知工具: {name}"}
    if callable(func):
        try:
            result = func(**arguments)
            return {"ok": True, "result": result}
        except Exception as e:                             # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": False, "error": f"工具 {name} 不可调用"}


_EXTRA_TOOLS: Dict[str, Any] = {}


def register_tool(name: str, func: Callable) -> None:
    """注册自定义工具（如 RAG 的 search_kb），供 run_tool 使用。"""
    _EXTRA_TOOLS[name] = func


def list_tools() -> list:
    """OpenAI 风格 tools 描述（供 /v1/chat/completions tools 参数与 /v1/agent 使用）。"""
    out = []
    for name in sorted(WHITELIST_FUNCS):
        out.append({"type": "function", "function": {"name": name, "description": "", "parameters": {"type": "object", "properties": {}}}})
    for name in sorted(_EXTRA_TOOLS):
        out.append({"type": "function", "function": {"name": name, "description": "", "parameters": {"type": "object", "properties": {}}}})
    return out
