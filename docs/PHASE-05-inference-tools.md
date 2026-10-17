# PHASE-05：推理封装与工具函数调用

> 分支：`p5-inference-tools` → 合并入 `main`
> 前置：P2 模型 / P3 权重（对话能力）

## 1. 本阶段目标

1. **对话推理封装**：多轮消息 → `<s><|user|>…<|assistant|>` 模板 → 续写至 `</s>`。
2. **OpenAI 风格 tools 解析**：从模型输出中提取 `{"name": ..., "arguments": {...}}`（支持 ```json 代码块、裸 JSON、嵌套大括号）。
3. **安全执行器（AST 白名单）**：只允许白名单节点与白名单函数，杜绝 `import`/属性访问/列表推导/未知调用；子进程 + 超时防死循环。

## 2. 安全执行器设计（tinygpt/tools/safe_exec.py）

| 防线 | 实现 |
|---|---|
| 节点白名单 | 仅 `Constant/Name/BinOp/Call/Tuple/List/Dict/...` 等基础表达式节点，`walk` 校验 |
| 调用白名单 | 只允许 `add/subtract/multiply/divide/concat/upper/...` 等 24 个纯函数；`x.f()` 属性调用一律拒绝 |
| 命名空间隔离 | `__builtins__={}`，只注入白名单函数 |
| 进程隔离 | `multiprocessing` 子进程执行 + 超时终止，防死循环 |

拒绝示例：`__import__('os').system(...)`、`'abc'.upper()`、`[x for x in range(3)]`、`open(...)`。

## 3. 工具调用解析

- `parse_tool_call(text)`：依次尝试 整体 JSON → ```json 块 → 贪心/非贪心片段；arguments 支持 dict 或 JSON 字符串。
- `run_tool(name, args)`：白名单函数 + `register_tool` 注册的自定义工具（P8 RAG 的 `search_kb` 走此通道）。
- `list_tools()`：生成 OpenAI 风格 `tools` 描述，供 P9 API 使用。

## 4. 运行

```bash
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, '.')
from tinygpt.tools import safe_exec, parse_tool_call
print(safe_exec("multiply(add(2,3), 4)"))          # {'ok': True, 'result': 20}
print(safe_exec("__import__('os')"))               # {'ok': False, ...}
EOF
```

## 5. 验收

- [x] 安全执行：算术/字符串助手通过；import、属性访问、推导式、lambda、未知函数全部拒绝
- [x] 工具解析：代码块 JSON / 裸 JSON / arguments 字符串 / 无调用返回 None
- [x] 对话模板：多轮消息 → ids 回环正确
- [x] 单测 13 passed（P5 新增）
- [x] 分支 `p5-inference-tools` 推送 gitee，合并 `main`

## 6. 下一步

P6：多模态（从零实现微型 ViT + 投影层 + 与 LLM 联合微调）；P7 MCP 客户端；P8 RAG。
