# PHASE-08：RAG 检索增强

> 分支：`p8-rag` → 合并入 `main`

## 1. 本阶段目标

知识库入库 → 字符 bigram + TF-IDF 检索 → 上下文注入；同时暴露为 `search_kb` 工具，
供 P9 的 `/v1/agent/completions` 工具循环使用。

## 2. 实现（tinygpt/rag/kb.py，不依赖 scikit-learn）

| 组件 | 说明 |
|---|---|
| `char_bigrams(text)` | 'abc' → ['ab','bc']；不足两字符退化为单字符 |
| `TFIDFIndex` | 词表 = 语料 bigram 集合；TF = 词频/文档长度；IDF = `log((1+N)/(1+df))+1`；余弦相似度排序 |
| `KnowledgeBase` | 入库 → 构建 → `save/load` JSON 索引 → `build_prompt` 上下文注入 |
| `search_kb` 工具 | 通过 `register_tool` 注册，`run_tool("search_kb", ...)` 可调 |

检索示例（build_kb.py 内置 7 条 demo 文档）：

```
Q: 什么是 LoRA？
  [0.198] LoRA 低秩微调通过在注意力与 FFN 的 q/k/v/o/gate/up/down 上注入低秩适配器...
```

## 3. 运行

```bash
.venv/bin/python scripts/build_kb.py
```

## 4. 验收

- [x] bigram 切分正确（含中文与单字符退化）
- [x] TF-IDF 相关性排序正确（豆包/上海/测试文档分别命中）
- [x] 索引 save/load 无损
- [x] `search_kb` 已注册为工具且可经 run_tool 调用
- [x] 单测 5 passed（tests/test_rag.py）；分支 `p8-rag` 推送 gitee，合并 `main`

## 5. 下一步

P9：OpenAI 兼容 API（chat/completions 流式+多模态+工具、agent 工具循环、rag 端点）。
