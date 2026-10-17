# PHASE-01：字节级 BPE 中文分词器

> 阶段目标：从零实现并训练一个字节级 BPE 分词器，支持中文/英文/emoji/生僻字，产出 `data/tokenizer/tokenizer.json`，为后续所有阶段提供 token 化能力。

## 设计

- **字节级合并**：文本 → UTF-8 字节序列，在字节层面统计相邻对并合并。优点：无未登录字、无需词表外的 OOV 处理、与 GPT 系列做法一致。
- **ID 布局**（可序列化、可复现）：
  - `[0, n_special)`：特殊 token（`<s> </s> <pad> <unk> <|user|> <|assistant|> <|image|> <|tool|>`）
  - `[n_special, n_special+256)`：原始字节
  - `[n_special+256, vocab)`：按训练顺序递增的合并结果
- **训练算法**：优先队列（按频率）+ 双向链表（维护邻接关系），每轮取全局最高频对合并，只更新受影响邻域，复杂度 ≈ O(语料规模)；比"每轮全量扫描"快几个数量级，CPU 可训。
- **编码**：严格按 merge 的 rank（训练顺序）反复合并，结果确定；**解码**：每个 token 记录其字节展开，无损回环。

## 数据

- `scripts/download_data.py`：从 hf-mirror 下载小型公开数据集
  - SFT：`shibing624/alpaca-zh`（48,818 条中文指令，研究用途）→ `data/datasets/sft_train.jsonl`
  - DPO：`liyucheng/zhihu_rlhf_3k`（3,000 条中文偏好对）→ `data/datasets/dpo_pairs.jsonl`
  - 预训练语料：内置 demo + alpaca 文本 → `data/corpus/pretrain.txt`
  - 下载失败自动回退到内置 `data/demo/demo_corpus.txt`（离线可用）
- 分词器训练语料默认截断 `max_tokens=1_000_000`（提速；推理不受影响）。

## 验证

```bash
.venv/bin/python scripts/download_data.py --dataset all --max-samples 2000
.venv/bin/python scripts/train_tokenizer.py --config configs/tokenizer.yaml
.venv/bin/python -m pytest tests/test_tokenizer.py -q
```

验收标准（全部通过）：
- 中英混排/emoji/生僻字 encode→decode 无损回环；
- 特殊 token id 稳定、原样解码；
- 词表规模封顶正确（`vocab_size_actual == 目标`）；
- save/load 后行为一致；
- 训练确定性（相同语料 → 相同 merges）。

## 产物

- `data/tokenizer/tokenizer.json`（词表 + 合并规则 + 特殊 token）
- `data/tokenizer/tokenizer_meta.json`（语料统计/词表统计）

## 下一步

P2：GPT decoder-only（RoPE/SwiGLU/RMSNorm/权重共享）+ 预训练循环，使用本阶段分词器。
