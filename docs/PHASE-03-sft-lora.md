# PHASE-03：SFT 监督微调 + LoRA 低秩微调

> 分支：`p3-sft-lora` → 合并入 `main`
> 前置：P2 预训练权重 `checkpoints/pretrain/base.pt`

## 1. 本阶段目标

1. **SFT**：指令数据（alpaca-zh）→ 对话模板 → **回答段掩码损失** + **打包训练**，全量微调。
2. **LoRA**：q/k/v/o/gate/up/down 七个投影全目标低秩适配；初始 B=0 与基座输出严格一致；
   支持 **合并/不合并**（`lora.merge` 开关）；与全量训练共用 `model.use_lora` 开关切换。
3. 产出：`checkpoints/sft/sft.pt`（全量）与 `checkpoints/lora_sft/adapter.pt`（LoRA adapter，可独立分发）。

## 2. 指令数据与模板（tinygpt/data/sft.py）

- 数据：`data/datasets/sft_train.jsonl`（2000 条 `instruction/input/output`）。
- 模板：`<s><|user|>{instruction}\n{input}<|assistant|>{answer}</s>`
- **回答段掩码**：`loss_mask` 在 prompt 段为 0、回答段（含 `</s>`）为 1；模型只在回答段算 CE。
- **打包**：样本顺序填充进 `seq_len` 的块，块尾补 `<pad>`（掩码 0），无 padding 浪费。

```
样本1: <s><|user|>问题<|assistant|>回答</s>        ← 掩码：000000111111
样本2: <s><|user|>问题<|assistant|>回答</s>
──────────────────────── 打包到同一 seq 块 ─────────────────────────
```

## 3. LoRA 实现（tinygpt/lora/lora.py）

```
y = W·x + (α/r) · B·(A·x)      A∈R^{r×in}, B∈R^{out×r}
初始化：A~N(0,0.02²)，B=0 → 初始输出与基座严格一致（可无缝衔接预训练权重）
```

- `apply_lora(model, cfg)`：按 targets 短名（q/k/v/o/gate/up/down）把 `nn.Linear` 原位替换为 `LoRALinear`。
- `trainable_only_adapters`：冻结基座，只留 A/B 可训练（LoRA 全量微调参数量 ≈ 14 层投影 × 2×r×hidden）。
- `merge_lora`：`W ← W + (B·A)·(α/r)`，A/B 置零避免 wrapper 双重计数；`unmerge_lora` 用合并前快照恢复。
- `adapter_state_dict`：只导出 A/B，生成独立 `adapter.pt`（几 KB 级，可热插拔）。

## 4. 运行

```bash
# 全量 SFT
.venv/bin/python scripts/run_sft.py --config configs/sft.yaml
# LoRA SFT（use_lora: true 切换；merge: true 时另存 merged.pt）
.venv/bin/python scripts/run_sft.py --config configs/lora_sft.yaml
```

## 5. 实测

- LoRA 可训练参数：rank=8 → 每层 7×(256×8+8×256)×2 ≈ 57K，6 层 ≈ **0.34M 可训练参数**（基座 7.21M 冻结）。
- 训练后对话样例见 `logs/sft_train.log`（`[chat] 提示词 → 回答`）。

## 6. 踩坑记录

| 问题 | 原因 | 修复 |
|---|---|---|
| LoRA 目标只包到 6 层 | targets 短名 `q` 与模块叶子名 `wq` 不匹配 | `SHORT_TO_ATTR` 映射 q→wq、k→wk、v→wv、o→wo |
| merge 后前向不一致 | wrapper 仍在叠加 LoRA 项，双重计数 | merge 时把 A/B 置零；unmerge 从快照（含 A/B）恢复 |
| 掩码测试 AttributeError | tokenizer 未暴露 special id 公共接口 | bpe.py 增加 `special_ids` property |

## 7. 验收

- [x] LoRA 单测：目标覆盖 14 层 / 初始等价 / 只训练 A/B / merge-unmerge 往返 / adapter 独立存取
- [x] SFT 数据单测：掩码仅覆盖回答段 / 打包长度 / batch 形状
- [x] 全量单测 35 passed
- [x] 全量 SFT 与 LoRA SFT 均训练完成，对话可产出
- [x] 分支 `p3-sft-lora` 推送 gitee，合并 `main`

## 8. 下一步

P4：DPO 偏好对齐（参考模型 logprob 缓存 + β 温度 + SFT 正则）。
