# PHASE-04：DPO 偏好对齐

> 分支：`p4-dpo` → 合并入 `main`
> 前置：P3 SFT 权重 `checkpoints/sft/sft.pt`

## 1. 本阶段目标

在 SFT 模型之上做偏好对齐，让模型更倾向"好答案"而非"坏答案"：

1. **参考模型 logprob 缓存**：DPO 需要参考模型（SFT 基座）对 chosen/rejected 的 logprob；
   一次离线算完缓存到 `data/datasets/dpo_ref_logprobs.pt`，训练时只读缓存、不必常驻参考模型（省一半显存/内存）。
2. **β 温度**：`dpo_beta` 控制偏好强度的缩放（0.1 默认）。
3. **附加 SFT 正则**：`dpo_sft_reg` 在 DPO 损失上叠加 chosen 回答段的 CE，防止策略偏离语言能力。

## 2. 损失函数（tinygpt/train/dpo.py）

```
logπ(y|x) = mean_{回答段 token} log pθ(token_t | x, y<t)     ← 只累计回答段，忽略 prompt

L_DPO = -log σ( β · ( logπθ(y_w) − logπref(y_w) − logπθ(y_l) + logπref(y_l) ) )
L     = L_DPO + dpo_sft_reg · CE(chosen 回答段)              ← SFT 正则
```

直观理解：`logπθ(y_w) − logπθ(y_l)` 是策略对 chosen/rejected 的偏好差距，参考模型 logprob 作为基线，
只让策略"比参考模型更偏好 chosen"即可（防止一味抬高概率）。

## 3. 流水线

```bash
# 1) 预计算参考模型 logprob 缓存（2000 对，CPU 约 2-4 分钟）
.venv/bin/python scripts/run_dpo.py --config configs/dpo.yaml --precompute
# 2) DPO 训练（策略 = SFT 基座 + LoRA，见 dpo.yaml use_lora: true）
.venv/bin/python scripts/run_dpo.py --config configs/dpo.yaml
```

产物：`checkpoints/dpo/adapter.pt`（LoRA adapter + dpo_beta 元信息）。

## 4. 配置（configs/dpo.yaml 关键项）

| 参数 | 默认 | 说明 |
|---|---|---|
| `train.dpo_beta` | 0.1 | 偏好强度温度 |
| `train.dpo_sft_reg` | 0.2 | SFT 正则权重 |
| `train.ref_logprobs_path` | data/datasets/dpo_ref_logprobs.pt | 参考 logprob 缓存路径 |
| `train.init_from` | checkpoints/sft/sft.pt | 策略/参考模型基座 |
| `data.dpo_path` | data/datasets/dpo_pairs.jsonl | 偏好对（prompt/chosen/rejected） |

## 5. 踩坑记录

| 问题 | 原因 | 修复 |
|---|---|---|
| 训练时 logprob 无梯度 | `sequence_logprobs` 内写死 no_grad/eval | 梯度开关交给调用方：预计算外层 no_grad，训练保持梯度 |

## 6. 验收

- [x] DPO 单测：logprob 掩码 / 训练后 chosen−rejected logprob>0 / 缓存形状 [N]
- [x] 全量单测 38 passed
- [x] 参考缓存预计算完成，DPO 训练完成并产出 adapter
- [x] 分支 `p4-dpo` 推送 gitee，合并 `main`

## 7. 下一步

P5：推理封装（chat / 工具函数调用 / 安全执行器 AST 白名单）。
