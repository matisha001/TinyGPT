# PHASE-06：多模态（从零微型 ViT + 投影层 + 联合微调）

> 分支：`p6-multimodal` → 合并入 `main`

## 1. 本阶段目标

从零实现微型 Vision Transformer（不用 torchvision 现成模型），经投影层接入 LLM，
支持「图片 + 问题 → 文本回答」的联合微调。

## 2. 微型 ViT（tinygpt/model/vit.py）

| 组件 | 实现 |
|---|---|
| patch 嵌入 | `Conv2d(kernel=stride=patch_size)` ≡ 逐 patch 线性投影 → [B, N, hidden] |
| 位置编码 | 可学习 `pos_embed`（含 cls token 位），trunc_normal 初始化 |
| 多头自注意力 | 双向 `nn.MultiheadAttention`（Pre-Norm） |
| MLP | Linear → GELU → Linear（ratio 4.0） |
| 分类头/池化 | `head(pooled)`；pool 支持 `cls` 或 `mean` |

Demo 配置：image 64×64 / patch 16 → 4×4=16 个 patch；hidden 192 / depth 6 / heads 6
（`configs/mm_finetune.yaml` 可改回 224 全尺寸）。

## 3. 投影层与联合微调

- `VisionProjector`：可学习查询（num_vision_tokens 个）× 注意力池化压缩 patch 特征 →
  MLP 投影到 LLM hidden。
- `VisionGPT`：ViT → projector → 视觉 token 拼接在文本序列头部 → GPT 前向；
  视觉 token 不参与损失，`loss_mask` 只作用于 assistant 回答段。
- 联合微调：`vision.freeze: true` 冻结 ViT（只训投影层 + LLM）；LLM 部分支持
  `model.use_lora` 开关（LoRA 目标 q/k/v/o/gate/up/down 与 P3 一致）。
- 从 `checkpoints/pretrain/base.pt` 载入 LLM 权重（`strict=False`，ViT/投影层随机初始化）。

## 4. Demo 数据

`scripts/gen_mm_demo.py`：生成 8 张 64×64 纯色（带白色圆形）PNG + `mm_train.jsonl`，
问答形如 「这张图片的主色调是什么？」→「主色调是红色。」。

## 5. 运行

```bash
.venv/bin/python scripts/gen_mm_demo.py
.venv/bin/python scripts/run_mm.py --config configs/mm_finetune.yaml --metrics logs/mm_metrics.jsonl
```

## 6. 验收

- [x] ViT：patch 嵌入 / 位置编码 / 多头注意力 / 池化 / 分类头 前向正确
- [x] 投影层：16 patch → 4 视觉 token
- [x] VisionGPT：带图前向 logits 序列 = 视觉 token + 文本；loss 有限且掩码生效
- [x] demo 数据生成 + mm 微调入口就绪
- [x] 单测 5 passed；分支 `p6-multimodal` 推送 gitee，合并 `main`

## 7. 说明与下一步

模型极小、样本仅 8 张，无法学到真实视觉语义——本阶段验证的是「ViT→投影→LLM
联合微调」的完整管线。下一步 P7：MCP 客户端（stdio JSON-RPC）。
