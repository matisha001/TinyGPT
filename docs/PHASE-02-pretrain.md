# PHASE-02：GPT decoder-only 预训练

> 分支：`p2-pretrain` → 合并入 `main`
> 前置：P1 分词器（`data/tokenizer/tokenizer.json`）+ 预训练语料（`data/corpus/pretrain.txt`）

## 1. 本阶段目标

1. 从零实现 GPT decoder-only 模型：RoPE / SwiGLU / RMSNorm / 输入输出词嵌入权重共享。
2. 实现打包式预训练数据流与完整训练循环（warmup + cosine、梯度累积、梯度裁剪、断点续训、checkpoint）。
3. 实现自回归采样（temperature / top-k / top-p）。
4. 在 Intel Mac 纯 CPU 上把 17.31M 参数的原始配置跑冒烟，发现 5s/步过慢，**整体下调默认配置到 ~7M 参数的 demo 档**（保留放大路径），保证 1-2 小时全链路预算。
5. 产出首个可用的 `checkpoints/pretrain/base.pt`，作为 SFT / DPO / 多模态的初始化权重。

## 2. 模型结构（tinygpt/model/gpt.py）

| 组件 | 实现 | 说明 |
|---|---|---|
| RMSNorm | `x / sqrt(mean(x²)+eps) * w` | 相比 LayerNorm 省去均值中心化与偏置，预归一化（Pre-Norm） |
| RoPE | 预计算 `cos/sin` 表，半切旋转施加到 q/k | 位置信息以旋转矩阵注入，天然支持相对位置外推 |
| 因果自注意力 | 单次矩阵注意力 + 下三角布尔掩码 | head_dim = hidden // heads |
| SwiGLU FFN | `down(silu(gate(x)) * up(x))` | gate/up 各投影到 `ffn_size`，再 down 回 hidden |
| 权重共享 | `lm_head.weight = token_embed.weight` | 输出层复用输入嵌入矩阵，显著省参数 |

### 参数规模（demo 档默认值）

```
词表 8192 × hidden 256（嵌入/输出共享） ≈ 2.10M
每层：注意力 4×256×256 ≈ 0.26M；SwiGLU 3×256×768 ≈ 0.59M；norm ≈ 0.5K
6 层 ≈ 5.1M → 合计 ≈ 7.3M 参数
```

**放大路径**（只需改 `configs/pretrain.yaml` 的 model 段，代码零改动）：
`hidden_size 256→384/768/1024`、`num_layers 6→8/12`、`num_heads`、`ffn_size`、`max_seq_len` 均为可配置项；CPU 上估算每步耗时随 hidden×layers×seq_len 线性增长。

## 3. 数据流（tinygpt/data/pretrain.py）

1. 语料按 20k 字符分块调用分词器 `encode`（P1 的批量合并算法，整语料 7.3M tokens 约 23s）。
2. 切成 `(seq_len+1)` 的连续块：块内 `input = blk[:, :-1]`、`target = blk[:, 1:]` —— 打包训练，几乎没有 padding 浪费。
3. 前 1% 块固定为评估集（`eval_blocks`），其余打乱进入训练。
4. `iter_batches()` 每 epoch 重洗一次，返回 `(input_ids, targets)`。

## 4. 训练循环（tinygpt/train/pretrain.py）

- AdamW：权重衰减只作用于 ≥2 维参数（不衰减 norm 与偏置）。
- LR：100 步 warmup 线性爬升 → cosine 衰减到 `min_lr`。
- 梯度累积 `grad_accum=4`：等效 batch = 16 × 4 = 64。
- 每 `log_every` 步打日志（JSONL 指标），每 `eval_every` 步在评估块上算 loss/PPL 并采样续写。
- checkpoint：`base.pt`（固定名，供后续阶段引用）+ `ckpt-last.pt`（断点续训用，`train.resume` 指到它）。

## 5. 运行

```bash
.venv/bin/python scripts/run_pretrain.py --config configs/pretrain.yaml
# 事后评估：loss/PPL + 提示词续写
.venv/bin/python scripts/eval.py --ckpt checkpoints/pretrain/base.pt
```

## 6. 实测结果（Intel i5 四核 CPU / torch 2.2.2）

- 语料编码：12.3M 字符 → 7,299,828 tokens，打包 28,120 训练块 + 284 评估块。
- 冒烟（384/8/8/1024，3000 步配置）：**约 5.0 s/步** → 3000 步需 4 小时+，超预算 → 下调。
- 下调后（256/6/4/768，1200 步）：loss 从初始 ≈ln(8192)=9.0 持续下降，PPL 同步改善；最终结果见 `logs/pretrain_train.log` 与 `logs/pretrain_metrics.jsonl`。

## 7. 踩坑记录

| 问题 | 原因 | 修复 |
|---|---|---|
| `view size is not compatible` | `targets` 来自 `blk[:, 1:]`，是非连续视图 | CE 前改用 `reshape(-1)` |
| 3000 步需 4h+ | 17.31M 模型在 2GHz CPU 上 5s/步 | 默认配置下调至 ~7M demo 档；放大走配置 |
| 生成测试 `IndexError` | 测试模型 vocab=300 但真实分词器 id 到 8191 | 测试按 `tokenizer.vocab_size_actual` 建模型 |

## 8. 验收

- [x] 模型结构单测：前向形状 / 权重共享 / 因果掩码位置独立性 / loss_mask / RoPE 等距性 / 小模型过拟合
- [x] 生成单测：回环 / 种子确定性 / 超长窗口截断 / top-k / top-p
- [x] 全量单测 27 passed
- [x] 预训练完整跑通，产出 `checkpoints/pretrain/base.pt`，loss 下降、采样可出中文
- [x] 分支 `p2-pretrain` 推送 gitee，合并 `main`

## 9. 下一步

P3：SFT（指令数据 + 回答段掩码 + 打包训练）与 LoRA（q/k/v/o/gate/up/down 全目标、可合并、`model.use_lora` 开关）。
