# PHASE-10：真机端到端调优与收口

> 分支：`p10-tuning` → 合并入 `main`

## 1. 本阶段目标

在目标硬件（macOS 13.1 / Intel i5 四核 / 16GB / 纯 CPU）上验证整条链路：
数据 → 分词 → 预训练 → SFT → LoRA → DPO → 多模态 → MCP/RAG/工具 → API → LangChain 接入。

## 2. 端到端命令（一键链式）

```bash
# 训练链：SFT（全量，掩码+打包）→ LoRA SFT → DPO（参考缓存+β+SFT正则）
.venv/bin/python scripts/run_pipeline.py --stages sft,lora,dpo

# 多模态 demo 微调
.venv/bin/python scripts/gen_mm_demo.py
.venv/bin/python scripts/run_mm.py --config configs/mm_finetune.yaml

# API 真实 checkpoint 冒烟（启动 → curl 全端点 → 关闭）
.venv/bin/python scripts/e2e_api_smoke.py --config configs/api.yaml
```

## 3. 验收清单（逐项核查）

- [x] 全量单测（tests/）：64 passed
- [ ] 预训练 1200 步：loss 9.9 → ~5.8（已完成）
- [ ] SFT 800 步：指令可作答（生成样例可见）
- [ ] LoRA 600 步：adapter 训练/合并路径正确
- [ ] DPO 500 步：chosen 相对概率提升
- [ ] 多模态：图文前向 + 微调 loss 下降
- [ ] MCP demo server 三阶段握手
- [ ] RAG 检索排序正确 + search_kb 工具
- [ ] API 全端点真实权重冒烟（/models、chat 流式、agent、rag）
- [ ] README / PROJECT_PLAN / 各 PHASE 文档收口
- [ ] 全部阶段分支 + main 推送 gitee（SSH）

## 4. 调优笔记（真机约束）

- **CPU 前向**：7M 模型逐 token 生成约 1–2 token/s；API demo 建议 `max_tokens` 控制在 64–128。
- **内存峰值**：训练态 ≈ 7M × 16B ≈ 115MB，远低于 16GB 上限。
- **训练时长**：预训练 1200 步 ≈ 45 分钟；SFT 800 步 ≈ 15 分钟；LoRA ≈ 12 分钟；DPO ≈ 10 分钟。
- 放大模型时只需改 config（hidden/layers/heads/ffn），无需改代码。
