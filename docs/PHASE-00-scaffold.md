# PHASE-00：仓库初始化与脚手架

> 阶段目标：建立可运行、可测试、可扩展的仓库骨架，全部配置走 YAML，为后续 10 个阶段打底。

## 本阶段内容

- Git 仓库初始化（`main` 基线），远程 gitee（SSH）。
- 目录结构：`configs / tinygpt / scripts / tests / data / checkpoints / docs`。
- 统一配置体系 `tinygpt/config.py`：YAML → dataclass 递归解析，未知字段报错。
- 7 份阶段配置：`tokenizer / pretrain / sft / lora_sft / dpo / mm_finetune / api`。
- 日志工具 `tinygpt/utils/logging.py`（控制台 + JSONL 指标）。
- 虚拟环境 `.venv`（Python 3.12.7）+ `requirements.txt`（torch 2.2.2 CPU 等）。

## 关键设计决策

1. **Python 版本**：Intel Mac 的 PyTorch wheel 止步于 2.2.x，且不支持 Python 3.13+；本机默认 python3 是 3.14.7，因此项目强制使用 `.venv`（Python 3.12.7）。
2. **配置驱动一切**：模型尺寸（hidden/layers/heads/vocab）、数据路径、训练超参全部来自 YAML，后续放大模型只改配置。
3. **全量 / LoRA 并存**：`model.use_lora` 开关从配置层就绪，LoRA 目标覆盖 q/k/v/o/gate/up/down 全 7 类。
4. **内存策略**：`num_workers=0`、小 batch + 梯度累积，适配 4 核 CPU。

## 验证

```bash
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
```

预期：`test_config.py` 全部通过（7 份 YAML 可解析、嵌套 LoRA 结构正确、未知字段报错、use_lora 开关生效）。

## 产物

- `docs/PROJECT_PLAN.md`：总体规划（已按确认项更新）。
- `docs/architecture.html`：架构与里程碑图。

## 下一步

P1：字节级 BPE 中文分词器（`tinygpt/tokenizer/bpe.py` + `scripts/train_tokenizer.py` + 小语料下载）。
