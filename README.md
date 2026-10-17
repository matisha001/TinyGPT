# tiny-gpt

从零实现的微型 GPT 全链路项目：**预训练 → SFT → LoRA → DPO → 多模态（视觉）→ 工具调用 / MCP / RAG → OpenAI 兼容 API → LangChain 接入**，纯 CPU 可训练可推理。

> 目标硬件：macOS 13.1（Intel）/ 2 GHz 四核 i5 / 16GB 内存。模型规模默认很小（~17M 参数），全部尺寸由 YAML 配置驱动，方便后续扩大。

## 快速开始

```bash
# 1. 创建虚拟环境（必须用 Python ≤3.12；Intel Mac 无 3.13+ 的 torch wheel）
/usr/local/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 下载小型公开数据（或使用内置 demo 语料）
python scripts/download_data.py --config configs/tokenizer.yaml

# 3. 训练分词器
python scripts/train_tokenizer.py --config configs/tokenizer.yaml

# 4. 预训练
python scripts/run_pretrain.py --config configs/pretrain.yaml

# 5. SFT / LoRA / DPO
python scripts/run_sft.py  --config configs/sft.yaml
python scripts/run_sft.py  --config configs/lora_sft.yaml
python scripts/run_dpo.py  --config configs/dpo.yaml

# 6. 启动 OpenAI 兼容 API（流式 / 多模态 / 工具 / agent 循环 / RAG）
python scripts/run_api.py --config configs/api.yaml

# 7. 测试
pytest tests/ -q
```

## 目录速览

```
configs/    各阶段 YAML 配置（模型尺寸、数据、训练超参）
tinygpt/    核心包：tokenizer / model / data / train / inference / server
scripts/    命令行入口：训练、评估、建库、API、数据下载
tests/      单元测试 + 冒烟
docs/       项目规划 + 每阶段学习文档（PHASE-XX-*.md）
data/       语料、分词器产物、数据集（下载内容不入库，demo 数据内置）
checkpoints/ 模型产物（不入库）
```

## 分支策略（按阶段学习）

| 分支 | 内容 |
| --- | --- |
| `main` | 最新稳定代码（每阶段合并回来） |
| `p0-scaffold` | 仓库骨架 + 配置体系 |
| `p1-tokenizer` | 字节级 BPE 中文分词器 |
| `p2-pretrain` | GPT decoder-only + 预训练 |
| `p3-sft-lora` | SFT（掩码+打包）+ LoRA |
| `p4-dpo` | DPO 偏好对齐 |
| `p5-inference-tools` | 采样 + 工具调用（AST 安全执行器） |
| `p6-multimodal` | 微型 ViT + 图文微调 |
| `p7-mcp` | MCP 客户端 |
| `p8-rag` | RAG 检索增强 |
| `p9-api-langchain` | OpenAI 兼容 API + LangChain |
| `p10-tuning` | 真机端到端调优 |

每个阶段都有一份 `docs/PHASE-XX-*.md` 说明设计、实现与验证结果。

## 远程仓库

- Gitee（SSH）：`git@gitee.com:matisha001/tiny-gpt.git`（本机 SSH 已认证）

## 说明

- 全部模型代码从零实现（不依赖 transformers 训练），仅用 PyTorch。
- 1–2 小时预算内可跑通**全链路 demo**（小数据、小模型）；要更好的模型质量，按 config 放大数据与训练量（数小时到一晚）。
- 可选依赖 langchain / scikit-learn 暂不安装（对应接入点提供示例脚本）。
