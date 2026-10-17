# tiny-gpt 项目规划

> 目标：在当前目录从零实现一个**可真正训练、可推理、可服务**的微型 GPT 全链路项目——预训练 → SFT → LoRA → DPO → 多模态（视觉）→ 工具调用 → MCP → RAG → OpenAI 兼容 API → LangChain 接入。
> 硬约束：在你的 Mac（2 GHz 四核 Intel Core i5 / Intel Iris Plus Graphics 1536MB / 16GB 内存 / macOS 13.1）上，纯 CPU 即可跑通全部训练。

---

## 1. 硬件约束分析与设计原则

| 硬件项 | 约束 | 应对策略 |
| --- | --- | --- |
| CPU：2GHz 四核 Intel i5（x86_64） | 无 NVIDIA GPU；**macOS Intel 不支持 MPS**（MPS 仅 Apple Silicon） | 训练走纯 CPU + fp32；`torch.set_num_threads(4)`；Dataloader 不用多进程（`num_workers=0`）避免内存翻倍 |
| 内存：16GB | 训练态内存 = 参数+梯度+优化器状态 ≈ 16B/参数（fp32+AdamW） | 目标模型 **~17M 参数**（训练态 < 300MB，峰值 < 2GB，余量充足）；LoRA 阶段更省 |
| iGPU：Intel Iris Plus 1536MB | 不能用于训练 | 不接入训练，仅做普通显示 |
| 核数少 | 吞吐有限 | 小 batch（8–16）+ 梯度累积；**打包训练**减少 padding 浪费；序列长度 256–512 |
| macOS 13.1 Intel | PyTorch 已停止 Intel Mac 新版 wheel | 项目使用 **Python 3.12 + torch 2.2.x（CPU）** 专用 venv（本机默认 Python 3.14 无 Intel Mac wheel） |

**规模结论**：推荐基座模型 ~17M 参数（hidden=384 / 8 层 / 8 头 / 词表 8192），小语料一轮预训练可在**一晚内完成**（精确时长先跑 `scripts/bench.py` 定标）。

---

## 2. 总体架构

```
┌─ 数据层：中文语料 / 指令集 / 偏好对 / 图文集 / 知识库
│
├─ 分词器：字节级 BPE（自训，词表 8192，支持中文）
│
├─ 训练层（配置开关 model.use_lora: false|true，全量 / LoRA 并存）
│   ├─ 预训练：GPT decoder-only（RoPE / SwiGLU / RMSNorm / 权重共享）
│   ├─ SFT：指令数据 + 回答段掩码损失 + 打包训练
│   ├─ LoRA：q/k/v/o/gate/up/down 全目标，可合并 / 不合并
│   ├─ DPO：参考模型 logprob 缓存 + β 温度 + 附加 SFT 正则
│   └─ 多模态：微型 ViT（patch 嵌入+位置编码+多头注意力+池化）→ 投影层 → 联合微调
│
├─ 推理与能力层
│   ├─ 生成/对话（贪心、top-k、top-p、温度采样）
│   ├─ 工具函数调用：OpenAI 风格 tools 解析 + AST 白名单安全执行器
│   ├─ MCP 客户端：stdio JSON-RPC（initialize → tools/list → tools/call）
│   └─ RAG：知识库入库 → 字符 bigram + TF-IDF 检索 → 上下文注入，暴露为 search_kb 工具
│
└─ 服务层：OpenAI 兼容 API + LangChain 接入（base_url 直连 / bind_tools）
```

详细模块图见 `docs/architecture.html`。

---

## 3. 目录结构

```
tiny-gpt/
├── README.md                  # 安装（Intel Mac CPU 指引）+ 快速开始
├── requirements.txt
├── configs/                   # YAML 配置，一个阶段一个文件
│   ├── tokenizer.yaml
│   ├── pretrain.yaml
│   ├── sft.yaml
│   ├── lora_sft.yaml
│   ├── dpo.yaml
│   ├── mm_finetune.yaml
│   └── api.yaml
├── tinygpt/
│   ├── config.py              # dataclass + YAML 加载/校验
│   ├── tokenizer/
│   │   ├── bpe.py             # 字节级 BPE：训练、编码、解码、词表读写
│   ├── data/
│   │   ├── pretrain.py        # 语料 → 打包 token 块
│   │   ├── sft.py             # 指令集 → 回答段掩码 + 打包 batch
│   │   ├── dpo.py             # 偏好对 → 参考 logprob 缓存
│   │   └── mm.py              # 图文对 → ViT 特征
│   ├── model/
│   │   ├── gpt.py             # decoder-only：RoPE/SwiGLU/RMSNorm/权重共享
│   │   ├── lora.py            # LoRA 层 + 合并/还原
│   │   ├── vit.py             # 微型 ViT（patch 嵌入/位置编码/多头注意力/池化）
│   │   └── projector.py       # 视觉 ↔ 语言投影
│   ├── train/
│   │   ├── common.py          # 优化器/调度器/checkpoint/resume/日志
│   │   ├── pretrain.py
│   │   ├── sft.py
│   │   └── dpo.py
│   ├── inference/
│   │   ├── generate.py        # 采样 + chat 模板
│   │   ├── tools.py           # tools 解析 + AST 白名单安全执行器
│   │   ├── mcp_client.py      # stdio JSON-RPC 客户端
│   │   ├── rag.py             # bigram + TF-IDF 检索器 / search_kb
│   │   └── multimodal.py      # 图片输入 → 问答
│   ├── server/
│   │   └── api.py             # FastAPI OpenAI 兼容端点
│   └── utils/                 # seed/日志/指标
├── scripts/
│   ├── train_tokenizer.py
│   ├── run_pretrain.py / run_sft.py / run_dpo.py / run_mm.py
│   ├── run_api.py             # 启动服务
│   ├── build_kb.py            # RAG 入库
│   ├── bench.py               # 训练吞吐定标
│   └── eval.py                # 困惑度 + 生成样例
├── tests/                     # 单测 + 冒烟训练 + 端到端
├── data/                      # 语料/词表/数据集（demo 集内置，真实集可配置路径）
└── checkpoints/               # 各阶段产物（base.pt / sft.pt / adapter.safetensors 等）
```

---

## 4. 模块设计（对照功能清单）

### 4.1 字节级 BPE 中文分词器（自训）
- 词表统计按 UTF-8 **字节**进行 merge（天然兼容中文/英文/生僻字/emoji，无未登录字问题）。
- 训练脚本：`scripts/train_tokenizer.py --config configs/tokenizer.yaml`；词表大小可配（默认 8192）。
- 持久化为 `tokenizer.json`（merges + vocab + special tokens），提供 `encode/decode`；特殊 token：`<s> </s> <pad> <unk> <|user|> <|assistant|> <|image|> <|tool|>` 等。
- 验收：中英混合文本 encode→decode 无损回环；OOV 字节可编码；与参考实现（如 tiktoken 思路）对照计数一致。

### 4.2 GPT decoder-only 模型（tinygpt/model/gpt.py）
- RoPE 旋转位置编码（q/k 施加，无需位置嵌入表）；SwiGLU 前馈（gate/up/down 三段）；RMSNorm（加 epsilon）；**输入输出词嵌入权重共享**；因果注意力 + 环形 buffer 缓存（推理 KV cache 可选）。
- 支持 `use_lora` 开关：True 时线性层替换为 LoRA 包装。
- 验收：单测校验 RoPE 相对位置不变性、掩码损失与手算一致、权重共享后参数量符合预期。

### 4.3 预训练
- 配置：`configs/pretrain.yaml`（模型尺寸 / seq_len=256 / batch=16 / 梯度累积 / AdamW / warmup + cosine / fp32）。
- 语料 → 打包（连续 token 块，块间 `<s>` 分隔，长度对齐截断），避免 padding 浪费。
- checkpoint：`base.pt`（模型+优化器+step+config），支持断点续训；日志 jsonl（loss/ppl/lr/吞吐）。
- 验收：小语料冒烟 1 步可跑；10~30MB 语料多 epoch 后训练/验证 loss 下降；生成中文文本通顺可读。

### 4.4 SFT 监督微调
- 输入：指令 JSONL（`{"instruction":..., "input":..., "output":...}` 或 chat turns）。
- **回答段掩码**：prompt 段 loss 置零，仅对回答段计算 CE 损失；用 segment id 或 special token 边界实现。
- **打包训练**：多条样本按回答段边界拼接进同一序列（`<s>prompt<|assistant|>answer</s>...`），掩码随样本迁移。
- 产出 `sft.pt`；验收：掩码位置与手算一致；训练后按指令能产出对应回答；冒烟 200 条样本可过拟合。

### 4.5 LoRA 低秩微调
- 目标：`q/k/v/o/gate/up/down` 全部 7 类投影（与模型实现一一对应），`rank/alpha/dropout` 可配。
- **可合并**：`lora.merge: true|false` 配置开关——合并（权重写回 W，无需包装）与不合并（保留 adapter，可单独存/载）均支持；合并可还原（保存原 W 副本）。
- 与全量并存：`model.use_lora: false|true` 全局切换；LoRA 时冻结基座、仅训练 adapter（内存/时间大幅下降）。
- 产出 `adapter.pt`（或合并后 `lora_merged.pt`）；验收：合并前后 logits 数学等价（误差 <1e-5）；不合并可单独推理。

### 4.6 DPO 偏好对齐
- 输入：偏好对 `(prompt, chosen, rejected)`；β 温度可配（默认 0.1）。
- **参考模型 logprob 缓存**：先用冻结参考模型对 chosen/rejected 一次前向，缓存 logprob 到磁盘（parquet/json），训练循环不再加载参考模型，CPU 内存减半。
- 损失 = DPO 项 + **附加 SFT 正则**（chosen 上的 CE 项，λ 可配，防分布漂移）。
- 产出 `dpo.pt`；验收：DPO 损失与参考实现（手写公式）一致；训练后 chosen 概率相对提升。

### 4.7 多模态（视觉）
- **微型 ViT 从零实现**：patch 嵌入（16×16 → 线性投影）→ 可学习位置编码（+CLS）→ N 层多头自注意力（LayerNorm + MLP + GELU）→ CLS 池化/分类头；尺寸可配（patch=16 / embed=192 / depth=6 / heads=6）。
- 投影层：ViT 输出（若干视觉 token / CLS）→ MLP → LLM hidden 维，插入 LLM 输入序列。
- **联合微调**：图文问答数据 → ViT（可冻结或微调）+ 投影 + LLM（全量或 LoRA）；损失与 SFT 相同（仅回答段）。
- 验收：ViT 前向 shape 正确；图文样本可训练并回答"图里有什么"类问题；API 支持 image_url 输入。

### 4.8 工具函数调用（安全执行器）
- OpenAI 风格 `tools` 解析：模型输出 `tool_calls`（name + arguments JSON）→ 分发执行 → 结果回填对话。
- **安全执行器**：AST 白名单——只允许白名单节点（表达式/赋值/分支/循环），仅开放白名单内置函数与模块（math/json/str 运算等），**禁止 import、属性访问、eval/exec、open、os 等**；代码长度/执行时间双限制，异常捕获并返回结构化错误。
- 内置工具示例：`calculator`、`search_kb`（RAG）；外部工具经 MCP 注入。
- 验收：正常算术可用；注入测试（`import os`、`os.system("rm -rf")`、`__import__`、`open("/etc/passwd")`、死循环）全部被拦截并给出安全报错。

### 4.9 MCP 客户端
- stdio 传输 + JSON-RPC（行分隔）：`initialize`（协议版本协商）→ `tools/list`（转成 OpenAI function schema）→ `tools/call`（按 tool_call_id 执行）。
- 任意 MCP Server：配置 `command + args + env` 即可连接；工具自动注册进工具循环，与内置工具并列。
- 验收：用小型 mock MCP Server（如 filesystem/echo）单测三阶段握手与调用返回。

### 4.10 RAG 检索增强
- 入库：文档切块 → 建立**字符 bigram**索引 + **TF-IDF**（自实现，无 sklearn 强依赖，可选 fallback）。
- 检索：bigram 重叠分 + TF-IDF 余弦分加权 → top-k 上下文注入（带来源引用模板）。
- 暴露为 `search_kb` 工具（可被 agent 循环调用），并提供 `/v1/rag/completions`。
- 验收：检索排序在小型知识库（如公司制度/产品 FAQ 文本）上与人工预期一致；注入后回答引用正确片段。

### 4.11 OpenAI 兼容 API（FastAPI + uvicorn）
- `GET /v1/models`；`POST /v1/chat/completions`（**非流式 + SSE 流式**、多模态 image_url、tools/tool_calls）；`POST /v1/completions`（legacy）；`POST /v1/agent/completions`（**工具循环**：tools+MCP+RAG 自动迭代至最终回答）；`POST /v1/rag/completions`。
- 请求/响应 schema 与 OpenAI 对齐（`choices/delta/tool_calls/usage` 等）。
- 验收：curl 冒烟各端点；流式逐 chunk 输出；工具循环多轮调用正确终止。

### 4.12 LangChain 接入
- `ChatOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-demo")` 直连；`bind_tools` 可用（依赖 /v1/chat/completions 的 tool_calls 格式与 OpenAI 一致）。
- 提供 `examples/langchain_demo.py`：绑定工具 + 多轮对话示例。
- 验收：LangChain 脚本端到端跑通，工具被正确调用并回填结果。

---

## 5. 数据方案与训练规模（适配硬件）

| 阶段 | 数据规模（推荐起步） | 关键超参 | 训练量 | 预计时长（4 核 CPU，实测定标为准） |
| --- | --- | --- | --- | --- |
| 分词器 | 语料 5–30MB 中文 | 词表 8192 | 单次 | 数分钟 |
| 预训练 | 5–30MB 中文语料（≈2–8M tokens） | hidden384/8层/8头，seq256，batch16 | 3–5 epoch | 1–4 小时 |
| SFT | 2k–20k 指令样本 | 掩码 + 打包，seq≤512 | 2–3 epoch | 30–90 分钟 |
| LoRA SFT | 同上 | r=8, α=16, 全 7 目标 | 2–3 epoch | 20–60 分钟 |
| DPO | 1k–5k 偏好对 | β=0.1，参考缓存，λ_sft 可配 | 1–2 epoch | 30–90 分钟 |
| 多模态 | 500–2k 图文问答对 | ViT 6M + 投影 + LoRA | 1–2 epoch | 1–2 小时 |
| RAG | 知识库 0.5–5MB 文本 | bigram + TF-IDF，top-k=3–5 | 入库一次 | 分钟级 |

- 内存峰值预估：17M 模型 fp32 训练态 ≈ 300MB + 激活/数据 ≈ **峰值 < 2GB**，16GB 内存无压力；40M 模型（可选"较大"档）峰值约 1.5–2GB 也放得下，但训练时间约 ×2–3。
- **数据来源策略（已确认：允许联网下载小型公开集）**：`scripts/download_data.py` 提供多源回退链（HF 镜像等）下载小型公开中文语料（预训练文本 / 指令集），并支持 `max_samples` 截断；仓库同时内置极小 demo 语料兜底，离线也能跑通全链路。DPO 偏好对可用指令集确定性构造；多模态先用内置合成图集（PIL 生成、标签可验证），后续可换真实 VQA 数据。

---

## 6. 工程与质量保障

- **依赖最小化**：`torch==2.2.2`（Intel Mac CPU wheel，Python 3.12 venv）、`numpy`、`pyyaml`、`fastapi`、`uvicorn`、`pydantic`、`pillow`（多模态）；`langchain`/`scikit-learn` **暂不安装**，接入点提供示例脚本。
- **可复现**：全局 seed；checkpoint 含 config + optimizer + step；断点续训；日志 jsonl。
- **验证策略**：
  - 单测：分词器回环 / RoPE 不变性 / 掩码损失手算对照 / LoRA 合并等价 / DPO 公式对照 / ViT shape / 执行器安全注入用例 / MCP mock 握手 / RAG 排序 / API 各端点与流式。
  - 冒烟训练：极小语料 1 步预训练、200 条 SFT 可过拟合、1k 条 DPO 损失下降。
  - 端到端：起服务后 curl 各端点 + LangChain 脚本 + 图片问答 + 工具循环。
  - 安全：工具执行器注入用例全套通过。
- **真机验证**：开发与验证直接在目标机器（你的 Mac）上进行；单元测试/冒烟/端到端全部在本机 `.venv` 中运行（Python 3.12 + torch 2.2.2 CPU）。

---

## 7. 里程碑与验收标准

| 阶段 | 交付物 | 验收标准 |
| --- | --- | --- |
| P0 脚手架 | 仓库结构 / config 体系 / 工具函数 / 测试骨架 | 所有空模块可 import，单测框架跑通 |
| P1 分词器 | tokenizer.json + BPE 模块 | 回环无损、计数对照一致 |
| P2 模型 + 预训练 | gpt.py + 预训练脚本 + base.pt | 冒烟 1 步可跑；loss 下降；可生成中文 |
| P3 SFT + LoRA | sft.py / lora.py + sft.pt / adapter.pt | 掩码正确；指令可答；合并等价 |
| P4 DPO | dpo.py + 参考缓存 + dpo.pt | 公式对照；chosen 概率提升 |
| P5 推理 + 工具 | generate.py / tools.py | 采样多样；注入全拦截 |
| P6 多模态 | vit.py / projector.py + 图文微调 | 图片问答可用 |
| P7 MCP | mcp_client.py + mock server 测试 | 三阶段握手 + 调用成功 |
| P8 RAG | rag.py + build_kb.py + search_kb | 检索排序正确、可注入 |
| P9 API + LangChain | server/api.py + 示例 | 全端点 + 流式 + bind_tools 跑通 |
| P10 真机调优 | 在你的 Mac 上端到端 + 文档 | 全链路一晚内跑通，README 完整 |

> 顺序策略：**先 P0–P2 跑通最小闭环，再逐模块叠加**；每阶段产物可独立验证，避免到最后才暴露问题。
>
> **分支与文档（已确认）**：每阶段在独立分支 `pN-<阶段名>` 上完成，配套 `docs/PHASE-0N-*.md` 学习文档；阶段完成后合并回 `main`，分支保留作为学习历史。远程仓库：`git@gitee.com:matisha001/tiny-gpt.git`（SSH 已认证）。

---

## 8. 运行方式（示例）

```bash
# 安装（Intel Mac）
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 分词器 → 预训练 → SFT → DPO → 服务
python scripts/train_tokenizer.py --config configs/tokenizer.yaml
python scripts/run_pretrain.py    --config configs/pretrain.yaml
python scripts/run_sft.py         --config configs/sft.yaml          # 或 lora_sft.yaml
python scripts/run_dpo.py         --config configs/dpo.yaml
python scripts/run_api.py         --config configs/api.yaml          # :8000

# 测试
pytest tests/ -q
python scripts/eval.py --config configs/pretrain.yaml
```

---

## 9. 已确认决策（2026-10-17）

| 决策项 | 结论 |
| --- | --- |
| 模型规模 | 默认不大（~17M），**全部尺寸 YAML 可配置**，后续直接改 config 扩大 |
| 数据来源 | **允许联网下载小型公开集**（download_data.py 多源回退 + max_samples 截断）；内置 demo 语料离线兜底 |
| 时长预算 | **1–2 小时跑通全链路 demo**（小数据小模型）；要更好效果按配置放大（数小时一晚） |
| Python | 本机默认 3.14.7 不兼容 Intel Mac torch；项目使用 **Python 3.12.7 venv + torch 2.2.2 CPU** |
| 过程拆分 | 每阶段独立分支 `pN-<阶段名>` + 独立 `docs/PHASE-0N-*.md`，阶段完成合并回 `main` |
| Git 远程 | `git@gitee.com:matisha001/tiny-gpt.git`（SSH 已认证可用） |
| 可选依赖 | langchain / scikit-learn 等**暂不安装**，接入点提供示例脚本 |

> 说明：1–2h 预算内各阶段以“跑通 + 验证正确”为验收标准，模型质量为 demo 级；每个 config 均预留放大路径。
