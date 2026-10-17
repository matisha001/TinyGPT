"""构建 RAG 知识库 demo 数据并建立索引。
python scripts/build_kb.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinygpt.rag import KnowledgeBase

DEMO_DOCS = [
    {"text": "豆包是字节跳动推出的 AI 智能助手，支持文本对话、文档总结、编程辅助等功能。",
     "meta": {"topic": "豆包"}},
    {"text": "tiny-gpt 是一个用于教学的全链路微型 GPT 项目：分词 → 预训练 → SFT → LoRA → DPO → 多模态。",
     "meta": {"topic": "tiny-gpt"}},
    {"text": "LoRA 低秩微调通过在注意力与 FFN 的 q/k/v/o/gate/up/down 上注入低秩适配器，大幅减少可训练参数。",
     "meta": {"topic": "LoRA"}},
    {"text": "DPO 直接偏好优化利用参考模型对数概率构造隐式奖励，配合 β 温度与 SFT 正则实现偏好对齐。",
     "meta": {"topic": "DPO"}},
    {"text": "RoPE 旋转位置编码将相对位置信息编码进注意力分数；SwiGLU 与 RMSNorm 是 GPT 风格 decoder-only 的常用组件。",
     "meta": {"topic": "模型结构"}},
    {"text": "Python 是一种广泛使用的编程语言，常用于机器学习与数据分析。",
     "meta": {"topic": "Python"}},
    {"text": "上海是中国最大的城市之一，也是重要的金融与科创中心。",
     "meta": {"topic": "上海"}},
]


def main() -> None:
    kb = KnowledgeBase("data/kb_index/index.json").load_or_build(DEMO_DOCS)
    for q in ["什么是 LoRA？", "tiny-gpt 能做什么？", "DPO 的原理是什么？"]:
        hits = kb.search(q, top_k=2)
        print(f"\nQ: {q}")
        for h in hits:
            print(f"  [{h['score']:.3f}] {h['text']}")
    print(f"\n索引已保存: {kb.index_path}（docs={len(kb.index.docs)}, vocab={len(kb.index.vocab)}）")


if __name__ == "__main__":
    main()
