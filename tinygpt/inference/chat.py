"""对话推理封装：多轮消息 → 模板 → 生成。"""
from __future__ import annotations

from typing import List, Optional

import torch

from tinygpt.inference.generate import generate
from tinygpt.model import GPT
from tinygpt.tokenizer import BytePairTokenizer

BOS = "<s>"
EOS = "</s>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"


def build_chat_ids(tokenizer: BytePairTokenizer,
                   messages: List[dict]) -> List[int]:
    """messages: [{"role": "system|user|assistant", "content": str}, ...] → token ids。"""
    ids: List[int] = [tokenizer.special_ids[BOS]]
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            ids += tokenizer.encode(content)
        elif role == "assistant":
            ids += tokenizer.encode(f"{ASSISTANT}{content}{EOS}")
        else:
            ids += tokenizer.encode(f"{USER}{content}")
    return ids


@torch.no_grad()
def chat(model: GPT, tokenizer: BytePairTokenizer, messages: List[dict],
         max_new_tokens: int = 128, temperature: float = 0.8, top_k: int = 40,
         top_p: float = 0.9, seed: Optional[int] = None) -> str:
    """多轮对话：输入消息列表，返回 assistant 回答文本（仅生成 assistant 段）。"""
    from tinygpt.inference.generate import _generate_from_ids
    ids = build_chat_ids(tokenizer, messages)
    return _generate_from_ids(model, tokenizer, ids, max_new_tokens,
                              temperature, top_k, top_p,
                              tokenizer.special_ids[EOS], seed)


@torch.no_grad()
def chat_vision(model: "GPT", tokenizer: BytePairTokenizer, messages: List[dict],
                image: torch.Tensor, max_new_tokens: int = 128,
                temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9,
                seed: Optional[int] = None) -> str:
    """带图像输入的多模态对话；视觉 token 由模型在序列头部拼接。"""
    from tinygpt.inference.generate import _filter_top_k, _filter_top_p
    from torch.nn import functional as F
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    ids = build_chat_ids(tokenizer, messages)
    max_seq = model.cfg.max_seq_len
    for _ in range(max_new_tokens):
        window = ids[-max_seq:]
        input_ids = torch.tensor([window], dtype=torch.long)
        logits = model(input_ids, images=image)[0][0, -1]          # 视觉 token 自动拼接
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
        logits = _filter_top_k(logits, top_k)
        logits = _filter_top_p(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1).item()
        ids.append(nxt)
        if nxt == tokenizer.special_ids[EOS]:
            break
    return tokenizer.decode(ids)


@torch.no_grad()
def stream_chat(model: GPT, tokenizer: BytePairTokenizer, messages: List[dict],
                max_new_tokens: int = 128, temperature: float = 0.8,
                top_k: int = 40, top_p: float = 0.9,
                seed: Optional[int] = None):
    """逐 token 生成器：yield 每个解码片段（供 SSE 流式）。"""
    from tinygpt.inference.generate import _filter_top_k, _filter_top_p
    from torch.nn import functional as F
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    ids = build_chat_ids(tokenizer, messages)
    max_seq = model.cfg.max_seq_len
    for _ in range(max_new_tokens):
        window = ids[-max_seq:]
        input_ids = torch.tensor([window], dtype=torch.long)
        logits = model(input_ids)[0][0, -1]
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
        logits = _filter_top_k(logits, top_k)
        logits = _filter_top_p(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1).item()
        ids.append(nxt)
        if nxt == tokenizer.special_ids[EOS]:
            break
        yield tokenizer.decode([nxt])
