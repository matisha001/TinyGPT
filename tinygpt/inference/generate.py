"""自回归生成：temperature / top-k / top-p 采样。"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from tinygpt.model import GPT
from tinygpt.tokenizer import BytePairTokenizer


def _filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    v, _ = torch.topk(logits, k)
    logits[logits < v[..., -1, None]] = float("-inf")
    return logits


def _filter_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0 or p >= 1.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    sorted_p, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    keep = cum <= p
    keep[..., 0] = True                     # 至少保留一个
    idx = torch.arange(logits.size(-1), device=logits.device)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[..., sorted_idx[keep]] = True
    logits[~mask] = float("-inf")
    return logits


@torch.no_grad()
def _generate_from_ids(model: GPT, tokenizer: BytePairTokenizer, ids: List[int],
                       max_new_tokens: int = 64, temperature: float = 0.8,
                       top_k: int = 40, top_p: float = 0.9,
                       eos_id: Optional[int] = None,
                       seed: Optional[int] = None,
                       decode_from: Optional[int] = None,
                       min_new_tokens: int = 0) -> str:
    """从已有 id 序列继续自回归采样。decode_from: 只解码该下标之后的新 token。

    min_new_tokens: 前 N 个 token 抑制 EOS（防止 demo 级模型过早收尾）。
    """
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    start = len(ids) if decode_from is None else decode_from
    max_seq = model.cfg.max_seq_len
    for _ in range(max_new_tokens):
        window = ids[-max_seq:]
        input_ids = torch.tensor([window], dtype=torch.long)
        logits = model(input_ids)[0][0, -1]                     # [V]
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
        new_n = len(ids) - start
        if eos_id is not None and new_n < min_new_tokens:
            logits[eos_id] = -float("inf")                       # EOS 抑制
        logits = _filter_top_k(logits, top_k)
        logits = _filter_top_p(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1).item()
        ids.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break
    return tokenizer.decode(ids[start:])


@torch.no_grad()
def generate(model: GPT, tokenizer: BytePairTokenizer, prompt: str,
             max_new_tokens: int = 64, temperature: float = 0.8,
             top_k: int = 40, top_p: float = 0.9,
             eos_id: Optional[int] = None, seed: Optional[int] = None) -> str:
    """给定 prompt 续写文本（CPU 友好：逐 token 前向）。返回 prompt + 续写。"""
    ids: List[int] = tokenizer.encode(prompt)
    return _generate_from_ids(model, tokenizer, ids, max_new_tokens,
                              temperature, top_k, top_p, eos_id, seed,
                              decode_from=0)
