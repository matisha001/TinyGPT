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
