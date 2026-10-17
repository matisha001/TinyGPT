"""预训练数据：语料 → 打包 token 块 → 随机 batch（输入/标签相邻移位）。"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch

from tinygpt.tokenizer import BytePairTokenizer

ENCODE_CHUNK = 20_000          # 分块编码，避免长文本内存/时间峰值


def encode_corpus(corpus_path: str | Path, tokenizer: BytePairTokenizer,
                  max_chars: Optional[int] = None) -> List[int]:
    text = Path(corpus_path).read_text(encoding="utf-8")
    if max_chars:
        text = text[:max_chars]
    ids: List[int] = []
    for i in range(0, len(text), ENCODE_CHUNK):
        ids.extend(tokenizer.encode(text[i:i + ENCODE_CHUNK]))
    return ids


class PretrainData:
    """打包训练：语料切成 (seq_len+1) 长的连续块，块内相邻 token 构成输入/标签。

    与逐样本 padding 相比，打包训练几乎没有 padding 浪费，适合 CPU 小模型。
    """

    def __init__(self, corpus_path: str | Path, tokenizer: BytePairTokenizer,
                 seq_len: int, batch_size: int, seed: int = 42,
                 max_samples: Optional[int] = None, eval_ratio: float = 0.01):
        self.seq_len = seq_len
        self.batch_size = batch_size
        ids = encode_corpus(corpus_path, tokenizer, max_samples)
        n = len(ids) // (seq_len + 1)
        if n == 0:
            raise ValueError(f"语料 token 数不足一个块（需要 ≥ {seq_len + 1}）")
        arr = np.array(ids[: n * (seq_len + 1)], dtype=np.int64).reshape(n, seq_len + 1)
        n_eval = max(1, int(n * eval_ratio))
        self.eval_blocks = torch.from_numpy(arr[:n_eval])     # 固定评估块（不参与训练）
        self.train_blocks = torch.from_numpy(arr[n_eval:])
        self.rng = random.Random(seed)
        self._order = list(range(len(self.train_blocks)))
        self._pos = 0

    def __len__(self) -> int:
        return len(self.train_blocks)

    def n_tokens(self) -> int:
        return int(self.train_blocks.numel()) + int(self.eval_blocks.numel())

    def _next_epoch_order(self) -> None:
        self.rng.shuffle(self._order)
        self._pos = 0

    def iter_batches(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """每 epoch 打乱一次；返回 (input_ids, targets)。"""
        while True:
            if self._pos == 0 or self._pos >= len(self._order):
                self._next_epoch_order()
            end = min(self._pos + self.batch_size, len(self._order))
            idx = self._order[self._pos:end]
            self._pos = end
            blk = self.train_blocks[idx]
            yield blk[:, :-1], blk[:, 1:]

    def eval_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        blk = self.eval_blocks
        return blk[:, :-1], blk[:, 1:]
