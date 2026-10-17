"""SFT 数据：指令样本 → 问答段掩码 + 打包训练。"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch

from tinygpt.tokenizer import BytePairTokenizer

BOS = "<s>"
EOS = "</s>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"


def format_prompt(instruction: str, input_text: str = "") -> str:
    if input_text.strip():
        return f"{USER}{instruction}\n{input_text}{ASSISTANT}"
    return f"{USER}{instruction}{ASSISTANT}"


class SFTData:
    """指令数据打包：每条样本 = prompt_ids(掩码0) + answer_ids(掩码1，含 </s>)。
    多条样本顺序打包进 seq_len 的块；块内样本以 <s> 开头分隔。
    """

    def __init__(self, jsonl_path: str | Path, tokenizer: BytePairTokenizer,
                 seq_len: int, batch_size: int, seed: int = 42,
                 max_samples: Optional[int] = None):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.special_ids[BOS]
        self.pad_id = tokenizer.special_ids["<pad>"]
        self.rng = random.Random(seed)
        samples = self._load(jsonl_path, max_samples)
        self.blocks = self._pack(samples)
        self._order = list(range(len(self.blocks)))
        self._pos = 0

    def _load(self, path, max_samples) -> List[Tuple[str, str, str]]:
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out.append((d["instruction"], d.get("input", ""), d["output"]))
                if max_samples and len(out) >= max_samples:
                    break
        return out

    def _encode_sample(self, instruction: str, input_text: str,
                       answer: str) -> Tuple[List[int], List[int], List[int]]:
        prompt_ids = self.tokenizer.encode(format_prompt(instruction, input_text))
        answer_ids = self.tokenizer.encode(answer) + [self.tokenizer.special_ids[EOS]]
        ids = [self.bos_id] + prompt_ids + answer_ids
        mask = [0] * (1 + len(prompt_ids)) + [1] * len(answer_ids)
        return ids, ids, mask          # 标签=输入（自回归移位）

    def _pack(self, samples) -> List[torch.Tensor]:
        """顺序打包：每条样本可跨块（块尾截断），块内以 <s> 开头。"""
        blocks_ids, blocks_labels, blocks_mask = [], [], []
        cur_ids, cur_mask = [], []
        for instruction, input_text, answer in samples:
            ids, labels, mask = self._encode_sample(instruction, input_text, answer)
            for t in range(0, len(ids), self.seq_len):
                seg = ids[t:t + self.seq_len]
                seg_mask = mask[t:t + self.seq_len]
                while len(seg) < self.seq_len:      # 最后一段补 <pad>（掩码0）
                    seg.append(self.pad_id)
                    seg_mask.append(0)
                blocks_ids.append(torch.tensor(seg, dtype=torch.long))
                blocks_mask.append(torch.tensor(seg_mask, dtype=torch.long))
        return list(zip(blocks_ids, blocks_mask))

    def __len__(self) -> int:
        return len(self.blocks)

    def iter_batches(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        while True:
            if self._pos == 0 or self._pos >= len(self._order):
                self.rng.shuffle(self._order)
                self._pos = 0
            end = min(self._pos + self.batch_size, len(self._order))
            idx = self._order[self._pos:end]
            self._pos = end
            ids = torch.stack([self.blocks[i][0] for i in idx])
            masks = torch.stack([self.blocks[i][1] for i in idx])
            yield ids[:, :-1], ids[:, 1:], masks[:, 1:]
