"""DPO 数据：偏好对（prompt/chosen/rejected）→ 编码 + 回答段掩码。"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch

from tinygpt.data.sft import BOS, EOS, USER, ASSISTANT, format_prompt
from tinygpt.tokenizer import BytePairTokenizer


class DPOData:
    """每条样本：chosen / rejected 两条完整对话序列（均含 prompt），
    掩码标记回答段；用于 logprob 与 DPO 损失计算。
    """

    def __init__(self, jsonl_path: str | Path, tokenizer: BytePairTokenizer,
                 seq_len: int, batch_size: int, seed: int = 42,
                 max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.pairs = self._load(jsonl_path, max_samples)
        self._order = list(range(len(self.pairs)))
        self._pos = 0

    def _load(self, path, max_samples) -> List[dict]:
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out.append({"prompt": d["prompt"], "chosen": d["chosen"],
                            "rejected": d["rejected"]})
                if max_samples and len(out) >= max_samples:
                    break
        return out

    def _seq(self, prompt: str, answer: str) -> Tuple[List[int], List[int]]:
        """返回 (ids, mask)：<s>+prompt+answer+</s>，掩码只在回答段。"""
        prompt_ids = self.tokenizer.encode(format_prompt(prompt, ""))
        answer_ids = self.tokenizer.encode(answer) + [self.tokenizer.special_ids[EOS]]
        ids = [self.tokenizer.special_ids[BOS]] + prompt_ids + answer_ids
        mask = [0] * (1 + len(prompt_ids)) + [1] * len(answer_ids)
        return ids[: self.seq_len], mask[: self.seq_len]

    def _pad(self, ids: List[int], mask: List[int]):
        pad_id = self.tokenizer.special_ids["<pad>"]
        ids = ids + [pad_id] * (self.seq_len - len(ids))
        mask = mask + [0] * (self.seq_len - len(mask))
        return ids, mask

    def __len__(self) -> int:
        return len(self.pairs)

    def iter_batches(self) -> Iterator[dict]:
        """返回 batch：{chosen_ids, chosen_mask, rejected_ids, rejected_mask}。"""
        while True:
            if self._pos == 0 or self._pos >= len(self._order):
                self.rng.shuffle(self._order)
                self._pos = 0
            end = min(self._pos + self.batch_size, len(self._order))
            idx = self._order[self._pos:end]
            self._pos = end
            c_ids, c_mask, r_ids, r_mask = [], [], [], []
            for i in idx:
                p = self.pairs[i]
                cid, cm = self._pad(*self._seq(p["prompt"], p["chosen"]))
                rid, rm = self._pad(*self._seq(p["prompt"], p["rejected"]))
                c_ids.append(cid); c_mask.append(cm)
                r_ids.append(rid); r_mask.append(rm)
            yield {
                "chosen_ids": torch.tensor(c_ids, dtype=torch.long),
                "chosen_mask": torch.tensor(c_mask, dtype=torch.long),
                "rejected_ids": torch.tensor(r_ids, dtype=torch.long),
                "rejected_mask": torch.tensor(r_mask, dtype=torch.long),
            }
