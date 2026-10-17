"""多模态数据：图片 + 问答 → batch（文本 ids / 掩码 / 图像张量）。"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
from PIL import Image

from tinygpt.data.sft import BOS, EOS, USER, ASSISTANT, format_prompt
from tinygpt.tokenizer import BytePairTokenizer


class MMData:
    """mm_train.jsonl 每行：{"image": "data/demo/mm/imgs/0.png", "question": "...", "answer": "..."}

    模板：<s><|user|>{question}<|assistant|>{answer}</s>（<|image|> 不出现；
    视觉 token 由模型在序列头部拼接）。
    """

    def __init__(self, jsonl_path: str | Path, tokenizer: BytePairTokenizer,
                 seq_len: int, batch_size: int, image_size: int,
                 seed: int = 42, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.image_size = image_size
        self.rng = random.Random(seed)
        self.samples = self._load(jsonl_path, max_samples)
        self._order = list(range(len(self.samples)))
        self._pos = 0

    def _load(self, path, max_samples) -> List[dict]:
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out.append(d)
                if max_samples and len(out) >= max_samples:
                    break
        return out

    def _encode(self, question: str, answer: str) -> Tuple[List[int], List[int]]:
        prompt_ids = self.tokenizer.encode(format_prompt(question, ""))
        answer_ids = self.tokenizer.encode(answer) + [self.tokenizer.special_ids[EOS]]
        ids = [self.tokenizer.special_ids[BOS]] + prompt_ids + answer_ids
        mask = [0] * (1 + len(prompt_ids)) + [1] * len(answer_ids)
        return ids, mask

    @staticmethod
    def load_image(path: str | Path, size: int) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((size, size))
        arr = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return arr

    def __len__(self) -> int:
        return len(self.samples)

    def iter_batches(self) -> Iterator[dict]:
        pad_id = self.tokenizer.special_ids["<pad>"]
        while True:
            if self._pos == 0 or self._pos >= len(self._order):
                self.rng.shuffle(self._order)
                self._pos = 0
            end = min(self._pos + self.batch_size, len(self._order))
            idx = self._order[self._pos:end]
            self._pos = end
            ids_list, mask_list, imgs = [], [], []
            for i in idx:
                s = self.samples[i]
                ids, mask = self._encode(s["question"], s["answer"])
                ids = ids[: self.seq_len] + [pad_id] * max(0, self.seq_len - len(ids))
                mask = mask[: self.seq_len] + [0] * max(0, self.seq_len - len(mask))
                ids_list.append(ids)
                mask_list.append(mask)
                imgs.append(self.load_image(s["image"], self.image_size))
            yield {
                "input_ids": torch.tensor(ids_list, dtype=torch.long),
                "labels": torch.tensor(ids_list, dtype=torch.long),
                "mask": torch.tensor(mask_list, dtype=torch.long),
                "images": torch.stack(imgs),
            }
