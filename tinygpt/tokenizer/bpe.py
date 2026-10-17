"""字节级 BPE（Byte-level BPE）分词器：训练 / 编码 / 解码 / 词表读写。

设计要点
--------
1. 输入文本先转 UTF-8 字节序列，在**字节层面**做 merge —— 天然兼容中文、英文、
   生僻字、emoji，不存在"未登录字"问题。
2. ID 布局（线性，可序列化）：
   - [0, n_special)                 特殊 token（<s> </s> <pad> <unk> <|user|> ...）
   - [n_special, n_special+256)     原始字节（byte value = id - n_special）
   - [n_special+256, ...)            按训练顺序递增的 merge 结果
3. 训练使用"优先队列 + 双向链表"精确实现：每轮取全局最高频字节对合并，
   只更新受影响邻域，复杂度 ≈ O(语料规模 × 常数)，适合在 CPU 上训练。
"""
from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

BPE_VERSION = 1


class BytePairTokenizer:
    def __init__(self, vocab_size: int = 8192, special_tokens: Optional[List[str]] = None):
        n_special = len(special_tokens or [])
        if vocab_size < n_special + 256:
            raise ValueError(f"vocab_size={vocab_size} 过小，至少需要 {n_special + 256}（特殊token+字节）")
        self.vocab_size = vocab_size
        self.special_tokens: List[str] = list(special_tokens or [])
        self.n_special = n_special
        self.byte_base = n_special
        self.merge_base = n_special + 256
        self._special_to_id = {t: i for i, t in enumerate(self.special_tokens)}
        self._id_to_special = {i: t for i, t in enumerate(self.special_tokens)}
        # 训练结果
        self._merges: List[Tuple[int, int]] = []          # 按训练顺序的 (左, 右) 合并对
        self._rank: Dict[Tuple[int, int], int] = {}        # 合并对 -> merge id（rank 越小越优先）
        self._expansion: Dict[int, List[int]] = {}         # token id -> 字节序列
        for i in range(256):
            self._expansion[self.byte_base + i] = [i]

    # ------------------------------------------------------------------
    # 基础属性
    # ------------------------------------------------------------------
    @property
    def n_merges(self) -> int:
        return len(self._merges)

    @property
    def vocab_size_actual(self) -> int:
        return self.n_special + 256 + self.n_merges

    def id_of(self, special: str) -> int:
        if special not in self._special_to_id:
            raise KeyError(f"非特殊 token: {special!r}")
        return self._special_to_id[special]

    # ------------------------------------------------------------------
    # 训练（优先队列 + 链表，精确 BPE）
    # ------------------------------------------------------------------
    def train(self, texts: Iterable[str], min_freq: int = 2, max_tokens: Optional[int] = None,
              verbose: bool = True) -> "BytePairTokenizer":
        # 1. 语料 → 扁平字节 id 数组 + 双向链表
        toks: List[int] = []
        nxt: List[int] = []
        prv: List[int] = []
        heads: List[int] = []          # 每个序列的头位置（仅用于统计，可无）
        budget = max_tokens or (1 << 62)
        for text in texts:
            if len(toks) >= budget:
                break
            seq = [self.byte_base + b for b in text.encode("utf-8")]
            if not seq:
                continue
            remaining = budget - len(toks)
            if len(seq) > remaining:
                seq = seq[:remaining]
            start = len(toks)
            toks.extend(seq)
            nxt.extend(list(range(start + 1, start + len(seq))) + [-1])
            prv.extend([-1] + list(range(start, start + len(seq) - 1)))
            heads.append(start)
            if len(toks) >= budget:
                break

        target_merges = self.vocab_size - self.merge_base
        if len(toks) == 0:
            raise ValueError("语料为空，无法训练分词器")

        # 2. 统计初始相邻对
        pair_pos: Dict[Tuple[int, int], List[int]] = {}   # 对 -> 左端点位置列表
        counts: Dict[Tuple[int, int], int] = {}
        for i in range(len(toks) - 1):
            p = (toks[i], toks[i + 1])
            counts[p] = counts.get(p, 0) + 1
            pair_pos.setdefault(p, []).append(i)
        heap: List[Tuple[int, int, int]] = []
        for (a, b), c in counts.items():
            heapq.heappush(heap, (-c, a, b))              # 负频率 → 最小堆即最高频

        # 3. 迭代合并
        while self.n_merges < target_merges:
            # 弹出有效项（跳过计数过期的旧条目）
            pair: Optional[Tuple[int, int, int]] = None
            while heap:
                neg_c, a, b = heap[0]
                cur = counts.get((a, b), 0)
                if cur > 0 and -neg_c == cur and cur >= min_freq:
                    heapq.heappop(heap)
                    pair = (a, b, cur)
                    break
                heapq.heappop(heap)
            if pair is None:
                break  # 没有满足 min_freq 的对
            a, b, c = pair

            new_id = self.merge_base + self.n_merges
            self._merges.append((a, b))
            self._rank[(a, b)] = new_id
            # 新 token 的字节展开 = 左 + 右
            self._expansion[new_id] = self._expansion[a] + self._expansion[b]

            # 合并该对的所有有效出现位置
            occs = pair_pos.get((a, b), [])
            for p in occs:
                q = nxt[p]
                if toks[p] != a or q == -1 or toks[q] != b:
                    continue  # 该位置已失效（被更早的合并改写）
                r = nxt[q]
                # --- 拆除旧关系 ---
                left_p = prv[p]
                self._del_pair(pair_pos, counts, heap, toks, prv, nxt, left_p, p)
                self._del_pair(pair_pos, counts, heap, toks, prv, nxt, p, q)
                if r != -1:
                    self._del_pair(pair_pos, counts, heap, toks, prv, nxt, q, r)
                # --- 合并 p+q 为新 token ---
                toks[p] = new_id
                nxt[p] = r
                if r != -1:
                    prv[r] = p
                toks[q] = -1  # 标记死亡
                # --- 建立新邻接关系 ---
                if left_p != -1:
                    self._add_pair(pair_pos, counts, heap, toks, left_p, p)
                if r != -1:
                    self._add_pair(pair_pos, counts, heap, toks, p, r)

        if verbose:
            print(f"[BPE] 语料 tokens={len(toks):,} merges={self.n_merges:,} "
                  f"vocab={self.vocab_size_actual:,} (目标 {self.vocab_size:,})")
        return self

    @staticmethod
    def _del_pair(pair_pos, counts, heap, toks, prv, nxt, i, j):
        if i == -1 or j == -1 or toks[i] == -1 or toks[j] == -1:
            return
        p = (toks[i], toks[j])
        if counts.get(p, 0) > 0:
            counts[p] -= 1
            if counts[p] == 0:
                counts.pop(p, None)
        lst = pair_pos.get(p)
        if lst is not None:
            # 惰性清理：保留但由有效性检查跳过；计数 0 时清空
            pass

    @staticmethod
    def _add_pair(pair_pos, counts, heap, toks, i, j):
        if i == -1 or j == -1 or toks[i] == -1 or toks[j] == -1:
            return
        p = (toks[i], toks[j])
        counts[p] = counts.get(p, 0) + 1
        pair_pos.setdefault(p, []).append(i)
        heapq.heappush(heap, (-counts[p], p[0], p[1]))

    # ------------------------------------------------------------------
    # 编码 / 解码
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """文本 → token id 列表（严格按 rank 顺序合并，结果确定）。"""
        ids = [self.byte_base + b for b in text.encode("utf-8")]
        while True:
            best: Optional[Tuple[int, int]] = None   # (rank, pos)
            for i in range(len(ids) - 1):
                r = self._rank.get((ids[i], ids[i + 1]))
                if r is not None and (best is None or r < best[0]):
                    best = (r, i)
                    if r == self.merge_base:  # 最低可能 rank
                        break
            if best is None:
                break
            _, pos = best
            new_id = self._rank[(ids[pos], ids[pos + 1])]
            ids = ids[:pos] + [new_id] + ids[pos + 2:]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """token id → 文本（字节级展开，保证无损回环；特殊 token 原样输出）。"""
        out = bytearray()
        for i in ids:
            if i in self._expansion:
                out.extend(self._expansion[i])
            elif i in self._id_to_special:
                out.extend(self._id_to_special[i].encode("utf-8"))
            else:
                raise ValueError(f"未知 token id: {i}")
        return out.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": BPE_VERSION,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "byte_base": self.byte_base,
            "merge_base": self.merge_base,
            "merges": [list(m) for m in self._merges],
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "BytePairTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != BPE_VERSION:
            raise ValueError(f"分词器版本不兼容: {data.get('version')}")
        tok = cls(vocab_size=data["vocab_size"], special_tokens=data["special_tokens"])
        for a, b in data["merges"]:
            new_id = tok.merge_base + len(tok._merges)
            tok._merges.append((a, b))
            tok._rank[(a, b)] = new_id
            tok._expansion[new_id] = tok._expansion[a] + tok._expansion[b]
        return tok
