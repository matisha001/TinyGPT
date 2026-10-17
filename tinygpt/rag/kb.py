"""RAG 检索增强：字符 bigram + TF-IDF 知识库检索。

管线：知识库文档 → 字符 bigram 词表 + TF-IDF 向量化 → 查询相似度排序 →
上下文注入（prompt 前缀）。同时通过 register_tool 暴露为 search_kb 工具。
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tinygpt.tools.safe_exec import register_tool

BIGRAM_RE = re.compile(r"[\s\S]")


def char_bigrams(text: str) -> List[str]:
    """字符 bigram：'abc' → ['ab', 'bc']；不足两字符时退化为单字符。"""
    chars = list(text)
    if len(chars) < 2:
        return chars or [""]
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


class TFIDFIndex:
    """字符 bigram TF-IDF 检索索引（从零实现，不依赖 sklearn）。"""

    def __init__(self) -> None:
        self.docs: List[dict] = []              # [{"id": int, "text": str, "meta": dict}]
        self.vocab: List[str] = []
        self._tf: List[Dict[str, float]] = []
        self._idf: Dict[str, float] = {}

    def add_document(self, text: str, meta: Optional[dict] = None) -> int:
        doc_id = len(self.docs)
        self.docs.append({"id": doc_id, "text": text, "meta": meta or {}})
        return doc_id

    def build(self) -> None:
        """统计词表、文档 TF 与 IDF。"""
        dfs: Dict[str, int] = {}
        self._tf = []
        for d in self.docs:
            grams = char_bigrams(d["text"])
            tf: Dict[str, float] = {}
            for g in grams:
                tf[g] = tf.get(g, 0.0) + 1.0
            total = max(len(grams), 1)
            self._tf.append({g: c / total for g, c in tf.items()})
            for g in tf:
                dfs[g] = dfs.get(g, 0) + 1
        self.vocab = sorted(dfs)
        n = len(self.docs)
        self._idf = {g: math.log((1 + n) / (1 + dfs[g])) + 1.0 for g in self.vocab}

    def _vec(self, grams: Sequence[str]) -> Dict[str, float]:
        tf: Dict[str, float] = {}
        for g in grams:
            if g in self._idf:
                tf[g] = tf.get(g, 0.0) + 1.0
        total = max(len(grams), 1)
        return {g: c / total * self._idf[g] for g, c in tf.items()}

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        keys = set(a) & set(b)
        dot = sum(a[k] * b[k] for k in keys)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        qv = self._vec(char_bigrams(query))
        scored = [(self._cosine(qv, self._tf[i]), self.docs[i])
                  for i in range(len(self.docs))]
        scored.sort(key=lambda x: -x[0])
        return [{"score": s, **d} for s, d in scored[:top_k] if s > 0]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "docs": self.docs, "vocab": self.vocab,
            "tf": self._tf, "idf": self._idf,
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TFIDFIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        idx = cls()
        idx.docs = data["docs"]
        idx.vocab = data["vocab"]
        idx._tf = data["tf"]
        idx._idf = data["idf"]
        return idx


class KnowledgeBase:
    """知识库：入库 → 构建索引 → search_kb 工具 → 上下文注入。"""

    def __init__(self, index_path: str | Path = "data/kb_index/index.json"):
        self.index_path = Path(index_path)
        self.index: Optional[TFIDFIndex] = None

    def load_or_build(self, docs: Optional[List[dict]] = None) -> "KnowledgeBase":
        if self.index_path.exists():
            self.index = TFIDFIndex.load(self.index_path)
        else:
            idx = TFIDFIndex()
            for d in docs or []:
                idx.add_document(d["text"], d.get("meta"))
            idx.build()
            idx.save(self.index_path)
            self.index = idx
        register_tool("search_kb", lambda query, top_k=3: self.search(query, top_k))
        return self

    def add_and_rebuild(self, text: str, meta: Optional[dict] = None) -> None:
        self.index.add_document(text, meta)
        self.index.build()
        self.index.save(self.index_path)

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        assert self.index is not None, "索引未构建"
        return self.index.search(query, top_k)

    def build_prompt(self, query: str, top_k: int = 3,
                     template: str = "基于以下资料回答问题。\n{context}\n问题：{query}") -> str:
        hits = self.search(query, top_k)
        context = "\n".join(f"[{h['id']}] {h['text']}" for h in hits) or "（无相关资料）"
        return template.format(context=context, query=query)
