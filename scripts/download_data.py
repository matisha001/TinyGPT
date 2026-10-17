"""下载小型公开中文数据集并构建训练数据（支持离线回退到内置 demo 语料）。

用法:
  .venv/bin/python scripts/download_data.py --dataset all --max-samples 2000
  .venv/bin/python scripts/download_data.py --dataset sft --force

说明:
  - sft   : shibing624/alpaca-zh（48k 中文指令，研究用途）→ data/datasets/sft_train.jsonl
  - dpo   : liyucheng/zhihu_rlhf_3k（3k 中文偏好对）→ data/datasets/dpo_pairs.jsonl
  - pretrain: demo 语料 + alpaca 文本 → data/corpus/pretrain.txt（作为预训练语料）
  下载走 hf-mirror（中国大陆可达）；失败时自动使用内置 demo 语料兜底。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
HF = "https://hf-mirror.com/datasets"

DATASETS = {
    "sft": {
        "url": f"{HF}/shibing624/alpaca-zh/resolve/main/alpaca_gpt4_data_zh.json",
        "dest": "data/datasets/alpaca_gpt4_data_zh.json",
        "note": "48k 中文指令（instruction/input/output）",
    },
    "dpo": {
        "url": f"{HF}/liyucheng/zhihu_rlhf_3k/resolve/main/zhihu_3k_rlfh.tsv",
        "dest": "data/datasets/zhihu_3k_rlfh.tsv",
        "note": "3k 中文偏好对（prompt/chosen/rejected）",
    },
}

DEMO_CORPUS = ROOT / "data" / "demo" / "demo_corpus.txt"
PRETRAIN_CORPUS = ROOT / "data" / "corpus" / "pretrain.txt"
SFT_OUT = ROOT / "data" / "datasets" / "sft_train.jsonl"
DPO_OUT = ROOT / "data" / "datasets" / "dpo_pairs.jsonl"


def download(name: str, force: bool = False) -> Path:
    info = DATASETS[name]
    dest = ROOT / info["dest"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[skip] {info['dest']} 已存在（--force 可重下）")
        return dest
    print(f"[download] {info['url']}")
    req = Request(info["url"], headers={"User-Agent": "tiny-gpt/0.1"})
    try:
        with urlopen(req, timeout=180) as r, open(dest, "wb") as f:
            f.write(r.read())
    except HTTPError as e:
        raise RuntimeError(f"下载失败 HTTP {e.code}: {info['url']}") from e
    print(f"[ok] {info['dest']} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def build_sft(max_samples: int | None) -> Path:
    raw = download("sft")
    data = json.loads(raw.read_text(encoding="utf-8"))
    if max_samples:
        data = data[:max_samples]
    with open(SFT_OUT, "w", encoding="utf-8") as f:
        for it in data:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[ok] SFT jsonl: {SFT_OUT}（{len(data)} 条）")
    return SFT_OUT


def build_dpo(max_samples: int | None) -> Path:
    raw = download("dpo")
    lines = raw.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split("\t")
    idx = {c: i for i, c in enumerate(header)}
    pairs = []
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) <= max(idx.values()):
            continue
        pairs.append({
            "prompt": cols[idx["prompt"]],
            "chosen": cols[idx["chosen"]],
            "rejected": cols[idx["rejected"]],
        })
        if max_samples and len(pairs) >= max_samples:
            break
    with open(DPO_OUT, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[ok] DPO jsonl: {DPO_OUT}（{len(pairs)} 对）")
    return DPO_OUT


def build_pretrain_corpus(max_chars: int | None) -> Path:
    """预训练语料 = 内置 demo 语料 + alpaca 文本（截断 max_chars）。"""
    PRETRAIN_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    if DEMO_CORPUS.exists():
        parts.append(DEMO_CORPUS.read_text(encoding="utf-8"))
    sft_raw = ROOT / "data" / "datasets" / "alpaca_gpt4_data_zh.json"
    if sft_raw.exists():
        data = json.loads(sft_raw.read_text(encoding="utf-8"))
        for it in data:
            parts.append(it.get("instruction", ""))
            parts.append(it.get("input", ""))
            parts.append(it.get("output", ""))
    text = "\n".join(parts)
    if max_chars:
        text = text[:max_chars]
    PRETRAIN_CORPUS.write_text(text, encoding="utf-8")
    print(f"[ok] 预训练语料: {PRETRAIN_CORPUS}（{len(text) / 1e6:.1f} MB / {len(text):,} 字符）")
    return PRETRAIN_CORPUS


def main() -> None:
    ap = argparse.ArgumentParser(description="下载并构建 tiny-gpt 训练数据")
    ap.add_argument("--dataset", choices=["pretrain", "sft", "dpo", "all"], default="all")
    ap.add_argument("--max-samples", type=int, default=None, help="SFT/DPO 样本截断")
    ap.add_argument("--max-chars", type=int, default=None, help="预训练语料字符截断")
    ap.add_argument("--force", action="store_true", help="强制重新下载")
    args = ap.parse_args()

    try:
        if args.dataset in ("sft", "all"):
            build_sft(args.max_samples)
        if args.dataset in ("dpo", "all"):
            build_dpo(args.max_samples)
        if args.dataset in ("pretrain", "all"):
            build_pretrain_corpus(args.max_chars)
    except RuntimeError as e:
        print(f"[warn] {e} → 继续使用内置 demo 语料兜底")
        if args.dataset in ("pretrain", "all"):
            build_pretrain_corpus(args.max_chars)
        sys.exit(2)


if __name__ == "__main__":
    main()
