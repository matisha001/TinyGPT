"""训练字节级 BPE 中文分词器。

用法:
  .venv/bin/python scripts/train_tokenizer.py --config configs/tokenizer.yaml
产物: data/tokenizer/tokenizer.json（含词表/合并规则/特殊 token）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinygpt.config import TokenizerConfig, load_config
from tinygpt.tokenizer import BytePairTokenizer
from tinygpt.utils.logging import setup_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tokenizer.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config, TokenizerConfig)
    logger = setup_logger()

    corpus_path = Path(cfg.data.corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"语料不存在: {corpus_path}（先运行 scripts/download_data.py，或修改 config 路径）"
        )
    text = corpus_path.read_text(encoding="utf-8")
    logger.info(f"语料: {corpus_path}（{len(text) / 1e6:.2f} MB / {len(text):,} 字符）")

    tok = BytePairTokenizer(vocab_size=cfg.vocab_size, special_tokens=cfg.special_tokens)
    tok.train([text], min_freq=cfg.data.min_freq, max_tokens=cfg.data.max_tokens)

    out = Path(cfg.out_dir) / "tokenizer.json"
    tok.save(out)
    logger.info(f"分词器已保存: {out}（vocab={tok.vocab_size_actual}，merges={tok.n_merges}）")

    # 回环验证 + 样例输出
    samples = ["豆包是一款智能助手。", "Hello world，你好世界！🚀", "机器学习与深度学习都是人工智能的分支。"]
    for s in samples:
        ids = tok.encode(s)
        back = tok.decode(ids)
        assert back == s, f"回环失败: {s!r}"
        pieces = " | ".join(tok.decode([i]) for i in ids)
        logger.info(f"  [{s[:20]}] → {len(ids)} tokens: {pieces}")

    # 元信息
    meta = {
        "corpus_path": str(corpus_path),
        "corpus_chars": len(text),
        "vocab_size_target": cfg.vocab_size,
        "vocab_size_actual": tok.vocab_size_actual,
        "n_merges": tok.n_merges,
        "n_special": tok.n_special,
        "special_tokens": cfg.special_tokens,
    }
    meta_path = Path(cfg.out_dir) / "tokenizer_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"元信息: {meta_path}")


if __name__ == "__main__":
    main()
