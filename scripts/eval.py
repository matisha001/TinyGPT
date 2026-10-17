"""预训练效果评估：加载 checkpoint → 评估 loss/PPL + 若干提示词续写。

python scripts/eval.py --ckpt checkpoints/pretrain/base.pt
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tinygpt.config import TrainStageConfig, load_config
from tinygpt.data.pretrain import PretrainData
from tinygpt.model import GPT
from tinygpt.tokenizer import BytePairTokenizer
from tinygpt.train.common import load_checkpoint, set_seed
from tinygpt.inference.generate import generate
from tinygpt.utils.logging import setup_logger

log = setup_logger("tinygpt.eval")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("--ckpt", default="checkpoints/pretrain/base.pt")
    ap.add_argument("--prompt", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, TrainStageConfig)
    set_seed(cfg.train.seed)

    tok = BytePairTokenizer.load(cfg.data.tokenizer_path)
    model = _restore_model(cfg, tok)
    load_checkpoint(args.ckpt, model)
    log.info(f"模型参数量: {model.n_params() / 1e6:.2f}M")

    data = PretrainData(cfg.data.corpus_path, tok, seq_len=cfg.data.seq_len,
                        batch_size=cfg.data.batch_size, seed=cfg.train.seed,
                        max_samples=cfg.data.max_samples, eval_ratio=cfg.data.eval_ratio)
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(8):
            x, y = data.eval_batch()
            _, loss = model(x[:1], targets=y[:1])
            losses.append(loss.item())
    avg = sum(losses) / len(losses)
    log.info(f"[eval] loss={avg:.4f} ppl={math.exp(avg):.2f}")

    prompts = list(args.prompt)
    pf = Path("data/demo/eval_prompts.txt")
    if pf.exists():
        prompts += [l.strip() for l in pf.read_text(encoding="utf-8").splitlines() if l.strip()]
    for p in prompts[:5]:
        s = generate(model, tok, p, max_new_tokens=48, temperature=0.8)
        log.info(f"[sample] {p} → {s}")


def _restore_model(cfg: TrainStageConfig, tok: BytePairTokenizer) -> GPT:
    import dataclasses
    return GPT(dataclasses.replace(cfg.model, vocab_size=tok.vocab_size_actual))


if __name__ == "__main__":
    main()
