"""SFT 入口：python scripts/run_sft.py --config configs/sft.yaml（全量）
                            --config configs/lora_sft.yaml（LoRA）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinygpt.config import TrainStageConfig, load_config
from tinygpt.train.sft import run_sft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--metrics", default="logs/sft_metrics.jsonl")
    ap.add_argument("--prompt", action="append", default=[], help="评估问答提示词（可多次）")
    ap.add_argument("--prompts-file", default="data/demo/sft_eval_prompts.txt")
    args = ap.parse_args()

    cfg = load_config(args.config, TrainStageConfig)
    prompts = list(args.prompt)
    pf = Path(args.prompts_file)
    if pf.exists():
        prompts += [l.strip() for l in pf.read_text(encoding="utf-8").splitlines() if l.strip()]
    run_sft(cfg, metrics_path=args.metrics, eval_prompts=prompts or None)


if __name__ == "__main__":
    main()
