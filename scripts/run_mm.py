"""多模态微调入口：python scripts/run_mm.py --config configs/mm_finetune.yaml"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinygpt.config import TrainStageConfig, load_config
from tinygpt.train.mm import run_mm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mm_finetune.yaml")
    ap.add_argument("--metrics", default="logs/mm_metrics.jsonl")
    args = ap.parse_args()
    run_mm(load_config(args.config, TrainStageConfig), metrics_path=args.metrics)


if __name__ == "__main__":
    main()
