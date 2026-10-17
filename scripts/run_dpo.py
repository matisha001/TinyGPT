"""DPO 偏好对齐入口：
  1) 预计算参考模型 logprob 缓存
     python scripts/run_dpo.py --precompute
  2) DPO 训练（策略 = SFT 基座 + LoRA）
     python scripts/run_dpo.py --config configs/dpo.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinygpt.config import TrainStageConfig, load_config
from tinygpt.train.dpo import precompute_ref_logprobs, run_dpo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--metrics", default="logs/dpo_metrics.jsonl")
    ap.add_argument("--precompute", action="store_true",
                    help="只预计算参考模型 logprob 缓存")
    args = ap.parse_args()

    cfg = load_config(args.config, TrainStageConfig)
    if args.precompute:
        precompute_ref_logprobs(cfg)
    else:
        run_dpo(cfg, metrics_path=args.metrics)


if __name__ == "__main__":
    main()
