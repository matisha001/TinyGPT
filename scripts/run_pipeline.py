"""全链路流水线：依次运行 预训练 → SFT → LoRA → DPO 预计算 → DPO。

用法：python scripts/run_pipeline.py [--stages pretrain,sft,lora,dpo]
注意：各阶段依赖前置 checkpoint，必须按顺序执行。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinygpt.config import TrainStageConfig, load_config
from tinygpt.train.pretrain import run_pretrain
from tinygpt.train.sft import run_sft
from tinygpt.train.dpo import precompute_ref_logprobs, run_dpo
from tinygpt.utils.logging import setup_logger

log = setup_logger("tinygpt.pipeline")

STAGES = ["pretrain", "sft", "lora", "dpo"]
STAGE_CONFIGS = {
    "pretrain": "configs/pretrain.yaml",
    "sft": "configs/sft.yaml",
    "lora": "configs/lora_sft.yaml",
    "dpo": "configs/dpo.yaml",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGE_CONFIGS:
            raise SystemExit(f"未知阶段: {s}（可选 {STAGES}）")

    t0 = time.time()
    for stage in stages:
        log.info(f"===== 阶段 {stage} 开始 =====")
        cfg = load_config(STAGE_CONFIGS[stage], TrainStageConfig)
        t1 = time.time()
        if stage == "pretrain":
            run_pretrain(cfg, metrics_path=f"logs/{stage}_metrics.jsonl",
                         eval_prompts=["机器学习", "深度学习", "豆包"])
        elif stage == "sft":
            run_sft(cfg, metrics_path=f"logs/{stage}_metrics.jsonl")
        elif stage == "lora":
            run_sft(cfg, metrics_path=f"logs/{stage}_metrics.jsonl")
        elif stage == "dpo":
            if not Path(cfg.train.ref_logprobs_path).exists():
                precompute_ref_logprobs(cfg)
            run_dpo(cfg, metrics_path=f"logs/{stage}_metrics.jsonl")
        log.info(f"===== 阶段 {stage} 完成，耗时 {time.time() - t1:.1f}s =====")
    log.info(f"全链路完成，总耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
