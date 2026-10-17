"""多模态联合微调：ViT（可冻结）+ 投影层 + LLM（可 LoRA）。"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional

import torch

from tinygpt.config import TrainStageConfig, load_config
from tinygpt.data.mm import MMData
from tinygpt.lora import adapter_state_dict, apply_lora, trainable_only_adapters
from tinygpt.model.multimodal import VisionGPT
from tinygpt.tokenizer import BytePairTokenizer
from tinygpt.train.common import build_optimizer, get_lr, load_checkpoint, save_checkpoint, set_seed
from tinygpt.utils.logging import MetricLogger, setup_logger

log = setup_logger("tinygpt.mm")


def build_mm_model(cfg: TrainStageConfig, tok: BytePairTokenizer) -> VisionGPT:
    model_cfg = dataclasses.replace(cfg.model, vocab_size=tok.vocab_size_actual)
    return VisionGPT(model_cfg, cfg.model.vision)


def run_mm(cfg: TrainStageConfig, metrics_path: Optional[str] = None) -> VisionGPT:
    set_seed(cfg.train.seed)
    ml = MetricLogger(metrics_path) if metrics_path else None

    tok = BytePairTokenizer.load(cfg.data.tokenizer_path)
    model = build_mm_model(cfg, tok)
    if cfg.train.init_from and Path(cfg.train.init_from).exists():
        # LLM 部分载入预训练权重；ViT/投影层保持随机初始化（strict=False）
        ckpt = torch.load(cfg.train.init_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=False)
        log.info(f"LLM 部分载入: {cfg.train.init_from}")

    # ViT 冻结开关
    if cfg.model.vision.freeze:
        for p in model.vision_encoder.parameters():
            p.requires_grad_(False)
        log.info("ViT 已冻结（仅训练投影层 + LLM 可训练部分）")

    if cfg.model.use_lora:
        apply_lora(model, cfg.model.lora)
        trainable_only_adapters(model)
        # 投影层始终可训练
        for p in model.vision_projector.parameters():
            p.requires_grad_(True)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"可训练参数: {n / 1e3:.1f}K（ViT 全参 {model.n_vision_params() / 1e3:.1f}K）")

    data = MMData(cfg.data.mm_path, tok, seq_len=cfg.data.seq_len,
                  batch_size=cfg.data.batch_size, image_size=cfg.model.vision.image_size,
                  seed=cfg.train.seed, max_samples=cfg.data.max_samples)
    log.info(f"多模态样本数: {len(data)}（图像 {cfg.model.vision.image_size}²，"
             f"patch {cfg.model.vision.patch_size}，视觉token "
             f"{cfg.model.vision.num_vision_tokens}）")

    opt = build_optimizer(model, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps = cfg.train.steps
    warmup = cfg.train.warmup_steps
    grad_accum = cfg.train.grad_accum
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = iter(data.iter_batches())
    model.train()
    micro = 0
    for step in range(steps):
        lr = get_lr(step, warmup, steps, cfg.train.lr, cfg.train.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = next(batches)
        _, loss = model(batch["input_ids"], images=batch["images"],
                        targets=batch["labels"], loss_mask=batch["mask"])
        loss = loss / grad_accum
        loss.backward()
        micro += 1
        if micro == grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            micro = 0
        if (step + 1) % cfg.train.log_every == 0:
            log.info(f"step {step + 1}/{steps} loss={loss.item() * grad_accum:.4f} lr={lr:.2e}")
            if ml:
                ml.log(phase="mm", step=step + 1, loss=loss.item() * grad_accum, lr=lr)

    save_checkpoint(out_dir / "mm.pt", model, opt, steps, dataclasses.asdict(cfg))
    log.info(f"多模态 checkpoint 已保存: {out_dir / 'mm.pt'}")
    if ml:
        ml.close()
    return model


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/mm_finetune.yaml"
    run_mm(load_config(cfg_path, TrainStageConfig))
