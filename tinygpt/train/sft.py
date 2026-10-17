"""SFT 监督微调：指令数据 + 回答段掩码损失 + 打包训练；支持全量 / LoRA。"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional

import torch

from tinygpt.config import TrainStageConfig, load_config
from tinygpt.data.sft import SFTData, format_prompt
from tinygpt.lora import (adapter_state_dict, apply_lora, merge_lora,
                          snapshot_weights, trainable_only_adapters)
from tinygpt.model import GPT
from tinygpt.tokenizer import BytePairTokenizer
from tinygpt.train.common import build_optimizer, get_lr, load_checkpoint, save_checkpoint, set_seed
from tinygpt.utils.logging import MetricLogger, setup_logger

log = setup_logger("tinygpt.sft")


def build_model_for_tokenizer(cfg: TrainStageConfig, tok: BytePairTokenizer) -> GPT:
    model_cfg = dataclasses.replace(cfg.model, vocab_size=tok.vocab_size_actual)
    return GPT(model_cfg)


@torch.no_grad()
def chat_generate(model: GPT, tok: BytePairTokenizer, prompt: str,
                  max_new_tokens: int = 64, temperature: float = 0.8) -> str:
    from tinygpt.inference.generate import generate
    return generate(model, tok, format_prompt(prompt), max_new_tokens=max_new_tokens,
                    temperature=temperature, eos_id=tok.special_ids["</s>"])


def run_sft(cfg: TrainStageConfig, metrics_path: Optional[str] = None,
            eval_prompts: Optional[list] = None) -> GPT:
    set_seed(cfg.train.seed)
    ml = MetricLogger(metrics_path) if metrics_path else None

    tok = BytePairTokenizer.load(cfg.data.tokenizer_path)
    model = build_model_for_tokenizer(cfg, tok)
    if cfg.train.init_from and Path(cfg.train.init_from).exists():
        load_checkpoint(cfg.train.init_from, model)
        log.info(f"从预训练权重初始化: {cfg.train.init_from}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if cfg.model.use_lora:
        apply_lora(model, cfg.model.lora)
        trainable_only_adapters(model)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"LoRA 启用（rank={cfg.model.lora.rank}, α={cfg.model.lora.alpha}, "
                 f"targets={cfg.model.lora.targets}）→ 可训练参数 {n_trainable / 1e3:.1f}K")
    else:
        log.info(f"全量微调 → 可训练参数 {n_trainable / 1e6:.2f}M")

    data = SFTData(cfg.data.sft_path, tok, seq_len=cfg.data.seq_len,
                   batch_size=cfg.data.batch_size, seed=cfg.train.seed,
                   max_samples=cfg.data.max_samples)
    log.info(f"SFT 打包块数: {len(data)}")

    opt = build_optimizer(model, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps = cfg.train.steps
    warmup = cfg.train.warmup_steps
    grad_accum = cfg.train.grad_accum

    start_step = 0
    if cfg.train.resume and Path(cfg.train.resume).exists():
        ckpt = load_checkpoint(cfg.train.resume, model, opt)
        start_step = ckpt.get("step", 0)

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = iter(data.iter_batches())
    model.train()
    micro = 0
    for step in range(start_step, steps):
        lr = get_lr(step, warmup, steps, cfg.train.lr, cfg.train.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y, mask = next(batches)
        _, loss = model(x, targets=y, loss_mask=mask)
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
                ml.log(phase="sft", step=step + 1, loss=loss.item() * grad_accum, lr=lr)
        if (step + 1) % cfg.train.eval_every == 0:
            model.eval()
            for p in (eval_prompts or [])[:2]:
                s = chat_generate(model, tok, p)
                log.info(f"  [chat] {p} → {s}")
            model.train()

    # 保存
    if cfg.model.use_lora:
        torch.save({"adapter": adapter_state_dict(model), "lora_cfg": dataclasses.asdict(cfg.model.lora)},
                   out_dir / "adapter.pt")
        log.info(f"LoRA adapter 已保存: {out_dir / 'adapter.pt'}")
        if cfg.model.lora.merge:
            base_snapshot = snapshot_weights(model)
            n = merge_lora(model)
            save_checkpoint(out_dir / "merged.pt", model, opt, steps, dataclasses.asdict(cfg))
            log.info(f"LoRA 已合并进基座（{n} 层）并保存: {out_dir / 'merged.pt'}")
    else:
        save_checkpoint(out_dir / "sft.pt", model, opt, steps, dataclasses.asdict(cfg))
        save_checkpoint(out_dir / "ckpt-last.pt", model, opt, steps, dataclasses.asdict(cfg))
        log.info(f"SFT checkpoint 已保存: {out_dir / 'sft.pt'}")
    if ml:
        ml.close()
    return model


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/sft.yaml"
    run_sft(load_config(cfg_path, TrainStageConfig))
