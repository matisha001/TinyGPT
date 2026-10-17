"""预训练训练循环：warmup + cosine、梯度裁剪、checkpoint、评估 PPL。"""
from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Optional

import torch

from tinygpt.config import TrainStageConfig, load_config
from tinygpt.data.pretrain import PretrainData
from tinygpt.model import GPT
from tinygpt.tokenizer import BytePairTokenizer
from tinygpt.train.common import (build_optimizer, get_lr, load_checkpoint,
                                  save_checkpoint, set_seed)
from tinygpt.utils.logging import MetricLogger, setup_logger

log = setup_logger("tinygpt.pretrain")


def build_model_for_tokenizer(cfg: TrainStageConfig, tok: BytePairTokenizer) -> GPT:
    """按分词器实际词表构造模型（配置里的 vocab_size 仅作上限参考）。"""
    model_cfg = dataclasses.replace(cfg.model, vocab_size=tok.vocab_size_actual)
    return GPT(model_cfg)


def run_pretrain(cfg: TrainStageConfig, metrics_path: Optional[str] = None,
                 eval_prompts: Optional[list] = None) -> GPT:
    set_seed(cfg.train.seed)
    ml = MetricLogger(metrics_path) if metrics_path else None

    tok = BytePairTokenizer.load(cfg.data.tokenizer_path)
    model = build_model_for_tokenizer(cfg, tok)
    log.info(f"模型参数量: {model.n_params() / 1e6:.2f}M（词表 {tok.vocab_size_actual}）")

    data = PretrainData(cfg.data.corpus_path, tok, seq_len=cfg.data.seq_len,
                        batch_size=cfg.data.batch_size, seed=cfg.train.seed,
                        max_samples=cfg.data.max_samples,
                        eval_ratio=cfg.data.eval_ratio)
    log.info(f"打包块数: train={len(data)} eval={len(data.eval_blocks)} "
             f"总token={data.n_tokens()}")

    opt = build_optimizer(model, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps = cfg.train.steps
    grad_accum = cfg.train.grad_accum
    warmup = cfg.train.warmup_steps

    start_step = 0
    if cfg.train.resume and Path(cfg.train.resume).exists():
        ckpt = load_checkpoint(cfg.train.resume, model, opt)
        start_step = ckpt.get("step", 0)
        log.info(f"断点续训: {cfg.train.resume}（step={start_step}）")

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batches = iter(data.iter_batches())
    model.train()
    micro = 0
    for step in range(start_step, steps):
        lr = get_lr(step, warmup, steps, cfg.train.lr, cfg.train.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = next(batches)
        _, loss = model(x, targets=y)
        loss = loss / grad_accum
        loss.backward()
        micro += 1
        if micro == grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            micro = 0

        if (step + 1) % cfg.train.log_every == 0:
            msg = f"step {step + 1}/{steps} loss={loss.item() * grad_accum:.4f} lr={lr:.2e}"
            log.info(msg)
            if ml:
                ml.log(phase="pretrain", step=step + 1,
                       loss=loss.item() * grad_accum, lr=lr)
        if (step + 1) % cfg.train.eval_every == 0:
            eval_loss = evaluate(model, data)
            ppl = math.exp(eval_loss)
            log.info(f"  [eval] step={step + 1} loss={eval_loss:.4f} ppl={ppl:.2f}")
            if ml:
                ml.log(phase="pretrain_eval", step=step + 1,
                       eval_loss=eval_loss, eval_ppl=ppl)
            if eval_prompts:
                for p in eval_prompts:
                    s = generate_sample(model, tok, p)
                    log.info(f"  [sample] {p} → {s}")
            model.train()
        if (step + 1) % cfg.train.save_every == 0:
            save_checkpoint(out_dir / f"ckpt-{step + 1}.pt", model, opt, step + 1,
                            dataclasses.asdict(cfg))

    save_checkpoint(out_dir / "base.pt", model, opt, steps, dataclasses.asdict(cfg))
    save_checkpoint(out_dir / "ckpt-last.pt", model, opt, steps, dataclasses.asdict(cfg))
    log.info(f"预训练完成，checkpoint: {out_dir / 'base.pt'}")
    if ml:
        ml.close()
    return model


def evaluate(model: GPT, data: PretrainData, num_batches: int = 4) -> float:
    """固定评估块上的平均 CE loss。"""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = data.eval_batch()
            x = x[:1]
            y = y[:1]
            _, loss = model(x, targets=y)
            losses.append(loss.item())
    return float(sum(losses) / max(1, len(losses)))


def generate_sample(model: GPT, tok: BytePairTokenizer, prompt: str,
                    max_new_tokens: int = 40, temperature: float = 0.8) -> str:
    from tinygpt.inference.generate import generate
    return generate(model, tok, prompt, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_k=40, top_p=0.9)


if __name__ == "__main__":
    run_pretrain(load_config("configs/pretrain.yaml", TrainStageConfig))
