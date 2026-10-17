"""DPO 偏好对齐：参考模型 logprob 缓存 + β 温度 + 附加 SFT 正则。"""
from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from tinygpt.config import TrainStageConfig, load_config
from tinygpt.data.dpo import DPOData
from tinygpt.lora import adapter_state_dict, apply_lora, trainable_only_adapters
from tinygpt.model import GPT
from tinygpt.tokenizer import BytePairTokenizer
from tinygpt.train.common import build_optimizer, get_lr, load_checkpoint, set_seed
from tinygpt.utils.logging import MetricLogger, setup_logger

log = setup_logger("tinygpt.dpo")


def build_model_for_tokenizer(cfg: TrainStageConfig, tok: BytePairTokenizer) -> GPT:
    return GPT(dataclasses.replace(cfg.model, vocab_size=tok.vocab_size_actual))


def sequence_logprobs(model: GPT, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B,L] → [B]：回答段（掩码=1 的位置）逐 token logprob 的均值。

    logprob(t) = log_softmax(logits[t])[target[t+1]]，只在 mask[t+1]=1 处累计。
    梯度开关由调用方控制：预计算参考缓存时外层 no_grad；训练时保持梯度。
    """
    logits = model(ids)[0]                       # [B, L, V]
    lg = F.log_softmax(logits, dim=-1)
    target = ids[:, 1:].unsqueeze(-1)            # [B, L-1, 1]
    per = lg[:, :-1].gather(-1, target).squeeze(-1)   # [B, L-1]
    m = mask[:, 1:].float()
    return (per * m).sum(-1) / (m.sum(-1) + 1e-8)


def precompute_ref_logprobs(cfg: TrainStageConfig, out_path: Optional[str] = None) -> None:
    """离线计算参考模型（SFT 基座、无 LoRA）对全部偏好对的 logprob 并缓存。"""
    set_seed(cfg.train.seed)
    tok = BytePairTokenizer.load(cfg.data.tokenizer_path)
    ref = build_model_for_tokenizer(cfg, tok)
    load_checkpoint(cfg.train.init_from, ref)         # ref = sft.pt（基座）
    for p in ref.parameters():
        p.requires_grad_(False)
    data = DPOData(cfg.data.dpo_path, tok, seq_len=cfg.data.seq_len,
                   batch_size=cfg.data.batch_size, seed=cfg.train.seed,
                   max_samples=cfg.data.max_samples)
    ref.eval()
    chosen_lp, rejected_lp = [], []
    with torch.no_grad():
        for batch in data.iter_batches():
            c = sequence_logprobs(ref, batch["chosen_ids"], batch["chosen_mask"])
            r = sequence_logprobs(ref, batch["rejected_ids"], batch["rejected_mask"])
            chosen_lp.append(c)
            rejected_lp.append(r)
            if sum(x.shape[0] for x in chosen_lp) >= len(data.pairs):
                break
    chosen = torch.cat(chosen_lp)[: len(data.pairs)]
    rejected = torch.cat(rejected_lp)[: len(data.pairs)]
    out = out_path or cfg.train.ref_logprobs_path
    torch.save({"chosen": chosen, "rejected": rejected}, out)
    log.info(f"参考模型 logprob 缓存已保存: {out}（{len(data.pairs)} 对）")


def dpo_loss(model: GPT, batch: dict, ref_logp: dict, beta: float,
             sft_reg: float, grad_accum: int) -> torch.Tensor:
    """DPO 损失 + SFT 正则。

    L = -log σ(β·(logπθ(y_w) − logπref(y_w) − logπθ(y_l) + logπref(y_l)))
        + λ · SFT_CE(chosen 回答段)
    """
    c_logp = sequence_logprobs(model, batch["chosen_ids"], batch["chosen_mask"])
    r_logp = sequence_logprobs(model, batch["rejected_ids"], batch["rejected_mask"])
    inner = beta * (c_logp - ref_logp["chosen"] - r_logp + ref_logp["rejected"])
    loss = -F.logsigmoid(inner).mean()
    if sft_reg > 0:
        logits = model(batch["chosen_ids"])[0]
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                             batch["chosen_ids"][:, 1:].reshape(-1), reduction="none")
        m = batch["chosen_mask"][:, 1:].reshape(-1).float()
        sft_ce = (ce * m).sum() / (m.sum() + 1e-8)
        loss = loss + sft_reg * sft_ce
    return loss / grad_accum


def run_dpo(cfg: TrainStageConfig, metrics_path: Optional[str] = None) -> GPT:
    set_seed(cfg.train.seed)
    ml = MetricLogger(metrics_path) if metrics_path else None

    tok = BytePairTokenizer.load(cfg.data.tokenizer_path)
    policy = build_model_for_tokenizer(cfg, tok)
    load_checkpoint(cfg.train.init_from, policy)      # 策略基座 = sft.pt
    if cfg.model.use_lora:
        apply_lora(policy, cfg.model.lora)
        trainable_only_adapters(policy)
    log.info(f"策略模型可训练参数: "
             f"{sum(p.numel() for p in policy.parameters() if p.requires_grad) / 1e3:.1f}K")

    ref_path = cfg.train.ref_logprobs_path
    if not (ref_path and Path(ref_path).exists()):
        raise FileNotFoundError(
            f"参考模型 logprob 缓存缺失: {ref_path}。请先运行 "
            f"scripts/precompute_ref_logprobs.py --config configs/dpo.yaml")
    cache = torch.load(ref_path, map_location="cpu")
    ref_logp = {"chosen": cache["chosen"], "rejected": cache["rejected"]}
    log.info(f"载入参考 logprob 缓存: {ref_path}")

    data = DPOData(cfg.data.dpo_path, tok, seq_len=cfg.data.seq_len,
                   batch_size=cfg.data.batch_size, seed=cfg.train.seed,
                   max_samples=cfg.data.max_samples)
    opt = build_optimizer(policy, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps = cfg.train.steps
    warmup = cfg.train.warmup_steps
    grad_accum = cfg.train.grad_accum

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = iter(data.iter_batches())
    policy.train()
    micro = 0
    for step in range(steps):
        lr = get_lr(step, warmup, steps, cfg.train.lr, cfg.train.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = next(batches)
        loss = dpo_loss(policy, batch, ref_logp, cfg.train.dpo_beta,
                        cfg.train.dpo_sft_reg, grad_accum)
        loss.backward()
        micro += 1
        if micro == grad_accum:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.train.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            micro = 0
        if (step + 1) % cfg.train.log_every == 0:
            msg = f"step {step + 1}/{steps} loss={loss.item() * grad_accum:.4f} lr={lr:.2e}"
            log.info(msg)
            if ml:
                ml.log(phase="dpo", step=step + 1, loss=loss.item() * grad_accum, lr=lr)

    torch.save({"adapter": adapter_state_dict(policy), "lora_cfg": dataclasses.asdict(cfg.model.lora),
                "dpo_beta": cfg.train.dpo_beta},
               out_dir / "adapter.pt")
    log.info(f"DPO adapter 已保存: {out_dir / 'adapter.pt'}")
    if ml:
        ml.close()
    return policy


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dpo.yaml"
    run_dpo(load_config(cfg_path, TrainStageConfig))
