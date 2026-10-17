"""训练通用工具：随机种子 / 优化器 / 学习率调度 / checkpoint。"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float,
                    betas=(0.9, 0.95)) -> torch.optim.Optimizer:
    """AdamW：权重衰减只作用于矩阵参数（≥2 维），避免衰减 norm/embedding 的偏置类项。"""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    """warmup + cosine 衰减。"""
    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step + 1) / (warmup_steps + 1)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * ratio))


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    step: int, cfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg,
        "extra": extra or {},
    }, p)


def load_checkpoint(path: str | Path, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
