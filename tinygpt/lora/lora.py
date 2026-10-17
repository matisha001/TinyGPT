"""LoRA 低秩适配：包装 nn.Linear 为目标模块追加 A/B 低秩分支。

- 初始 B=0 → LoRA 前向与原始权重完全一致（无缝衔接已有 checkpoint）
- 只训练 A/B（其余冻结）→ 参数量极小、省内存
- merge：把 (B·A)·scaling 合入原始权重，得到"合并后"模型（推理无需额外分支）
- unmerge：恢复原始权重（合并前已存副本）
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from tinygpt.config import LoRAConfig


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float,
                 dropout: float = 0.0):
        super().__init__()
        self.linear = linear                       # 冻结的原始投影
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, linear.in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base + lora * self.scaling


SHORT_TO_ATTR = {"q": "wq", "k": "wk", "v": "wv", "o": "wo",
                 "gate": "gate", "up": "up", "down": "down"}


def _is_lora_target(name: str, targets: List[str]) -> bool:
    """targets 用短名（q/k/v/o/gate/up/down），模块叶子名是 wq/wk/wv/wo/..."""
    leaf = name.split(".")[-1]
    return any(leaf == SHORT_TO_ATTR.get(t, t) for t in targets)


def apply_lora(model: nn.Module, cfg: LoRAConfig) -> nn.Module:
    """把目标 nn.Linear 原位替换为 LoRALinear；返回同一模型（带 _lora 元信息）。"""
    if not cfg.targets:
        return model
    applied = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not isinstance(mod, LoRALinear) \
                and _is_lora_target(name, cfg.targets):
            parent = model
            for part in name.split(".")[:-1]:
                parent = getattr(parent, part)
            attr = name.split(".")[-1]
            wrapped = LoRALinear(mod, cfg.rank, cfg.alpha, cfg.dropout)
            setattr(parent, attr, wrapped)
            applied.append(name)
    model._lora_applied = applied
    return model


def trainable_only_adapters(model: nn.Module) -> None:
    """冻结除 lora_A/lora_B 外的全部参数。"""
    for name, p in model.named_parameters():
        p.requires_grad_(name in {"lora_A", "lora_B"} or name.endswith(".lora_A") or
                          name.endswith(".lora_B"))


def lora_adapters(model: nn.Module) -> List[LoRALinear]:
    out = []
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            out.append(mod)
    return out


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """把 (B·A)·scaling 合入原权重，并将 A/B 置零（避免 wrapper 双重计数）。

    合并后 LoRALinear 前向 = 原始权重(含增量) + 0 = 合并结果；unmerge 时由
    调用方保存的权重快照（含 A/B）恢复。返回合并的层数。
    """
    n = 0
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            mod.linear.weight.add_(mod.lora_B @ mod.lora_A * mod.scaling)
            mod.lora_A.zero_()
            mod.lora_B.zero_()
            n += 1
    return n


@torch.no_grad()
def unmerge_lora(model: nn.Module, saved: Optional[Dict[str, torch.Tensor]] = None) -> None:
    """从合并态恢复原始权重。saved 为合并前权重副本（由 merge_lora 前调用方保存）。"""
    if saved is None:
        raise ValueError("unmerge 需要合并前的权重副本（merge 前先 deepcopy 原权重）")
    for name, p in model.named_parameters():
        if name in saved:
            p.copy_(saved[name])


def snapshot_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """深度拷贝当前权重（合并前调用，供 unmerge 恢复）。"""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def adapter_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """仅保存 adapter 参数（lora_A/lora_B），用于独立 adapter checkpoint。"""
    return {k: v for k, v in model.state_dict().items()
            if k.endswith(".lora_A") or k.endswith(".lora_B")}


def load_adapter_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=False)
