"""从零实现的微型 Vision Transformer：patch 嵌入 + 位置编码 + 多头自注意力 + 分类头/池化。"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinygpt.config import VisionConfig


class PatchEmbed(nn.Module):
    """patch 嵌入：Conv2d(kernel=stride=patch_size) 等价于逐 patch 的线性投影。"""

    def __init__(self, image_size: int, patch_size: int, in_ch: int = 3, hidden: int = 192):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, hidden, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,3,H,W] → [B, N, hidden]"""
        B = x.shape[0]
        return self.proj(x).flatten(2).transpose(1, 2)


class ViTBlock(nn.Module):
    """双向多头自注意力 + MLP（Pre-Norm，GELU）。"""

    def __init__(self, hidden: int, heads: int, mlp_ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        assert hidden % heads == 0
        self.ln1 = nn.LayerNorm(hidden, eps=eps)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, int(hidden * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden * mlp_ratio), hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x


class VisionTransformer(nn.Module):
    """微型 ViT：patch 嵌入 → [cls]+位置编码 → L 层 block → 池化/分类头。"""

    def __init__(self, cfg: VisionConfig, num_classes: int = 10):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, in_ch=3, hidden=cfg.hidden)
        n = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, cfg.hidden))
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([ViTBlock(cfg.hidden, cfg.heads) for _ in range(cfg.depth)])
        self.ln = nn.LayerNorm(cfg.hidden)
        self.head = nn.Linear(cfg.hidden, num_classes)     # 分类头（pool 输出）
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, pool: str = "mean") -> tuple:
        """x: [B,3,H,W] → (patch_features[B,N,h], pooled[B,h])；pool='cls'|'mean'。"""
        x = self.patch_embed(x)                            # [B,N,h]
        B, N, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                      # [B,N+1,h]
        x = x + self.pos_embed[:, : N + 1]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        feats = x[:, 1:]                                    # [B,N,h]（去掉 cls）
        if pool == "cls":
            pooled = x[:, 0]
        else:
            pooled = feats.mean(dim=1)
        logits = self.head(pooled)                          # 分类头
        return feats, pooled, logits
