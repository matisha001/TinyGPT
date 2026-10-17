"""多模态模型：微型 ViT + 投影层（注意力池化压缩） + LLM 联合微调。

结构：图像 → ViT patch 特征 → 可学习查询注意力池化压缩为 num_vision_tokens 个
视觉 token → 投影到 LLM hidden → 拼接在文本序列头部（<s> 之后）→ GPT 前向。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from tinygpt.config import VisionConfig
from tinygpt.model.gpt import GPT
from tinygpt.model.vit import VisionTransformer


class VisionProjector(nn.Module):
    """patch 特征 → 固定数量视觉 token：可学习查询 + 注意力池化 + MLP 投影。"""

    def __init__(self, in_hidden: int, out_hidden: int, num_queries: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, in_hidden) * 0.02)
        self.kv = nn.Linear(in_hidden, in_hidden, bias=False)
        self.q = nn.Linear(in_hidden, in_hidden, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(in_hidden, in_hidden),
            nn.GELU(),
            nn.Linear(in_hidden, out_hidden),
        )
        self.ln = nn.LayerNorm(in_hidden)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: [B,N,in_h] → [B, num_queries, out_h]"""
        B, N, _ = feats.shape
        q = self.q(self.queries.expand(B, -1, -1))              # [B,Q,in_h]
        k = self.kv(self.ln(feats))                              # [B,N,in_h]
        att = torch.softmax(q @ k.transpose(-2, -1) / (feats.size(-1) ** 0.5), dim=-1)
        pooled = att @ feats                                      # [B,Q,in_h]
        return self.mlp(pooled)


class VisionGPT(GPT):
    """LLM + 视觉编码器；forward 可同时接收图像。"""

    def __init__(self, cfg, vision_cfg: VisionConfig, num_classes: int = 10):
        super().__init__(cfg)
        self.vision_cfg = vision_cfg
        self.vision_encoder = VisionTransformer(vision_cfg, num_classes=num_classes)
        self.vision_projector = VisionProjector(vision_cfg.hidden, cfg.hidden_size,
                                                vision_cfg.num_vision_tokens)

    def n_vision_params(self) -> int:
        return sum(p.numel() for p in self.vision_encoder.parameters()) + \
               sum(p.numel() for p in self.vision_projector.parameters())

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.vision_encoder(images)[0]                    # [B,N,vh]
        return self.vision_projector(feats)                       # [B,Q,hh]

    def forward(self, input_ids: torch.Tensor, images: Optional[torch.Tensor] = None,
                targets: Optional[torch.Tensor] = None,
                loss_mask: Optional[torch.Tensor] = None) -> tuple:
        """images: [B,3,H,W]；视觉 token 拼接在文本序列头部。"""
        text_emb = self.token_embed(input_ids)                    # [B,T,hh]
        if images is not None:
            vis = self.encode_image(images)                       # [B,Q,hh]
            x = torch.cat([vis, text_emb], dim=1)
        else:
            x = text_emb
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        n_vis = 0 if images is None else vis.size(1)
        # 视觉 token 不参与损失：targets 对齐文本部分
        if n_vis:
            text_logits = logits[:, n_vis:]
            loss = torch.nn.functional.cross_entropy(
                text_logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                reduction="none")
            if loss_mask is not None:
                m = loss_mask.reshape(-1).float()
                loss = (loss * m).sum() / (m.sum() + 1e-8)
            else:
                loss = loss.mean()
            return logits, loss
        return super().forward(input_ids, targets, loss_mask)


def build_vision_model(cfg, vision_cfg: VisionConfig) -> VisionGPT:
    import dataclasses
    return VisionGPT(dataclasses.replace(cfg, vocab_size=cfg.vocab_size), vision_cfg)
