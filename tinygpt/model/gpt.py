"""GPT decoder-only 模型：RoPE / SwiGLU / RMSNorm / 输入输出词嵌入权重共享。

全部从零实现（仅依赖 PyTorch），结构与规模由 ModelConfig 驱动，方便后续放大。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinygpt.config import ModelConfig


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# RoPE 旋转位置编码
# ---------------------------------------------------------------------------
def precompute_rope(max_seq_len: int, head_dim: int, theta: float = 10000.0) -> tuple:
    """预计算 cos/sin 表，形状 [max_seq_len, head_dim // 2]。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)              # [T, d/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 q/k 施加旋转。x: [B,H,T,D]；cos/sin: [T, D/2]。

    半切旋转：out[:h] = x1·cos - x2·sin；out[h:] = x2·cos + x1·sin。
    """
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    cos = cos.unsqueeze(0).unsqueeze(0)           # [1,1,T,h]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---------------------------------------------------------------------------
# 因果自注意力
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.hidden_size
        if d % cfg.num_heads != 0:
            raise ValueError(f"hidden_size={d} 必须能被 num_heads={cfg.num_heads} 整除")
        self.n_heads = cfg.num_heads
        self.head_dim = d // cfg.num_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        cos, sin = precompute_rope(cfg.max_seq_len, self.head_dim, cfg.rope_theta)
        self.register_buffer("cos", cos)                       # [max_seq, h]
        self.register_buffer("sin", sin)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        d = self.head_dim
        q = self.wq(x).view(B, T, self.n_heads, d).transpose(1, 2)   # [B,H,T,d]
        k = self.wk(x).view(B, T, self.n_heads, d).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, d).transpose(1, 2)
        q = apply_rope(q, self.cos[:T], self.sin[:T])
        k = apply_rope(k, self.cos[:T], self.sin[:T])
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d)              # [B,H,T,T]
        att = att.masked_fill(~self.mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v                                                 # [B,H,T,d]
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * d)
        return self.wo(y)


# ---------------------------------------------------------------------------
# SwiGLU 前馈
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, dim: int, ffn_size: int):
        super().__init__()
        self.gate = nn.Linear(dim, ffn_size, bias=False)
        self.up = nn.Linear(dim, ffn_size, bias=False)
        self.down = nn.Linear(ffn_size, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# 解码块（Pre-Norm）
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg.hidden_size, cfg.ffn_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# GPT
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embed.weight           # 权重共享
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None,
                loss_mask: Optional[torch.Tensor] = None) -> tuple:
        """前向。loss_mask 形状同 targets：1=计入损失（如回答段），0=掩码。"""
        x = self.token_embed(input_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                    # [B,T,V]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), reduction="none")
            if loss_mask is not None:
                m = loss_mask.reshape(-1).float()
                loss = (loss * m).sum() / (m.sum() + 1e-8)
            else:
                loss = loss.mean()
        return logits, loss

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_params(self) -> dict:
        """参数统计（按模块分组），便于核对规模。"""
        groups = {}
        for name, mod in self.named_modules():
            n = sum(p.numel() for p in mod.parameters(recurse=False))
            if n:
                groups[name or "root"] = n
        return groups
