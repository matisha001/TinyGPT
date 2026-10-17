"""GPT 模型单元测试：前向/权重共享/因果掩码/损失掩码/RoPE。"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from tinygpt.config import ModelConfig
from tinygpt.model import GPT, apply_rope, precompute_rope


def tiny_cfg(**kw) -> ModelConfig:
    cfg = ModelConfig(vocab_size=300, hidden_size=64, num_layers=2,
                      num_heads=4, ffn_size=128, max_seq_len=32)
    return dataclasses.replace(cfg, **kw)


def test_forward_shape_and_weight_tie():
    cfg = tiny_cfg()
    m = GPT(cfg)
    x = torch.randint(0, 300, (2, 20))
    logits, loss = m(x, targets=x)
    assert logits.shape == (2, 20, 300)
    assert loss is not None and torch.isfinite(loss)
    assert m.lm_head.weight is m.token_embed.weight          # 权重共享
    _, loss2 = m(x)
    assert loss2 is None


def test_causal_mask_position_independence():
    """位置 i 的 logit 只依赖 [0..i] 的 token，不受后续 token 影响。"""
    cfg = tiny_cfg()
    m = GPT(cfg)
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[1, 2, 3, 99]])
    la = m(a)[0]
    lb = m(b)[0]
    for i in range(3):                                    # pos 0..2 相同
        assert torch.allclose(la[0, i], lb[0, i], atol=1e-6)
    assert not torch.allclose(la[0, 3], lb[0, 3], atol=1e-6)   # pos 3 不同


def test_loss_mask():
    cfg = tiny_cfg()
    m = GPT(cfg)
    x = torch.randint(0, 300, (1, 6))
    y = torch.randint(0, 300, (1, 6))
    logits = m(x)[0]
    ce = F.cross_entropy(logits.view(-1, 300), y.view(-1), reduction="none")
    # 只对后 3 位计损失
    mask = torch.zeros_like(y)
    mask[0, 3:] = 1
    _, masked = m(x, targets=y, loss_mask=mask)
    assert torch.allclose(masked, ce[3:].mean(), atol=1e-5)


def test_rope_rotation():
    """RoPE：不同位置旋转不同；旋转为等距变换（保持范数）。"""
    cos, sin = precompute_rope(8, 4)
    q = torch.randn(1, 2, 8, 4)
    q0 = apply_rope(q, cos[:1], sin[:1])
    q1 = apply_rope(q, cos[1:2], sin[1:2])
    assert not torch.allclose(q0, q1, atol=1e-6)          # 位置相关
    assert torch.allclose(q0.norm(), q.norm(), atol=1e-5)  # 正交旋转
    assert torch.allclose(q1.norm(), q.norm(), atol=1e-5)


def test_loss_decreases_on_overfit():
    """小模型在重复短序列上训练应快速过拟合（loss 显著下降）。"""
    cfg = tiny_cfg()
    m = GPT(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]).repeat(8, 1)
    y = x.clone()
    losses = []
    for _ in range(100):
        opt.zero_grad()
        _, loss = m(x, targets=y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5                    # 至少减半
    assert losses[-1] < 2.5
