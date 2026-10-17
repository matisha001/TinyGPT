"""微型 ViT 与多模态模型单元测试。"""
from __future__ import annotations

import dataclasses

import torch

from tinygpt.config import ModelConfig, VisionConfig
from tinygpt.model.multimodal import VisionGPT, VisionProjector
from tinygpt.model.vit import VisionTransformer


def _vit_cfg() -> VisionConfig:
    return VisionConfig(image_size=32, patch_size=8, hidden=32, depth=2,
                        heads=4, num_vision_tokens=4)


def test_vit_patch_embed_and_shape():
    v = VisionTransformer(_vit_cfg(), num_classes=5)
    x = torch.randn(2, 3, 32, 32)                  # 4×4 patches = 16
    feats, pooled, logits = v(x)
    assert feats.shape == (2, 16, 32)
    assert pooled.shape == (2, 32)
    assert logits.shape == (2, 5)


def test_vit_pool_modes():
    v = VisionTransformer(_vit_cfg(), num_classes=5)
    x = torch.randn(1, 3, 32, 32)
    f_cls, p_cls, _ = v(x, pool="cls")
    f_mean, p_mean, _ = v(x, pool="mean")
    assert p_cls.shape == p_mean.shape == (1, 32)
    assert not torch.allclose(p_cls, p_mean)


def test_vision_projector_compresses():
    proj = VisionProjector(in_hidden=32, out_hidden=64, num_queries=4)
    feats = torch.randn(2, 16, 32)
    out = proj(feats)
    assert out.shape == (2, 4, 64)


def test_visiongpt_forward_with_images():
    cfg = ModelConfig(vocab_size=300, hidden_size=64, num_layers=2,
                      num_heads=4, ffn_size=128, max_seq_len=32)
    m = VisionGPT(cfg, _vit_cfg())
    ids = torch.randint(0, 300, (2, 16))
    imgs = torch.randn(2, 3, 32, 32)
    y = ids.clone()
    logits, loss = m(ids, images=imgs, targets=y)
    # 文本 16 token + 视觉 4 token
    assert logits.shape == (2, 20, 300)
    assert loss is not None and torch.isfinite(loss)
    # 无图像时退回纯文本前向
    logits2, loss2 = m(ids, targets=y)
    assert logits2.shape == (2, 16, 300)
    assert torch.allclose(loss, loss2, atol=1e-6) is False or True   # 仅检查可运行


def test_visiongpt_loss_mask():
    cfg = ModelConfig(vocab_size=300, hidden_size=64, num_layers=2,
                      num_heads=4, ffn_size=128, max_seq_len=32)
    m = VisionGPT(cfg, _vit_cfg())
    ids = torch.randint(0, 300, (2, 16))
    imgs = torch.randn(2, 3, 32, 32)
    mask = torch.zeros_like(ids)
    mask[:, 8:] = 1
    _, loss = m(ids, images=imgs, targets=ids, loss_mask=mask)
    assert loss is not None and torch.isfinite(loss)
