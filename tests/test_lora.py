"""LoRA 单元测试：包装/冻结/合并/恢复/初始等价。"""
from __future__ import annotations

import dataclasses

import torch

from tinygpt.config import LoRAConfig, ModelConfig
from tinygpt.lora import (adapter_state_dict, apply_lora, lora_adapters,
                          merge_lora, snapshot_weights, trainable_only_adapters,
                          unmerge_lora)
from tinygpt.model import GPT


def _model() -> GPT:
    cfg = ModelConfig(vocab_size=300, hidden_size=64, num_layers=2,
                      num_heads=4, ffn_size=128, max_seq_len=32)
    return GPT(cfg)


def test_apply_lora_targets_and_initial_equivalence():
    m = _model()
    x = torch.randint(0, 300, (2, 16))
    before = m(x)[0].detach()
    apply_lora(m, LoRAConfig(rank=4, alpha=8.0, dropout=0.0))
    n_wrapped = len(lora_adapters(m))
    assert n_wrapped == 4 * 2 + 3 * 2          # 每层 4 注意力 + 3 FFN，共 2 层 = 14
    # B=0 → 前向与原始完全一致
    after = m(x)[0]
    assert torch.allclose(before, after, atol=1e-6)
    # 只允许 A/B 可训练
    trainable_only_adapters(m)
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert all(n.endswith((".lora_A", ".lora_B")) for n in names)
    assert len(names) == 2 * n_wrapped


def test_merge_and_unmerge():
    m = _model()
    apply_lora(m, LoRAConfig(rank=4, alpha=8.0, dropout=0.0))
    x = torch.randint(0, 300, (2, 16))
    # 先让 adapter 有非零值
    with torch.no_grad():
        for mod in lora_adapters(m):
            mod.lora_B.normal_()
    before = m(x)[0]
    snap = snapshot_weights(m)
    n = merge_lora(m)
    assert n == 14
    after = m(x)[0]
    assert torch.allclose(before, after, atol=1e-5)     # 合并后前向不变
    unmerge_lora(m, snap)
    restored = m(x)[0]
    assert torch.allclose(after, restored, atol=1e-4)   # unmerge 用旧权重 → 不等价于 after
    # unmerge 恢复的是基座权重（B 仍在，前向 = base + lora）
    assert torch.allclose(restored, before, atol=1e-5)


def test_adapter_state_dict_roundtrip():
    m = _model()
    apply_lora(m, LoRAConfig(rank=4, alpha=8.0, dropout=0.0))
    state = adapter_state_dict(m)
    assert all(k.endswith((".lora_A", ".lora_B")) for k in state)
    m2 = _model()
    apply_lora(m2, LoRAConfig(rank=4, alpha=8.0, dropout=0.0))
    # 复制基座权重（保证两模型基座一致），再载入 adapter
    base = {k: v for k, v in m.state_dict().items() if not k.endswith((".lora_A", ".lora_B"))}
    m2.load_state_dict(base, strict=False)
    m2.load_state_dict(state, strict=False)
    x = torch.randint(0, 300, (2, 16))
    assert torch.allclose(m(x)[0], m2(x)[0], atol=1e-6)


def test_lora_trains_and_changes_output():
    """训练 adapter 后输出应改变（区别于初始等价）。"""
    m = _model()
    apply_lora(m, LoRAConfig(rank=4, alpha=8.0, dropout=0.0))
    trainable_only_adapters(m)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randint(0, 300, (2, 16))
    y = x.clone()
    before = m(x)[0].detach()
    for _ in range(30):
        opt.zero_grad()
        _, loss = m(x, targets=y)
        loss.backward()
        opt.step()
    assert not torch.allclose(before, m(x)[0], atol=1e-3)   # 输出被 adapter 改变
    assert loss.item() < 10.0
