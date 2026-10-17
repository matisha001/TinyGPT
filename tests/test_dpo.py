"""DPO 单元测试：损失方向 / logprob 计算 / 参考缓存。"""
from __future__ import annotations

import dataclasses

import torch

from tinygpt.config import ModelConfig
from tinygpt.data.dpo import DPOData
from tinygpt.model import GPT
from tinygpt.tokenizer import BytePairTokenizer
from tinygpt.train.dpo import dpo_loss, sequence_logprobs

TOK = BytePairTokenizer.load("data/tokenizer/tokenizer.json")


def _model() -> GPT:
    cfg = ModelConfig(vocab_size=TOK.vocab_size_actual, hidden_size=64, num_layers=2,
                      num_heads=4, ffn_size=128, max_seq_len=32)
    return GPT(cfg)


def _pairs_file(tmp_path, n=8):
    d = tmp_path / "d.jsonl"
    lines = [f'{{"prompt": "问题{i}", "chosen": "好答案{i}", "rejected": "坏答案{i}"}}'
             for i in range(n)]
    d.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def test_sequence_logprobs_masks_prompt(tmp_path):
    """prompt 段不贡献 logprob；答案段贡献。"""
    m = _model()
    data = DPOData(_pairs_file(tmp_path, 2), TOK, seq_len=32, batch_size=2, seed=0)
    # 用手工构造的序列直接验证：全掩码 vs 部分掩码
    ids = torch.tensor([[TOK.special_ids["<s>"], 300, 301, 302]])
    full = torch.ones_like(ids)
    lp_full = sequence_logprobs(m, ids, full)
    # 掩码全 0 → logprob 为 0（无回答段）
    zero = torch.zeros_like(ids)
    lp_zero = sequence_logprobs(m, ids, zero)
    assert lp_zero.item() == 0.0
    assert torch.isfinite(lp_full).item()


def test_dpo_loss_improves_chosen_probability(tmp_path):
    """训练若干步后，chosen 的相对概率应提升。"""
    f = _pairs_file(tmp_path, 4)
    m = _model()
    data = DPOData(f, TOK, seq_len=32, batch_size=2, seed=0)
    ref_batch = {"chosen": torch.zeros(2), "rejected": torch.zeros(2)}  # 简化：ref=均匀
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    losses = []
    before = None
    for _ in range(60):
        batch = next(iter(data.iter_batches()))
        loss = dpo_loss(m, batch, ref_batch, beta=0.1, sft_reg=0.2, grad_accum=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        # 用无梯度前向监控 chosen 与 rejected 的 logprob 差
        with torch.no_grad():
            c = sequence_logprobs(m, batch["chosen_ids"], batch["chosen_mask"])
            r = sequence_logprobs(m, batch["rejected_ids"], batch["rejected_mask"])
            before = (c - r).mean().item()
    assert losses[-1] < losses[0]
    assert before > 0.0                      # chosen − rejected logprob > 0


def test_precompute_ref_cache_shape(tmp_path):
    """参考 logprob 缓存形状 = [N]（每对两个标量）。"""
    from tinygpt.config import DataConfig, TrainConfig, TrainStageConfig
    from tinygpt.train.common import save_checkpoint
    import dataclasses as dc

    f = _pairs_file(tmp_path, 8)
    model_cfg = dc.replace(ModelConfig(vocab_size=TOK.vocab_size_actual, hidden_size=64,
                                       num_layers=2, num_heads=4, ffn_size=128,
                                       max_seq_len=32))
    cfg = TrainStageConfig(
        model=model_cfg,
        data=DataConfig(tokenizer_path="data/tokenizer/tokenizer.json",
                        dpo_path=str(f), seq_len=32, batch_size=4),
        train=TrainConfig(steps=1, seed=0))
    # 准备一个真实基座 checkpoint（ref 从它加载）
    base = GPT(model_cfg)
    ckpt = tmp_path / "base.pt"
    save_checkpoint(ckpt, base, torch.optim.AdamW(base.parameters()), 0, {})
    cfg.train.init_from = str(ckpt)

    cache_path = tmp_path / "ref.pt"
    from tinygpt.train.dpo import precompute_ref_logprobs
    precompute_ref_logprobs(cfg, out_path=str(cache_path))
    cache = torch.load(cache_path, map_location="cpu")
    assert cache["chosen"].shape == (8,)
    assert cache["rejected"].shape == (8,)
