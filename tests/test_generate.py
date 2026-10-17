"""生成模块单元测试。"""
from __future__ import annotations

import torch

from tinygpt.config import ModelConfig
from tinygpt.inference.generate import _filter_top_k, _filter_top_p, generate
from tinygpt.model import GPT
from tinygpt.tokenizer import BytePairTokenizer


def _tiny_model(vocab_size: int = 300) -> GPT:
    cfg = ModelConfig(vocab_size=vocab_size, hidden_size=64, num_layers=2,
                      num_heads=4, ffn_size=128, max_seq_len=32)
    return GPT(cfg)


def _tok_model() -> tuple:
    tok = BytePairTokenizer.load("data/tokenizer/tokenizer.json")
    return _tiny_model(vocab_size=tok.vocab_size_actual), tok


def test_generate_roundtrip():
    m, tok = _tok_model()
    out = generate(m, tok, "测试", max_new_tokens=8, temperature=0.7, seed=0)
    assert out.startswith("测试")                          # 前缀保留
    tok.decode(tok.encode(out))                            # 回环不抛异常


def test_generate_deterministic_with_seed():
    m, tok = _tok_model()
    a = generate(m, tok, "你好", max_new_tokens=6, temperature=0.9, seed=123)
    b = generate(m, tok, "你好", max_new_tokens=6, temperature=0.9, seed=123)
    assert a == b


def test_generate_respects_max_seq_len():
    m, tok = _tok_model()
    long_prompt = "长" * 40
    out = generate(m, tok, long_prompt, max_new_tokens=5, temperature=0.5, seed=1)
    assert len(out) > 0                                     # 窗口截断不崩


def test_topk_filter():
    l = torch.tensor([1.0, 2.0, 3.0, 4.0])
    k = _filter_top_k(l.clone(), 2)
    assert torch.isinf(k[0]) and torch.isinf(k[1])          # 两个最小值被屏蔽
    assert k[2] == 3.0 and k[3] == 4.0


def test_topp_filter():
    l = torch.tensor([1.0, 2.0, 3.0, 4.0])
    p = _filter_top_p(l.clone(), 0.6)
    assert torch.isinf(p[0]) and torch.isinf(p[1]) and torch.isinf(p[2])
    assert p[3] == 4.0
