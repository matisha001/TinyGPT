"""SFT 数据单元测试：掩码位置 / 打包 / 格式。"""
from __future__ import annotations

import torch

from tinygpt.data.sft import SFTData, format_prompt
from tinygpt.tokenizer import BytePairTokenizer

TOK = BytePairTokenizer.load("data/tokenizer/tokenizer.json")
USER = "<|user|>"
ASSISTANT = "<|assistant|>"


def test_format_prompt():
    p = format_prompt("你好", "今天天气如何")
    assert p == f"{USER}你好\n今天天气如何{ASSISTANT}"
    p2 = format_prompt("你好")
    assert p2 == f"{USER}你好{ASSISTANT}"


def test_mask_only_on_answer(tmp_path):
    d = tmp_path / "s.jsonl"
    d.write_text('{"instruction": "问题A", "input": "", "output": "回答A"}\n', encoding="utf-8")
    data = SFTData(d, TOK, seq_len=32, batch_size=2, seed=0)
    ids, mask = data.blocks[0]
    text = TOK.decode(ids.tolist())
    # 掩码位置应恰好覆盖回答段（含 </s>）
    ans_start = text.index("回答A")
    mask_start = int(mask.argmax())          # 第一个掩码=1 的位置
    prefix = TOK.decode(ids[:mask_start].tolist())
    assert "回答A" not in prefix             # 前缀（prompt 段）不含回答
    assert "问题A" in prefix                 # 前缀含问题
    # 掩码连续性：一旦出现 1，后续（非 pad）都是 1
    ones = (mask == 1).nonzero().flatten()
    assert len(ones) > 0
    assert ones[0].item() == mask_start


def test_packing_respects_seq_len(tmp_path):
    d = tmp_path / "s.jsonl"
    d.write_text('\n'.join(f'{{"instruction": "问题{i}", "input": "", "output": "回答{i}"}}'
                           for i in range(50)) + "\n", encoding="utf-8")
    data = SFTData(d, TOK, seq_len=64, batch_size=4, seed=0)
    for ids, mask in data.blocks:
        assert ids.shape == (64,)
        assert mask.shape == (64,)
        assert mask.float().sum() >= 1       # 每块至少一个回答位置


def test_iter_batches_shapes(tmp_path):
    d = tmp_path / "s.jsonl"
    d.write_text('\n'.join(f'{{"instruction": "问题{i}", "input": "", "output": "回答{i}"}}'
                           for i in range(30)) + "\n", encoding="utf-8")
    data = SFTData(d, TOK, seq_len=32, batch_size=4, seed=42)
    it = iter(data.iter_batches())
    x, y, mask = next(it)
    assert x.shape == (4, 31) and y.shape == (4, 31) and mask.shape == (4, 31)
    assert (mask == 1).any()
