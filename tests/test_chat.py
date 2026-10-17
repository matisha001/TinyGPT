"""对话模板单元测试。"""
from __future__ import annotations

from tinygpt.inference.chat import build_chat_ids
from tinygpt.tokenizer import BytePairTokenizer

TOK = BytePairTokenizer.load("data/tokenizer/tokenizer.json")


def test_build_chat_ids_roundtrip():
    messages = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
    ]
    ids = build_chat_ids(TOK, messages)
    text = TOK.decode(ids)
    assert text.startswith("<s>")
    assert "<|user|>你好" in text
    assert "<|assistant|>你好！</s>" in text
    # 末轮 user 不带 </s>（等待模型续写 assistant 段）
    assert text.endswith("<|user|>再见")
