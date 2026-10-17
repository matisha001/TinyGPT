"""字节级 BPE 分词器单元测试。"""
import pytest

from tinygpt.tokenizer import BytePairTokenizer

SPECIALS = ["<s>", "</s>", "<pad>", "<unk>", "<|user|>", "<|assistant|>", "<|image|>", "<|tool|>"]


def make_tok(vocab_size: int = 500) -> BytePairTokenizer:
    return BytePairTokenizer(vocab_size=vocab_size, special_tokens=SPECIALS)


def test_id_layout():
    tok = make_tok()
    assert tok.id_of("<s>") == 0
    assert tok.id_of("<|tool|>") == 7
    assert tok.byte_base == 8          # 特殊 token 后是 256 个字节
    assert tok.merge_base == 264
    assert tok.encode("A")[0] == 8 + 65


def test_train_roundtrip_zh_en_emoji():
    tok = make_tok()
    corpus = [
        "这是一个测试语料。Hello world! 机器学习很棒。",
        "自然语言处理 NLP 12345 🚀 中文分词与字节级编码。",
        "The quick brown fox 跳跃 over the lazy dog。",
    ] * 20
    tok.train(corpus, min_freq=2, verbose=False)
    for s in corpus[:6]:
        assert tok.decode(tok.encode(s)) == s, f"回环失败: {s}"


def test_special_tokens_stable():
    tok = make_tok()
    tok.train(["训练分词器 special token 不参与 训练 文本 abc abc abc", "第二行 测试"], min_freq=1, verbose=False)
    assert tok.id_of("<s>") == 0
    assert tok.id_of("<|assistant|>") == 5
    # 特殊 token 原样解码
    ids = [tok.id_of("<s>")] + tok.encode("你好") + [tok.id_of("</s>")]
    assert tok.decode(ids) == "<s>你好</s>"


def test_vocab_size_cap():
    import random
    tok = BytePairTokenizer(vocab_size=300, special_tokens=["<s>"])
    # 确定性随机文本：合并规则远超 43 个，验证词表封顶逻辑
    alph = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()"
    corpus = ["".join(random.Random(0).choices(alph, k=3000))]
    tok.train(corpus, min_freq=1, verbose=False)
    assert tok.vocab_size_actual == 300          # 1 特殊 + 256 字节 + 43 merges
    assert tok.n_merges == 300 - 1 - 256


def test_save_load_roundtrip(tmp_path):
    tok = make_tok()
    tok.train(["测试 save load 模型 参数 训练 训练", "深度学习 PyTorch 分词器"], min_freq=1, verbose=False)
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = BytePairTokenizer.load(p)
    s = "深度学习 PyTorch 测试"
    assert tok2.decode(tok2.encode(s)) == s
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.special_tokens == tok.special_tokens
    assert tok2.n_merges == tok.n_merges


def test_deterministic():
    corpus = ["确定性 与 随机性 测试 测试 语料 语料 语料"] * 5
    a = BytePairTokenizer(400, ["<s>"]); a.train(corpus, min_freq=1, verbose=False)
    b = BytePairTokenizer(400, ["<s>"]); b.train(corpus, min_freq=1, verbose=False)
    assert a.encode("测试 语料") == b.encode("测试 语料")
    assert a.n_merges == b.n_merges


def test_unknown_special_raises():
    tok = make_tok()
    with pytest.raises(KeyError):
        tok.id_of("不存在")
