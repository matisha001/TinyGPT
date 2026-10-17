"""RAG 检索单元测试。"""
from __future__ import annotations

from tinygpt.rag import KnowledgeBase, TFIDFIndex, char_bigrams
from tinygpt.tools import run_tool


def test_char_bigrams():
    assert char_bigrams("abc") == ["ab", "bc"]
    assert char_bigrams("机器学习") == ["机器", "器学", "学习"]
    assert char_bigrams("a") == ["a"]


def test_tfidf_relevant_first():
    idx = TFIDFIndex()
    idx.add_document("豆包是字节跳动推出的 AI 助手")
    idx.add_document("上海是中国最大的城市之一")
    idx.add_document("LoRA 是低秩微调方法")
    idx.build()
    hits = idx.search("豆包助手", top_k=3)
    assert hits[0]["id"] == 0
    hits2 = idx.search("上海", top_k=3)
    assert hits2[0]["id"] == 1


def test_tfidf_save_load():
    idx = TFIDFIndex()
    idx.add_document("测试文档A 内容")
    idx.add_document("测试文档B 内容")
    idx.build()
    p = "/tmp/test_kb_index.json"
    idx.save(p)
    idx2 = TFIDFIndex.load(p)
    assert idx2.vocab == idx.vocab
    assert len(idx2.docs) == 2


def test_kb_search_tool_registered(tmp_path):
    docs = [
        {"text": "豆包是字节跳动推出的 AI 智能助手", "meta": {}},
        {"text": "Python 常用于机器学习", "meta": {}},
    ]
    kb = KnowledgeBase(tmp_path / "idx.json").load_or_build(docs)
    r = run_tool("search_kb", {"query": "字节跳动的 AI 助手", "top_k": 1})
    assert r["ok"]
    hits = r["result"]
    assert hits[0]["text"].startswith("豆包")


def test_kb_build_prompt():
    kb = KnowledgeBase().load_or_build([
        {"text": "LoRA 是低秩微调方法", "meta": {}},
        {"text": "上海是金融中心", "meta": {}},
    ])
    prompt = kb.build_prompt("什么是 LoRA？", top_k=1)
    assert "LoRA 是低秩微调方法" in prompt
    assert "问题：什么是 LoRA？" in prompt
