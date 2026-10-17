"""OpenAI 兼容 API 集成测试：用随机初始化的小模型 + TestClient 全端点冒烟。"""
from __future__ import annotations

import dataclasses

import pytest
import torch
from fastapi.testclient import TestClient

from tinygpt.config import APIConfig, ModelConfig, ServerConfig, CapabilitiesConfig
from tinygpt.model import GPT
from tinygpt.rag import KnowledgeBase
from tinygpt.server.app import Engine, create_app


@pytest.fixture(scope="module")
def engine_and_client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api_test")
    tok_cfg = ModelConfig(vocab_size=8192, hidden_size=128, num_layers=2,
                          num_heads=4, ffn_size=256, max_seq_len=64)
    model = GPT(tok_cfg)
    ckpt_path = tmp / "tiny.pt"
    torch.save({"model_cfg": tok_cfg, "model_state": model.state_dict()}, ckpt_path)

    # 知识库
    kb = KnowledgeBase(tmp / "kb" / "index.json").load_or_build([
        {"text": "豆包是字节跳动推出的 AI 智能助手", "meta": {}},
        {"text": "LoRA 是低秩微调方法", "meta": {}},
    ])

    cfg = APIConfig(server=ServerConfig(max_tokens=8, temperature=0.7),
                    model=dataclasses.replace(
                        dataclasses.replace(APIConfig().model, model_path=str(ckpt_path)),
                        tokenizer_path="data/tokenizer/tokenizer.json"),
                    capabilities=CapabilitiesConfig(tools=True, agent_loop=True,
                                                    rag=True, kb_path=str(tmp / "kb" / "index.json")))
    engine = Engine(cfg)
    engine.kb = kb
    app = create_app(engine)
    client = TestClient(app)
    yield client, engine
    client.close()


def test_models(engine_and_client):
    client, _ = engine_and_client
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "tiny-gpt"


def test_completions(engine_and_client):
    client, _ = engine_and_client
    r = client.post("/v1/completions", json={"prompt": "你好", "max_tokens": 8})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert isinstance(body["choices"][0]["text"], str)


def test_chat_completions(engine_and_client):
    client, _ = engine_and_client
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "你好"}], "max_tokens": 8})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["role"] == "assistant"


def test_chat_completions_stream(engine_and_client):
    client, _ = engine_and_client
    with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True, "max_tokens": 8}) as r:
        assert r.status_code == 200
        text = "".join(r.iter_text())
    assert "data: [DONE]" in text
    assert "chat.completion.chunk" in text


def test_agent_completions_tool_loop(engine_and_client):
    client, _ = engine_and_client
    r = client.post("/v1/agent/completions", json={
        "messages": [{"role": "user", "content": "帮我算一下 2 加 3。"}],
        "max_turns": 2, "max_tokens": 8})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "agent.completion"
    assert "turns" in body and isinstance(body["trace"], list)


def test_rag_completions(engine_and_client):
    client, _ = engine_and_client
    r = client.post("/v1/rag/completions", json={"query": "LoRA 是什么？", "top_k": 1,
                                                 "max_tokens": 8})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "rag.completion"
    assert body["sources"][0]["text"].startswith("LoRA")
