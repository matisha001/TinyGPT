"""OpenAI 兼容 API 服务：/v1/models、/v1/chat/completions（流式/多模态/工具）、
/v1/completions、/v1/agent/completions（工具循环）、/v1/rag/completions。

启动：python scripts/run_api.py --config configs/api.yaml
"""
from __future__ import annotations

import base64
import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from tinygpt.config import APIConfig, load_config
from tinygpt.inference.chat import chat, chat_vision
from tinygpt.inference.generate import generate
from tinygpt.model import GPT, VisionGPT
from tinygpt.model.multimodal import VisionGPT as _VG
from tinygpt.rag import KnowledgeBase
from tinygpt.tokenizer import BytePairTokenizer
from tinygpt.tools import list_tools, parse_tool_call, run_tool
from tinygpt.utils.logging import setup_logger

log = setup_logger("tinygpt.api")


# ---------- 请求/响应模型 ----------
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[dict]]


class ChatRequest(BaseModel):
    model: str = "tiny-gpt"
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[str] = None


class CompletionRequest(BaseModel):
    model: str = "tiny-gpt"
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class AgentRequest(BaseModel):
    model: str = "tiny-gpt"
    messages: List[ChatMessage]
    max_turns: int = 4
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class RAGRequest(BaseModel):
    model: str = "tiny-gpt"
    query: str
    top_k: int = 3
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class Engine:
    """模型/分词器/知识库/可选 MCP 的持有者。"""

    def __init__(self, cfg: APIConfig):
        self.cfg = cfg
        self.tokenizer = BytePairTokenizer.load(cfg.model.tokenizer_path)
        self.model, self.is_vision = self._load_model(cfg)
        self.kb: Optional[KnowledgeBase] = None
        if cfg.capabilities.rag:
            kb_path = Path(cfg.capabilities.kb_path) / "index.json" \
                if Path(cfg.capabilities.kb_path).is_dir() else Path(cfg.capabilities.kb_path)
            self.kb = KnowledgeBase(kb_path).load_or_build([])
        self.settings = cfg.server

    def _load_model(self, cfg: APIConfig):
        ckpt = torch.load(cfg.model.model_path, map_location="cpu")
        state = ckpt["model_state"]
        has_vision = any("vision_encoder" in k for k in state)
        vision_cfg = cfg.model.vision
        from tinygpt.config import ModelConfig
        if has_vision or vision_cfg is not None:
            from dataclasses import replace
            model = _VG(replace(ckpt.get("model_cfg", ModelConfig()),
                                vocab_size=self.tokenizer.vocab_size_actual),
                        vision_cfg)
            model.load_state_dict(state, strict=False)
            log.info(f"加载多模态模型（LLM 参数 {sum(p.numel() for p in model.parameters()) // 1000}K）")
            return model, True
        model = GPT(ckpt.get("model_cfg", ModelConfig()))
        model.load_state_dict(state, strict=False)
        log.info(f"加载文本模型（{sum(p.numel() for p in model.parameters()) // 1000}K 参数）")
        return model, False


def _mk_engine(cfg_path: str = "configs/api.yaml") -> Engine:
    return Engine(load_config(cfg_path, APIConfig))


def create_app(engine: Optional[Engine] = None) -> FastAPI:
    eng = engine

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal eng
        if eng is None:
            eng = Engine(load_config("configs/api.yaml", APIConfig))
        yield

    app = FastAPI(title="tiny-gpt API", lifespan=lifespan)
    app.state.engine = eng

    def _settings(req):
        def _pick(name):
            v = getattr(req, name, None)
            return v if v is not None else getattr(eng.settings, name)
        return {
            "temperature": _pick("temperature"),
            "top_p": _pick("top_p"),
            "top_k": _pick("top_k"),
            "max_tokens": _pick("max_tokens"),
        }

    # ---------- /v1/models ----------
    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "tiny-gpt", "object": "model",
                                            "created": 0, "owned_by": "tiny-gpt"}]}

    # ---------- 消息/图像处理 ----------
    def _extract_content(message: ChatMessage) -> tuple:
        """返回 (text, image_tensor|None)。content 支持 str 或 OpenAI 风格 parts。"""
        text_parts: List[str] = []
        image = None
        if isinstance(message.content, str):
            return message.content, None
        for part in message.content:
            if not isinstance(part, dict):
                continue
            t = part.get("type")
            if t == "text":
                text_parts.append(part.get("text", ""))
            elif t in ("image_url", "image"):
                image = part.get("image_url") or part.get("image")
                url = image.get("url") if isinstance(image, dict) else image
                if url:
                    img = _decode_image(url)
                    image = img
        return "".join(text_parts), image

    def _decode_image(src: str) -> torch.Tensor:
        if src.startswith("data:"):
            b64 = src.split(",", 1)[1]
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        elif src.startswith("http"):
            import urllib.request
            with urllib.request.urlopen(src) as r:
                img = Image.open(io.BytesIO(r.read())).convert("RGB")
        else:
            img = Image.open(src).convert("RGB")
        size = eng.model.vision_cfg.image_size
        img = img.resize((size, size))
        arr = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return arr.unsqueeze(0)

    # ---------- /v1/completions ----------
    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        s = _settings(req)
        if eng.is_vision:
            raise HTTPException(400, "该模型为多模态模型，请使用 /v1/chat/completions")
        text = generate(eng.model, eng.tokenizer, req.prompt,
                        max_new_tokens=s["max_tokens"], temperature=s["temperature"],
                        top_k=s["top_k"], top_p=s["top_p"])
        return {"id": "cmpl-1", "object": "text_completion",
                "model": req.model, "choices": [{"text": text, "index": 0,
                                                 "finish_reason": "stop"}]}

    # ---------- /v1/chat/completions ----------
    def _chat_once(messages: List[dict], s: dict, with_vision: Optional[torch.Tensor] = None) -> str:
        if with_vision is not None:
            if not eng.is_vision:
                raise HTTPException(400, "模型不支持图像输入")
            return chat_vision(eng.model, eng.tokenizer, messages, with_vision,
                               max_new_tokens=s["max_tokens"], temperature=s["temperature"],
                               top_k=s["top_k"], top_p=s["top_p"])
        return chat(eng.model, eng.tokenizer, messages,
                    max_new_tokens=s["max_tokens"], temperature=s["temperature"],
                    top_k=s["top_k"], top_p=s["top_p"])

    def _to_openai_messages(req: ChatRequest) -> List[dict]:
        out: List[dict] = []
        image = None
        for m in req.messages:
            text, img = _extract_content(m)
            out.append({"role": m.role, "content": text})
            if img is not None:
                image = img
        return out, image

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        msgs, image = _to_openai_messages(req)
        s = _settings(req)
        if req.stream:
            from tinygpt.inference.chat import stream_chat
            async def gen() -> AsyncIterator[str]:
                yield _sse({"id": "cmpl-stream", "object": "chat.completion.chunk",
                            "model": req.model, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                                             "finish_reason": None}]})
                for token in stream_chat(eng.model, eng.tokenizer, msgs,
                                         max_new_tokens=s["max_tokens"],
                                         temperature=s["temperature"],
                                         top_k=s["top_k"], top_p=s["top_p"]):
                    yield _sse({"id": "cmpl-stream", "object": "chat.completion.chunk",
                                "model": req.model, "choices": [{"index": 0,
                                                                 "delta": {"content": token},
                                                                 "finish_reason": None}]})
                yield _sse({"id": "cmpl-stream", "object": "chat.completion.chunk",
                            "model": req.model, "choices": [{"index": 0, "delta": {},
                                                             "finish_reason": "stop"}]})
                yield "data: [DONE]\n\n"
            from fastapi.responses import StreamingResponse
            return StreamingResponse(gen(), media_type="text/event-stream")
        text = _chat_once(msgs, s, image)
        tool_call = parse_tool_call(text) if (req.tools and eng.cfg.capabilities.tools) else None
        if tool_call:
            return {"id": "cmpl-1", "object": "chat.completion", "model": req.model,
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant", "content": text,
                                             "tool_calls": [{"id": "call_1", "type": "function",
                                                             "function": tool_call}]},
                                 "finish_reason": "tool_calls"}]}
        return {"id": "cmpl-1", "object": "chat.completion", "model": req.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}]}

    # ---------- /v1/agent/completions（工具循环） ----------
    @app.post("/v1/agent/completions")
    def agent_completions(req: AgentRequest):
        msgs = [{"role": m.role, "content": m.content if isinstance(m.content, str)
                 else _extract_content(m)[0]} for m in req.messages]
        s = _settings(req)
        trace: List[dict] = []
        for turn in range(min(req.max_turns, 8)):
            text = chat(eng.model, eng.tokenizer, msgs,
                        max_new_tokens=s["max_tokens"], temperature=s["temperature"],
                        top_k=s["top_k"], top_p=s["top_p"])
            call = parse_tool_call(text)
            if call is None:
                return {"id": "agent-1", "object": "agent.completion", "model": req.model,
                        "content": text, "trace": trace, "turns": turn + 1}
            result = run_tool(call["name"], call["arguments"])
            trace.append({"turn": turn + 1, "tool_call": call, "result": result})
            msgs.append({"role": "assistant", "content": text})
            msgs.append({"role": "tool", "content": json.dumps(
                {"name": call["name"], "result": result}, ensure_ascii=False)})
            if not result.get("ok"):
                return {"id": "agent-1", "object": "agent.completion", "model": req.model,
                        "content": f"工具调用失败: {result.get('error')}", "trace": trace,
                        "turns": turn + 1, "error": True}
        return {"id": "agent-1", "object": "agent.completion", "model": req.model,
                "content": "达到最大工具循环轮次", "trace": trace, "turns": req.max_turns}

    # ---------- /v1/rag/completions ----------
    @app.post("/v1/rag/completions")
    def rag_completions(req: RAGRequest):
        if eng.kb is None:
            raise HTTPException(400, "RAG 未启用")
        prompt = eng.kb.build_prompt(req.query, top_k=req.top_k)
        s = _settings(req)
        text = chat(eng.model, eng.tokenizer,
                    [{"role": "user", "content": prompt}],
                    max_new_tokens=s["max_tokens"], temperature=s["temperature"],
                    top_k=s["top_k"], top_p=s["top_p"])
        hits = eng.kb.search(req.query, top_k=req.top_k)
        return {"id": "rag-1", "object": "rag.completion", "model": req.model,
                "query": req.query, "answer": text,
                "sources": [{"score": h["score"], "text": h["text"]} for h in hits]}

    return app


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


app = create_app()
