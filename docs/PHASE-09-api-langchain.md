# PHASE-09：OpenAI 兼容 API + LangChain 接入

> 分支：`p9-api-langchain` → 合并入 `main`

## 1. 本阶段目标

对外提供 OpenAI 兼容 HTTP 服务，并支持 LangChain `base_url` 直连 + `bind_tools`。

## 2. 端点（tinygpt/server/app.py，FastAPI）

| 端点 | 能力 |
|---|---|
| `GET /v1/models` | 模型列表 |
| `POST /v1/completions` | 传统续写（prompt → text） |
| `POST /v1/chat/completions` | 对话；支持 **流式 SSE**、**多模态**（content parts 的 `image_url`，base64 / 本地路径 / http）、**tools**（返回 `tool_calls` 消息） |
| `POST /v1/agent/completions` | **工具循环**：chat → `parse_tool_call` → `run_tool` → 结果回填，最多 `max_turns` 轮，返回 trace |
| `POST /v1/rag/completions` | 查询 → KB 检索 → 上下文注入 → 生成；返回 `sources` |

模型加载：`Engine._load_model` 根据 checkpoint 是否含 `vision_encoder` 键自动判别
多模态模型（`VisionGPT`）或文本模型（`GPT`）。

## 3. 启动与调用

```bash
# 终端1：启动服务（默认加载 checkpoints/dpo/dpo.pt）
.venv/bin/python scripts/run_api.py --config configs/api.yaml

# 终端2：curl 冒烟
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"stream":true}'
```

## 4. LangChain 接入（scripts/langchain_demo.py）

可选依赖 `langchain-openai` / `langchain-core`（用户明确暂不安装，脚本自带判断）：

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-tiny-gpt-demo",
                 model="tiny-gpt")
llm.invoke([HumanMessage(content="你好")])          # 普通对话
llm.bind_tools([...]).invoke([...])                 # 工具声明 → 模型可返回 tool_calls
```

## 5. 验收

- [x] 6 个端点全部通过 TestClient 集成测试（tests/test_api.py，6 passed）
- [x] 流式 SSE 格式符合 OpenAI chunk 协议（含 `data: [DONE]`）
- [x] agent 工具循环返回 content + trace + turns
- [x] RAG 端点返回检索来源
- [x] 分支 `p9-api-langchain` 推送 gitee，合并 `main`

## 6. 下一步

P10：真机端到端调优（真实 checkpoint 全端点冒烟、README/计划文档收口、远程推送校验）。
