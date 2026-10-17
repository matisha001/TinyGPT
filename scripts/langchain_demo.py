"""LangChain 接入 demo：OpenAI 兼容 base_url 直连 + bind_tools。

先安装可选依赖（用户暂不装，本脚本提供安装后即可用的示例）：
    pip install langchain-openai langchain-core

用法：
    1) 启动服务：  python scripts/run_api.py
    2) 运行脚本：  python scripts/langchain_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_URL = os.environ.get("TINYGPT_API", "http://127.0.0.1:8000/v1")
API_KEY = "sk-tiny-gpt-demo"


def main() -> None:
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
    except ImportError as e:
        print(f"缺少可选依赖（{e}）。请先安装：pip install langchain-openai langchain-core")
        return

    # 1) base_url 直连本地 tiny-gpt
    llm = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model="tiny-gpt",
                     temperature=0.8, max_tokens=64)

    # 2) 普通对话
    resp = llm.invoke([HumanMessage(content="你好，请自我介绍一下。")])
    print("[chat]", resp.content)

    # 3) bind_tools：声明工具后由模型决定是否调用
    llm_tools = llm.bind_tools([
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "两个整数相加",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            },
        }
    ])
    out = llm_tools.invoke([HumanMessage(content="请计算 20 加 22 等于多少？")]
                           ).additional_kwargs.get("tool_calls")
    print("[tools]", out)


if __name__ == "__main__":
    main()
