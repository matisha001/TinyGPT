"""端到端 API 冒烟：启动真实权重服务 → 全端点调用 → 关闭。

用法：python scripts/e2e_api_smoke.py --config configs/api.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinygpt.config import APIConfig, load_config


def get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8")


def post(url: str, payload: dict, stream: bool = False):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/api.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config, APIConfig)
    if not Path(cfg.model.model_path).exists():
        raise SystemExit(f"checkpoint 不存在: {cfg.model.model_path}（先跑训练链）")
    host, port = cfg.server.host, cfg.server.port
    base = f"http://{host}:{port}"

    import subprocess
    proc = subprocess.Popen([sys.executable, "scripts/run_api.py",
                             "--config", args.config],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # 等待服务就绪
        for _ in range(60):
            try:
                urllib.request.urlopen(f"{base}/v1/models", timeout=2)
                break
            except Exception:                              # noqa: BLE001
                time.sleep(1)
        else:
            raise SystemExit("API 启动超时")

        models = get(f"{base}/v1/models")
        print("[1/5] /v1/models ->", models[:120])

        comp = post(f"{base}/v1/completions", {"prompt": "机器学习", "max_tokens": 32})
        print("[2/5] /v1/completions ->", comp[:150])

        chat = post(f"{base}/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "你好"}], "max_tokens": 32})
        print("[3/5] /v1/chat/completions ->", chat[:150])

        stream = post(f"{base}/v1/chat/completions",
                      {"messages": [{"role": "user", "content": "你好"}],
                       "stream": True, "max_tokens": 16})
        print("[4/5] 流式 chunks:", stream.count("data:"), "| DONE:", "data: [DONE]" in stream)

        rag = post(f"{base}/v1/rag/completions", {"query": "什么是 LoRA？", "top_k": 1,
                                                  "max_tokens": 32})
        print("[5/5] /v1/rag/completions ->", rag[:150])

        agent = post(f"{base}/v1/agent/completions",
                     {"messages": [{"role": "user", "content": "帮我算 2 加 3。"}],
                      "max_turns": 2, "max_tokens": 32})
        print("[+]  /v1/agent/completions ->", agent[:150])
        print("\n✅ 端到端 API 冒烟全部完成")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
