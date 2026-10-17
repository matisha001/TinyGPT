"""启动 OpenAI 兼容 API：python scripts/run_api.py --config configs/api.yaml"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

from tinygpt.config import APIConfig, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/api.yaml")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, APIConfig)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    print(f"tiny-gpt API → http://{host}:{port}  (OpenAI 兼容: /v1/models, /v1/chat/completions, "
          f"/v1/completions, /v1/agent/completions, /v1/rag/completions)")
    uvicorn.run("tinygpt.server.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
