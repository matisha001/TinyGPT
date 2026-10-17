"""轻量日志与指标：控制台 logger + JSONL 指标文件。"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def setup_logger(name: str = "tinygpt", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class MetricLogger:
    """追加写 JSONL 指标（每条一行），便于事后分析与可视化。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._f = None

    def log(self, **kwargs) -> None:
        if self._f is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self.path, "a", encoding="utf-8")
        record = {"ts": time.time(), **kwargs}
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
