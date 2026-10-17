"""生成多模态 demo 数据：8 张纯色小图 + 问答 jsonl。
python scripts/gen_mm_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

OUT = Path("data/demo/mm")
COLORS = ["红色", "绿色", "蓝色", "黄色", "紫色", "橙色", "青色", "灰色"]
RGB = [(220, 60, 60), (60, 180, 60), (60, 90, 220), (240, 200, 40),
       (150, 60, 200), (240, 130, 40), (60, 190, 190), (130, 130, 130)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (name, rgb) in enumerate(zip(COLORS, RGB)):
        img = Image.new("RGB", (64, 64), rgb)
        # 加一个简单几何图形便于肉眼区分
        d = ImageDraw.Draw(img)
        d.ellipse([16, 16, 48, 48], fill=(255, 255, 255))
        path = OUT / f"img_{i}.png"
        img.save(path)
        rows.append({"image": str(path), "question": "这张图片的主色调是什么？",
                     "answer": f"主色调是{name}。"})
    (OUT / "mm_train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"已生成 {len(rows)} 张 demo 图片: {OUT}")


if __name__ == "__main__":
    main()
