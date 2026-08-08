"""用 minicpm-v 视觉模型分析训练图片，生成准确的 LoRA caption。

之前 caption 全靠猜，导致 LoRA 学到错误特征。
现在用视觉模型实际看图，生成准确描述。

流程:
1. 逐张读训练图片
2. minicpm-v 描述图片内容（人物特征 + 场景）
3. 生成英文 caption（xiaor girl 触发词 + 准确特征）
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "minicpm-v:latest"

DATA_DIR = Path(r"D:\RoleMatrix\data\image_train")
OUT_DIR = Path(r"D:\RoleMatrix\data\image_train_captions")

PROMPT = """Describe this photo for AI training. Output ONLY a caption line in English, no explanation.

Format: "xiaor girl, [scene], [person features: gender/age/ethnicity/hair/face/glasses/clothing], [lighting/quality]"

Rules:
- Start with "xiaor girl, "
- Describe ONLY what you actually see in the photo
- Person features must be specific and accurate (hair color/style, glasses type, clothing)
- Do NOT guess or add features not visible
- Keep it under 60 words
- End with "photorealistic, high quality"

Example: "xiaor girl, portrait photo facing camera, young asian woman, long black hair, round glasses, white t-shirt, natural lighting, photorealistic, high quality"
"""


def encode_image(path: Path) -> str:
    """ollama API 只接受纯 base64 字符串（不带 data URI 前缀）。"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def describe_image(path: Path, retries: int = 2) -> str:
    img_b64 = encode_image(path)
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [img_b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 200},
    }
    for attempt in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=60)
            r.raise_for_status()
            caption = r.json()["message"]["content"].strip().strip('"').strip()
            # 确保以 xiaor girl 开头
            if not caption.lower().startswith("xiaor girl"):
                caption = "xiaor girl, " + caption
            return caption
        except Exception as e:
            print(f"  重试 {attempt+1}/{retries}: {e}")
            time.sleep(2)
    return ""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.png")))
    print(f"分析 {len(images)} 张训练图片...")

    results: dict[str, str] = {}
    t0 = time.time()
    for i, img in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {img.name}")
        caption = describe_image(img)
        if caption:
            print(f"  -> {caption}")
            results[img.stem] = caption
            # 同步写 .txt
            (OUT_DIR / f"{img.stem}.txt").write_text(caption, encoding="utf-8")
        else:
            print(f"  -> 失败")

    # 汇总输出
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"完成: {len(results)}/{len(images)} 张, 耗时 {elapsed:.1f}s")
    print(f"caption 保存到: {OUT_DIR}")
    print("=" * 60)
    print("\ncaption 汇总:")
    for name, cap in results.items():
        print(f"  {name}: {cap}")


if __name__ == "__main__":
    main()
