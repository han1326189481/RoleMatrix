"""面部质量评审器：多 seed 抽卡后的选优评委。

用 Ollama 的 VL 模型（默认 qwen2.5vl:7b，视觉细节敏感型）逐张评审：
- 面部是否扭曲 / 眼镜是否清晰 / 眼睛是否正常 / 整体评分
输出按评分排序的优劣列表，供"抽卡选优"流水线使用。

用法：
  python scripts/judge_face.py <图片目录或文件> [--model qwen2.5vl:7b] [--top 3]
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import httpx

OLLAMA = "http://127.0.0.1:11434/api/generate"

# 评审 prompt：要求结构化 + 数值评分，聚焦面部细节
JUDGE_PROMPT = """你是一名严格的 AI 人像质量评审员，专门检查 AI 生成人像的面部缺陷。
仔细放大观察图中人物的脸，逐项严格检查并输出 JSON（不要输出其他文字）：
{
  "face": "正常|轻微扭曲|明显扭曲",
  "eyes": "正常|轻微异常|明显异常",
  "glasses": "无眼镜|清晰完整|轻微变形|明显变形",
  "face_score": 0-100,   // 面部自然度（100=真人般自然）
  "detail_score": 0-100, // 五官细节清晰度
  "overall_score": 0-100,
  "defects": ["缺陷列表，如'右眼镜片变形'、'左眼大小不一'，无则空数组"]
}
注意：眼镜佩戴者的镜片变形、眼睛不对称、五官糊成一团都是扣分项。"""


def judge_image(model: str, img_path: Path) -> dict:
    with httpx.Client(timeout=300) as c:
        r = c.post(
            OLLAMA,
            json={
                "model": model,
                "prompt": JUDGE_PROMPT,
                "images": [base64.b64encode(img_path.read_bytes()).decode()],
                "stream": False,
                "format": "json",
            },
        )
    raw = r.json().get("response", "")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 兜底：解析失败返回原始文本
        return {"_raw": raw}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="图片目录或单个图片文件")
    ap.add_argument("--model", default="qwen2.5vl:7b")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    target = Path(args.target)
    if target.is_dir():
        imgs = sorted(p for p in target.glob("*.png") if p.stat().st_size > 0)
    else:
        imgs = [target]

    results: list[tuple[float, Path, dict]] = []
    for p in imgs:
        print(f"评审 {p.name} ...", flush=True)
        try:
            verdict = judge_image(args.model, p)
        except Exception as e:  # noqa: BLE001
            print(f"  {p.name} 评审失败: {e}", flush=True)
            continue
        score = float(verdict.get("overall_score", 0) or 0)
        results.append((score, p, verdict))
        print(f"  {p.name}: face={verdict.get('face')} score={score} "
              f"defects={verdict.get('defects', [])}", flush=True)

    results.sort(reverse=True)
    print("\n===== 优劣排序（前 %d）=====" % args.top, flush=True)
    for i, (score, p, v) in enumerate(results[: args.top], 1):
        print(f"{i}. {p.name}  overall={score}  face={v.get('face')}  "
              f"detail={v.get('detail_score')}  defects={v.get('defects')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
