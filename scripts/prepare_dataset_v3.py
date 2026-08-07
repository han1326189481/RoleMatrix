"""小R 训练数据集 v3 准备（纠正 v2 的眼镜漏判 + 混合数据源）。

关键修正（依据用户确认 + opencv 检测交叉验证）：
1. 正脸照 1-5、侧脸照 1-5 全部戴黑框眼镜——caption 全部加 glasses
   （v2 基于 minicpm-v 漏判写成无眼镜，错误）
2. 数据源策略（回应"超分破坏原图"质疑）：
   - 原图 ≥ 512px（正脸/侧脸/生活照/新 3 张）：直接用原图，信任原始风格
   - 原图 < 512px（全身照）：用超分图（否则 256px 太糊）
3. 脸部特写增强：从原图裁剪，强化"脸+眼镜"
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

SRC_ORIG = Path(r"D:\RoleMatrix\training basic")          # 原图（含新 3 张）
SRC_UPSCALED = Path(r"D:\RoleMatrix\training basic_upscaled")  # 超分图（仅全身照用）
DST = Path(r"D:\RoleMatrix\data\lora_xiaor_real_v3\img")
DST.mkdir(parents=True, exist_ok=True)

FIXED = (
    "little_r, young east asian woman, slightly chubby, soft round face, "
    "baby face, long black layered hair, wispy bangs, fair skin"
)
GLASSES = "black rectangular glasses"

# folder/name -> (可变caption, 是否用原图, 是否戴眼镜)
# 全身照用超分图（origin=False）
CAPTIONS: dict[tuple[str, str], tuple[str, bool, bool]] = {
    ("侧脸照", "1.jpg"): ("side profile, light top, indoor, cozy bedroom background, laptop and plush toys, warm lighting, candid, realistic photo", True, True),
    ("侧脸照", "2.jpg"): ("reading a book, side profile, long sleeve shirt, library, bookshelf background, natural lighting, quiet atmosphere, candid photography, realistic", True, True),
    ("侧脸照", "3.jpg"): ("white top, coffee shop, warm indoor lighting, side profile, realistic candid photo", True, True),
    ("侧脸照", "4.jpg"): ("night street, white top, city lights background, side profile, realistic candid photo", True, True),
    ("侧脸照", "5.jpg"): ("sitting on train, white hoodie, side profile, thinking, natural light through window, realistic candid photo", True, True),
    ("全身照", "1.jpg"): ("full body, oversized sweatshirt, black pleated skirt, black shoulder bag, white socks, indoor living room, warm lighting, standing, realistic", False, False),
    ("全身照", "2.jpg"): ("full body, oversized white sweatshirt, black pleated skirt, black shoulder bag, white socks, outdoor by water, daylight, side view, realistic", False, False),
    ("全身照", "3.jpg"): ("full body, oversized sweatshirt, black shorts, sneakers, outdoor waterfront, sunset, natural light, realistic", False, False),
    ("全身照", "4.jpg"): ("oversized beige sweatshirt, black pleated skirt, convenience store, indoor, front view, full body, realistic", False, True),
    ("全身照", "5.jpg"): ("oversized white sweatshirt, black pleated skirt, night street, street lights, full body, realistic", False, True),
    ("全身照", "6.jpg"): ("oversized white sweatshirt, dark pleated skirt, striped socks, indoor dim lighting, full body, realistic", False, True),
    ("全身照", "7.jpg"): ("full body, oversized white sweatshirt, black pleated skirt, night city street, tall buildings lights, realistic", False, False),
    ("全身照", "9.jpg"): ("sitting on stairs, white sweatshirt, dark pleated skirt, knee socks, chunky sneakers, indoor dim lighting, realistic", False, False),
    ("全身照", "10.jpg"): ("ponytail, oversized sweatshirt, black pleated skirt, white sneakers, parking lot, cold lighting, full body, realistic", False, True),
    ("正脸照", "1.jpg"): ("front portrait, looking at camera, white hoodie, cozy bedroom, plush toys background, soft warm lighting, realistic close-up", True, True),
    ("正脸照", "2.jpg"): ("front portrait, looking at camera, white hoodie, indoor, plush toys and shelf background, warm lighting, realistic", True, True),
    ("正脸照", "3.jpg"): ("front portrait, looking at camera, white turtleneck sweater, cozy bedroom, bookshelf and plush toys, warm lighting, realistic", True, True),
    ("正脸照", "4.jpg"): ("front portrait, looking at camera, white t-shirt, bedroom, soft lighting, realistic", True, True),
    ("正脸照", "5.jpg"): ("front portrait, looking at camera, light long sleeve top, bedroom, computer screen glow, realistic", True, True),
    ("生活照", "1.jpg"): ("holding phone, light top, cafe, window natural light, side view, candid, realistic daily life", True, False),
    ("生活照", "2.jpg"): ("black and white photo, loose white top, black bag, bookstore, bookshelf background, side view, candid, realistic", True, False),
    ("生活照", "3.jpg"): ("looking at phone, white hoodie, black pleated skirt, black backpack, night street, city lights, candid, realistic", True, False),
    ("生活照", "4.jpg"): ("browsing groceries, white shirt and dark cardigan, crossbody bag, grocery store, side view, candid, realistic daily life", True, False),
    ("生活照", "6.jpg"): ("sitting on subway, white sweater and dark vest, other passengers behind, side view, candid, realistic daily life", True, False),
    ("生活照", "8.jpg"): ("ponytail, white top, black pleated skirt, dark tote bag, cafe counter, side profile, slightly overexposed, film photo style, realistic", True, False),
}

# 新生成的 3 张（在 training basic 根目录，原图）
NEW_IMAGES = {
    "正脸照.png": ("front portrait, selfie, looking at camera, indoor, warm lighting, cozy bedroom, phone selfie, realistic", True),
    "侧脸照.png": ("side profile, reading book, library, bookshelf background, natural lighting, quiet atmosphere, candid, realistic", True),
    "全身照.png": ("full body, standing, casual outfit, slightly fitted top, pleated skirt, university campus, natural light, realistic", True),
}

# 脸部特写（从原图裁剪）：正脸 5 + 侧脸 5 + 新正脸
FACE_CROPS = [
    ("正脸照", "1.jpg"), ("正脸照", "2.jpg"), ("正脸照", "3.jpg"),
    ("正脸照", "4.jpg"), ("正脸照", "5.jpg"),
    ("侧脸照", "1.jpg"), ("侧脸照", "2.jpg"), ("侧脸照", "3.jpg"),
    ("侧脸照", "4.jpg"), ("侧脸照", "5.jpg"),
]
NEW_FACE_CROPS = ["正脸照.png"]


def _load(folder: str, name: str, use_orig: bool) -> Image.Image:
    if folder == "新":
        return Image.open(SRC_ORIG / name).convert("RGB")
    base = SRC_ORIG if use_orig else SRC_UPSCALED
    fname = name if use_orig else name.replace(".jpg", ".png")
    return Image.open(base / folder / fname).convert("RGB")


def _write(img: Image.Image, out_name: str, caption: str) -> None:
    img.save(DST / out_name)
    (DST / (out_name.rsplit(".", 1)[0] + ".txt")).write_text(caption, encoding="utf-8")


def _caption(tail: str, has_glasses: bool) -> str:
    g = f", {GLASSES}" if has_glasses else ""
    return f"{FIXED}{g}, {tail}"


def main() -> int:
    for old in DST.glob("*"):
        old.unlink()
    print(f"输出目录: {DST}", flush=True)

    n = 0
    # 原 25 张
    for (folder, name), (tail, use_orig, has_gl) in CAPTIONS.items():
        img = _load(folder, name, use_orig)
        out_name = f"{folder[:2]}{name.rsplit('.', 1)[0]}.png"
        _write(img, out_name, _caption(tail, has_gl))
        n += 1
        src_tag = "原图" if use_orig else "超分"
        print(f"  [{src_tag}] {folder}/{name} glasses={has_gl}", flush=True)

    # 新 3 张
    for name, (tail, has_gl) in NEW_IMAGES.items():
        img = _load("新", name, True)
        _write(img, name, _caption(tail, has_gl))
        n += 1
        print(f"  [原图] 新/{name} glasses={has_gl}", flush=True)

    # 脸部特写（原图裁剪）
    for folder, name in FACE_CROPS:
        img = _load(folder, name, True)
        w, h = img.size
        face = img.crop((0, 0, w, int(h * 0.5)))
        out_name = f"{folder[:2]}{name.rsplit('.', 1)[0]}_face.png"
        _write(face, out_name, _caption(
            "close-up portrait, face, gentle gaze, looking at camera", True
        ))
        n += 1
    for name in NEW_FACE_CROPS:
        img = _load("新", name, True)
        w, h = img.size
        face = img.crop((0, 0, w, int(h * 0.5)))
        out_name = name.rsplit(".", 1)[0] + "_face.png"
        _write(face, out_name, _caption(
            "close-up portrait, face, gentle gaze, looking at camera", True
        ))
        n += 1

    glasses_count = sum(
        1 for f in DST.glob("*.txt")
        if GLASSES in f.read_text(encoding="utf-8")
    )
    print(f"\n数据集 v3 就绪：{n} 张，含眼镜标注 {glasses_count} 个", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
