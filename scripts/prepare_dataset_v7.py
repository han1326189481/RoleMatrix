"""小R 训练数据集 v6：基于三份图片解释文档（专业视觉模型标注）。

数据：19 张原图（旧正脸 5 + 旧侧脸 5 + 补充 9：正脸2/侧脸3/全身4）+ 脸部特写。

Caption 要点（全部来自解释文档）：
- FIXED 统一特征：black long hair, messy bangs, wispy bangs, tousled hair
  + black thick rectangular glasses + pale skin, soft blush, pink lips
  + beauty mark under left eye（左眼下小痣，个人特征锚点）
- 字母卫衣统一 hoodie with black text（绝不写 BATMAN/SCUMMER 具体字样）
- studded glasses（铆钉）仅加在明确带铆钉的正脸图
- 全身照强化：black pleated mini skirt, white mid-calf socks, white chunky sneakers,
  black shoulder bag, dappled sunlight, campus（防只出半身）
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SRC = Path(r"D:\RoleMatrix\training basic")
DST = Path(r"D:\RoleMatrix\data\lora_xiaor_real_v7\img")
DST.mkdir(parents=True, exist_ok=True)
HC = Path(r"D:\RoleMatrix\.tmp\haarcascades")

# v7 变更：
# 1. 角色美化：FIXED 加温和美化词（beautiful/pretty/delicate features/clean skin，不过度）
# 2. 削弱侧脸镜头感：移除唯一直视镜头的样本（补充/侧脸照1，解释文档明确'眼神直视镜头'）；
#    其余侧脸 caption 强化视线方向（looking down / looking away）
FIXED = (
    "little_r, 1girl, solo, young asian woman, 20 years old, beautiful, pretty, "
    "delicate features, clean skin, slender soft face, soft oval face, "
    "black long hair, messy bangs, wispy bangs, tousled hair, "
    "black thick rectangular glasses, pale skin, soft blush, pink lips, light makeup, "
    "beauty mark under left eye"
)

# 旧正脸照 5 张（folder, name, 可变caption）
OLD_FRONT = {
    "1.jpg": "front view, looking at viewer, hand on cheek, side braid, studded glasses, white oversized hoodie, indoor, shelf, plushies, warm lighting, realistic",
    "2.jpg": "front view, looking at viewer, hand on cheek, studded glasses, white oversized hoodie, indoor, shelf, plushies, warm lighting, realistic",
    "3.jpg": "front view, looking at viewer, hand on cheek, tousled hair, studded glasses, white oversized hoodie, indoor, shelf, plushies, warm lighting, realistic",
    "4.jpg": "front view, looking at viewer, hand near mouth, studded glasses, white hoodie with black text, indoor, warm lighting, realistic",
    "5.jpg": "front view, looking at viewer, side braid, touching glasses, studded glasses, white oversized hoodie, indoor, desk, figurines, laptop, realistic",
}
# 旧侧脸照 5 张（视线方向强化：全部 looking down / looking away，削弱镜头感）
OLD_SIDE = {
    "1.jpg": "three quarter view, looking down, focused, white oversized hoodie, indoor, desk, plushies shelf, laptop, dim lighting, realistic",
    "2.jpg": "side profile, looking down, reading, library, wooden bookshelf, bright lighting, quiet atmosphere, realistic",
    "3.jpg": "side profile, looking down at drink, holding drink straw, coffee shop, warm pendant light, casual, realistic",
    "4.jpg": "side profile, looking away, gazing at display case, night street, illuminated display case, neon lights, cool and warm light, pensive, realistic",
    "5.jpg": "side profile, looking out window, hand on chin, bus window, daytime, natural light, realistic",
}
# 补充照片（高清）
NEW = {
    "正脸照1.png": "front view, looking at viewer, studded glasses, white hoodie with black text, indoor, shelf, plushies, figurines, computer screen glow, warm lighting, realistic",
    "正脸照2.png": "front view, looking at viewer, hands in pockets, white hoodie with black text, black pleated mini skirt, black shoulder bag, outdoors, campus, sunny, dappled sunlight, tree shadows, realistic",
    # 补充侧脸照1 已移除（唯一'眼神直视镜头'的侧脸样本，削弱镜头感）
    "侧脸照2.png": "side profile, looking down at book, reading, white hoodie with black text, desk, light blue pencil case, window, indoor, warm and cool light, realistic",
    "侧脸照3.png": "three quarter view, looking down, writing, holding black pen, public reading room, bookshelf, bright even lighting, realistic",
    "全身照1.png": "full body, standing, looking down, cream white hoodie with black text, black pleated mini skirt, white mid-calf socks, white chunky sneakers, black shoulder bag, outdoors, campus, sunny, dappled sunlight, tree shadows, realistic",
    "全身照2.png": "full body, walking, hands in pockets, cream white hoodie with black text, black pleated mini skirt, white mid-calf socks, white chunky sneakers, black shoulder bag, campus path, trees, sunny, dappled sunlight, realistic",
    "全身照3.png": "full body, walking, carrying black shoulder bag, small plush bag charm, looking down, cream white hoodie with black text, black pleated mini skirt, white mid-calf socks, white chunky sneakers, campus, sunny, dappled sunlight, realistic",
    "全身照4.png": "full body, reading book, holding book, side view, cream white hoodie with black text, black pleated mini skirt, white mid-calf socks, library, bookshelf, indoor, cool lighting, realistic",
}

# 脸部特写来源（全部 19 张原图）
FACE_CROPS = (
    [( "正脸照", k) for k in OLD_FRONT]
    + [("侧脸照", k) for k in OLD_SIDE]
    + [("补充照片", k) for k in NEW]
)

_face_cascade = cv2.CascadeClassifier(str(HC / "haarcascade_frontalface_default.xml"))
_profile_cascade = cv2.CascadeClassifier(str(HC / "haarcascade_profileface.xml"))


def _imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def crop_face(img: Image.Image) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    boxes = list(_face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30)))
    boxes += list(_profile_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30)))
    boxes += list(_profile_cascade.detectMultiScale(cv2.flip(gray, 1), 1.1, 5, minSize=(30, 30)))
    boxes = [b for b in boxes if b[2] >= w * 0.08 and b[3] >= h * 0.06]
    if boxes:
        bx, by, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
        px, py = int(bw * 0.2), int(bh * 0.25)
        x, y = max(0, bx - px), max(0, by - py)
        x2, y2 = min(w, bx + bw + px), min(h, by + bh + py)
        box_h = min(y2 - y, int((x2 - x) * 1.4))
        return img.crop((x, y, x2, y + box_h))
    # fallback 上部 40%
    return img.crop((0, 0, w, int(h * 0.40)))


def _load(folder: str, name: str) -> Image.Image:
    return Image.open(SRC / folder / name).convert("RGB")


def _write(img: Image.Image, name: str, caption: str) -> None:
    img.save(DST / name)
    (DST / (name.rsplit(".", 1)[0] + ".txt")).write_text(caption, encoding="utf-8")


def main() -> int:
    for old in DST.glob("*"):
        old.unlink()
    n = 0
    # 旧正脸/侧脸
    for folder, mapping in (("正脸照", OLD_FRONT), ("侧脸照", OLD_SIDE)):
        for name, tail in mapping.items():
            stem = f"{folder[:2]}{name.rsplit('.', 1)[0]}"
            _write(_load(folder, name), f"{stem}.png", f"{FIXED}, {tail}")
            n += 1
    # 补充照片
    for name, tail in NEW.items():
        _write(_load("补充照片", name), name, f"{FIXED}, {tail}")
        n += 1
    # 脸部特写
    for folder, name in FACE_CROPS:
        img = _load(folder, name)
        face = crop_face(img)
        stem = name.rsplit(".", 1)[0]
        if folder != "补充照片":
            stem = f"{folder[:2]}{stem}"
        _write(face, f"{stem}_face.png", f"{FIXED}, close-up portrait, face")
        n += 1

    glasses = sum(
        1 for f in DST.glob("*.txt") if "rectangular glasses" in f.read_text(encoding="utf-8")
    )
    studs = sum(
        1 for f in DST.glob("*.txt") if "studded" in f.read_text(encoding="utf-8")
    )
    print(f"数据集 v6 就绪：{n} 张（原图 19 + 脸部特写 19），眼镜 {glasses}，铆钉 {studs}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
