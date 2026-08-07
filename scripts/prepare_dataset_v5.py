"""小R 训练数据集 v5：聚焦正脸 + 侧脸（依据用户指令：全身/生活照为灾区先剔除）。

关键改进：
1. 仅正脸 5 + 侧脸 5 + 新正脸 + 新侧脸（12 原图），全身/生活照全部剔除（等用户补高清图）
2. caption 依据重新识别结果 + 用户专业描述重建（场景准确：抓娃娃机/火车窗边等）
3. 脸部特写用 opencv 人脸检测（frontal + profile）精准裁剪，fallback 上部 45%
4. 发型按用户专业描述：straight + slightly wavy + fluffy messy + bangs + side bangs
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SRC = Path(r"D:\RoleMatrix\training basic")
DST = Path(r"D:\RoleMatrix\data\lora_xiaor_real_v5\img")
DST.mkdir(parents=True, exist_ok=True)
HC = Path(r"D:\RoleMatrix\.tmp\haarcascades")

# 固定前缀（v5）：保留人种/年龄锚定 + 用户专业描述的发型/主体特征
FIXED = (
    "little_r, 1girl, young asian woman, 20 years old, pale skin, natural makeup, "
    "long black straight hair, slightly wavy, fluffy messy hair, bangs, side bangs, "
    "black thick-rimmed rectangular glasses, calm cool vibe, serene"
)

# folder/name -> 场景 caption（依据重新识别 + 用户专业描述）
CAPTIONS: dict[tuple[str, str], str] = {
    ("侧脸照", "1.jpg"): "side profile, looking at laptop, indoor, cozy warm room, evening, soft lighting, candid, realistic",
    ("侧脸照", "2.jpg"): "side profile, reading a book, library, bookshelf background, bright soft lighting, quiet atmosphere, candid, realistic",
    ("侧脸照", "3.jpg"): "three quarter view, drinking, coffee shop, daytime, cold tone, bright lighting, calm, realistic",
    ("侧脸照", "4.jpg"): "side profile, playing arcade claw machine, night street, arcade machine, glass reflections, cool tone, neon lights, candid, realistic",
    ("侧脸照", "5.jpg"): "side profile, sitting by train window, daytime, train interior, window frame, daylight, natural light, quiet, realistic",
    ("正脸照", "1.jpg"): "front view, looking at camera, white hoodie, indoor, cozy bedroom, plush toys background, warm lighting, realistic",
    ("正脸照", "2.jpg"): "front view, looking at camera, white hoodie, indoor, plush toys and shelf background, warm lighting, realistic",
    ("正脸照", "3.jpg"): "front view, looking at camera, white turtleneck sweater, cozy bedroom, bookshelf and plush toys, warm lighting, realistic",
    ("正脸照", "4.jpg"): "front view, looking at camera, white t-shirt, bedroom, soft lighting, realistic",
    ("正脸照", "5.jpg"): "front view, looking at camera, light long sleeve top, bedroom, computer screen glow, realistic",
}

# 新生成的 2 张（正脸+侧脸，用户要求；全身照.png 暂不纳入，等用户补图）
NEW_IMAGES = {
    "正脸照.png": "front view, selfie, looking at camera, indoor, warm lighting, cozy bedroom, phone selfie, realistic",
    "侧脸照.png": "three quarter view, reading a book, library, bookshelf background, soft lighting, quiet atmosphere, candid, realistic",
}

# 脸部特写来源（全部原图）
FACE_CROPS = [
    ("侧脸照", "1.jpg"), ("侧脸照", "2.jpg"), ("侧脸照", "3.jpg"),
    ("侧脸照", "4.jpg"), ("侧脸照", "5.jpg"),
    ("正脸照", "1.jpg"), ("正脸照", "2.jpg"), ("正脸照", "3.jpg"),
    ("正脸照", "4.jpg"), ("正脸照", "5.jpg"),
    ("新", "正脸照.png"), ("新", "侧脸照.png"),
]

_face_cascade = cv2.CascadeClassifier(str(HC / "haarcascade_frontalface_default.xml"))
_profile_cascade = cv2.CascadeClassifier(str(HC / "haarcascade_profileface.xml"))


def _imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def detect_face_box(img: Image.Image) -> tuple[int, int, int, int] | None:
    """opencv 检测人脸（正脸 + 左右侧脸），返回 (x,y,w,h)，失败返回 None。"""
    arr = np.asarray(img.convert("RGB"))[:, :, ::-1]  # RGB->BGR
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    boxes = list(_face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30)))
    boxes += list(_profile_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30)))
    mirror = cv2.flip(gray, 1)
    boxes += list(_profile_cascade.detectMultiScale(mirror, 1.1, 5, minSize=(30, 30)))
    # 取最大的框（最可能的主脸）
    if not boxes:
        return None
    boxes = [b for b in boxes if b[2] >= w * 0.1 and b[3] >= h * 0.08]
    if not boxes:
        return None
    bx, by, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
    # 外扩 20% 确保包含完整脸+眼镜
    pad_x, pad_y = int(bw * 0.2), int(bh * 0.25)
    x = max(0, bx - pad_x)
    y = max(0, by - pad_y)
    x2 = min(w, bx + bw + pad_x)
    y2 = min(h, by + bh + pad_y)
    return (x, y, x2 - x, y2 - y)


def crop_face(img: Image.Image) -> Image.Image:
    box = detect_face_box(img)
    if box:
        x, y, w, h = box
        # 长宽比限制（防止过高）
        h = min(h, int(w * 1.4))
        return img.crop((x, y, x + w, y + h))
    # fallback：上部 45%
    w, h = img.size
    return img.crop((0, 0, w, int(h * 0.45)))


def _write(img: Image.Image, name: str, caption: str) -> None:
    img.save(DST / name)
    (DST / (name.rsplit(".", 1)[0] + ".txt")).write_text(caption, encoding="utf-8")


def _load(folder: str, name: str) -> Image.Image:
    path = SRC / name if folder == "新" else SRC / folder / name
    return Image.open(path).convert("RGB")


def main() -> int:
    for old in DST.glob("*"):
        old.unlink()
    print(f"输出目录: {DST}", flush=True)
    n = 0

    for (folder, name), tail in CAPTIONS.items():
        _write(_load(folder, name), f"{folder[:2]}{name.rsplit('.', 1)[0]}.png", f"{FIXED}, {tail}")
        n += 1
    for name, tail in NEW_IMAGES.items():
        _write(_load("新", name), name, f"{FIXED}, {tail}")
        n += 1

    # 脸部特写（opencv 精准裁剪）
    for folder, name in FACE_CROPS:
        img = _load(folder, name)
        face = crop_face(img)
        stem = (name if folder == "新" else f"{folder[:2]}{name.rsplit('.', 1)[0]}")
        _write(face, f"{stem}_face.png", f"{FIXED}, close-up portrait, face, gentle gaze")
        n += 1

    # 统计
    glasses = sum(
        1 for f in DST.glob("*.txt") if "rectangular glasses" in f.read_text(encoding="utf-8")
    )
    print(f"数据集 v5 就绪：{n} 张（原图 {len(CAPTIONS)+len(NEW_IMAGES)} + 脸部特写 {len(FACE_CROPS)}），含眼镜标注 {glasses}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
