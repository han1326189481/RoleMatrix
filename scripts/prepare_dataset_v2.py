"""小R 训练数据集 v2 准备。

依据专业视觉分析（Character Profile V1）：
1. 固定特征/可变化特征分离：每张图 caption = 固定前缀（触发词+不变特征）
   + 按图的可变部分（衣着/动作/场景/构图/摄影风格）
2. 图文一致：只有图里真戴眼镜才写 "black rectangular glasses"
3. 脸部特写增强：戴眼镜图 + 清晰正脸图裁剪上部区域为 close-up 样本，
   强化"脸+眼镜"的学习权重

输入：training basic_upscaled（超分后的图）
输出：data/lora_xiaor_real_v2/img（.png + .txt）
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

SRC = Path(r"D:\RoleMatrix\training basic_upscaled")
DST = Path(r"D:\RoleMatrix\data\lora_xiaor_real_v2\img")
DST.mkdir(parents=True, exist_ok=True)

# 固定前缀：不变的人物特征（不含眼镜，眼镜按图补充）
FIXED = (
    "little_r, young east asian woman, slightly chubby, soft round face, "
    "baby face, long black layered hair, wispy bangs, fair skin"
)

# 每张图的可变特征（按审查结果逐张编写；folder/name -> caption）
CAPTIONS: dict[tuple[str, str], str] = {
    ("侧脸照", "1.png"): "side profile, light top, indoor, cozy bedroom background, laptop and plush toys, warm lighting, candid, realistic photo",
    ("侧脸照", "2.png"): "reading a book, side profile, long sleeve shirt, library, bookshelf background, natural lighting, quiet atmosphere, candid photography, realistic",
    ("侧脸照", "3.png"): "black rectangular glasses, white top, coffee shop, warm indoor lighting, side profile, realistic candid photo",
    ("侧脸照", "4.png"): "night street, white top, city lights background, side profile, realistic candid photo",
    ("侧脸照", "5.png"): "sitting on train, white hoodie, side profile, thinking, natural light through window, realistic candid photo",
    ("全身照", "1.png"): "full body, oversized sweatshirt, black pleated skirt, black shoulder bag, white socks, indoor living room, warm lighting, standing, realistic",
    ("全身照", "2.png"): "full body, oversized white sweatshirt, black pleated skirt, black shoulder bag, white socks, outdoor by water, daylight, side view, realistic",
    ("全身照", "3.png"): "full body, oversized sweatshirt, black shorts, sneakers, outdoor waterfront, sunset, natural light, realistic",
    ("全身照", "4.png"): "black rectangular glasses, oversized beige sweatshirt, black pleated skirt, convenience store, indoor, front view, full body, realistic",
    ("全身照", "5.png"): "black rectangular glasses, oversized white sweatshirt, black pleated skirt, night street, street lights, full body, realistic",
    ("全身照", "6.png"): "black rectangular glasses, oversized white sweatshirt, dark pleated skirt, striped socks, indoor dim lighting, full body, realistic",
    ("全身照", "7.png"): "full body, oversized white sweatshirt, black pleated skirt, night city street, tall buildings lights, realistic",
    ("全身照", "9.png"): "sitting on stairs, white sweatshirt, dark pleated skirt, knee socks, chunky sneakers, indoor dim lighting, realistic",
    ("全身照", "10.png"): "black rectangular glasses, ponytail, oversized sweatshirt, black pleated skirt, white sneakers, parking lot, cold lighting, full body, realistic",
    ("正脸照", "1.png"): "front portrait, looking at camera, white hoodie, cozy bedroom, plush toys background, soft warm lighting, realistic close-up",
    ("正脸照", "2.png"): "front portrait, looking at camera, white hoodie, indoor, plush toys and shelf background, warm lighting, realistic",
    ("正脸照", "3.png"): "front portrait, looking at camera, white turtleneck sweater, cozy bedroom, bookshelf and plush toys, warm lighting, realistic",
    ("正脸照", "4.png"): "front portrait, looking at camera, white t-shirt, bedroom, soft lighting, realistic",
    ("正脸照", "5.png"): "front portrait, looking at camera, light long sleeve top, bedroom, computer screen glow, realistic",
    ("生活照", "1.png"): "holding phone, light top, cafe, window natural light, side view, candid, realistic daily life",
    ("生活照", "2.png"): "black and white photo, loose white top, black bag, bookstore, bookshelf background, side view, candid, realistic",
    ("生活照", "3.png"): "looking at phone, white hoodie, black pleated skirt, black backpack, night street, city lights, candid, realistic",
    ("生活照", "4.png"): "browsing groceries, white shirt and dark cardigan, crossbody bag, grocery store, side view, candid, realistic daily life",
    ("生活照", "6.png"): "sitting on subway, white sweater and dark vest, other passengers behind, side view, candid, realistic daily life",
    ("生活照", "8.png"): "ponytail, white top, black pleated skirt, dark tote bag, cafe counter, side profile, slightly overexposed, film photo style, realistic",
}

# 需要脸部特写裁剪的图（戴眼镜 5 张 + 清晰正脸 2 张作基准）
FACE_CROPS = [
    ("侧脸照", "3.png"),
    ("全身照", "4.png"),
    ("全身照", "5.png"),
    ("全身照", "6.png"),
    ("全身照", "10.png"),
    ("正脸照", "1.png"),
    ("正脸照", "2.png"),
]


def crop_face_upper(img: Image.Image, ratio: float = 0.5) -> Image.Image:
    """裁剪图片上部区域作为脸部特写（人物面部通常在图片上部）。"""
    w, h = img.size
    return img.crop((0, 0, w, int(h * ratio)))


def main() -> int:
    # 清理旧目录
    for old in DST.glob("*"):
        old.unlink()
    print(f"输出目录: {DST}", flush=True)

    n = 0
    for (folder, name), tail in CAPTIONS.items():
        src = SRC / folder / name
        if not src.exists():
            print(f"!! 缺失源图: {src}", flush=True)
            continue
        out_name = f"{folder[:2]}{name}"  # 如 侧脸照1.png -> 侧脸1.png（保持可追溯）
        img = Image.open(src).convert("RGB")
        img.save(DST / out_name)
        caption = f"{FIXED}, {tail}"
        (DST / (out_name.rsplit(".", 1)[0] + ".txt")).write_text(
            caption, encoding="utf-8"
        )
        n += 1

    # 脸部特写
    for folder, name in FACE_CROPS:
        src = SRC / folder / name
        if not src.exists():
            continue
        out_name = f"{folder[:2]}{name.rsplit('.', 1)[0]}_face.png"
        img = Image.open(src).convert("RGB")
        face = crop_face_upper(img)
        face.save(DST / out_name)
        has_glasses = "black rectangular glasses" in CAPTIONS[(folder, name)]
        extra = (
            "close-up portrait, face, black rectangular glasses, gentle gaze, "
            "looking at camera" if has_glasses
            else "close-up portrait, face, gentle gaze, looking at camera"
        )
        caption = f"{FIXED}, {extra}"
        (DST / (out_name.rsplit(".", 1)[0] + ".txt")).write_text(
            caption, encoding="utf-8"
        )
        n += 1

    print(f"数据集 v2 就绪：{n} 张（原图 {len(CAPTIONS)} + 脸部特写 {len(FACE_CROPS)}）", flush=True)

    # 打印带眼镜的样本数（验证眼镜强化）
    glasses_count = sum(
        1 for f in DST.glob("*.txt")
        if "black rectangular glasses" in f.read_text(encoding="utf-8")
    )
    print(f"含眼镜标注的样本：{glasses_count} 个", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
