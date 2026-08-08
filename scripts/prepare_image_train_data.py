"""小R 形象 LoRA 训练数据准备（真人风格）

输入: D:\\RoleMatrix\\training basic\\{侧脸照,全身照,正脸照,生活照}\\*.jpg
输出: D:\\RoleMatrix\\data\\image_train\\
  ├── side_01.jpg + side_01.txt
  ├── full_01.jpg + full_01.txt
  ├── face_01.jpg + face_01.txt
  └── life_01.jpg + life_01.txt

处理:
1. center crop 到 512x512（取中心正方形，保持人物完整）
2. 按子目录生成 caption .txt（真人风格 + 小R 触发词）

触发词: "xiaor girl"（训练后用此词生成小R 形象）
caption 策略: 触发词 + 场景描述 + 小R 基础特征（让 LoRA 学习人物特征）
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

# === 路径硬约束（rule/HARDWARE_AND_DEV_RULES.md 2.3 节）===
os.environ["HF_HOME"] = r"D:\RoleMatrix\.hf_cache"

SRC_ROOT = Path(r"D:\RoleMatrix\training basic")
DST_DIR = Path(r"D:\RoleMatrix\data\image_train")
RESOLUTION = 512

# 子目录 → 文件名前缀 + 场景描述
SUBDIR_MAP = {
    "侧脸照": ("side", "side profile photo, looking away"),
    "全身照": ("full", "full body photo, standing"),
    "正脸照": ("face", "portrait photo, facing camera, looking at viewer"),
    "生活照": ("life", "casual life photo, natural pose, indoor"),
}

# 小R 触发词 + 基础描述（真人风格，非二次元）
TRIGGER = "xiaor girl"
BASE_DESC = (
    "photorealistic, realistic photo, "
    "young asian girl, round face, round glasses, low ponytail, "
    "oversized hoodie, natural lighting, high quality, detailed face"
)


def center_crop_square(img: Image.Image, size: int) -> Image.Image:
    """center crop 到正方形，然后 resize 到 size×size。

    取中心正方形保证人物面部和身体完整，
    避免随意裁剪切掉头部或脚部。
    """
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    print("=" * 60)
    print("小R 形象 LoRA 训练数据准备（真人风格）")
    print("=" * 60)
    print(f"输入: {SRC_ROOT}")
    print(f"输出: {DST_DIR}")
    print(f"分辨率: {RESOLUTION}x{RESOLUTION}")
    print(f"触发词: {TRIGGER}")
    print("=" * 60)

    for subdir, (prefix, scene_desc) in SUBDIR_MAP.items():
        src_dir = SRC_ROOT / subdir
        if not src_dir.exists():
            print(f"跳过（目录不存在）: {src_dir}")
            continue
        files = sorted(list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")))
        print(f"\n[{subdir}] {len(files)} 张 → 前缀 {prefix}_")
        for i, f in enumerate(files, 1):
            try:
                img = Image.open(f).convert("RGB")
            except Exception as e:
                print(f"  跳过（读取失败）: {f.name} | {e}")
                continue
            orig_size = img.size
            img = center_crop_square(img, RESOLUTION)
            name = f"{prefix}_{i:02d}"
            img.save(DST_DIR / f"{name}.jpg", quality=95)
            caption = f"{TRIGGER}, {scene_desc}, {BASE_DESC}"
            (DST_DIR / f"{name}.txt").write_text(caption, encoding="utf-8")
            total += 1
            print(f"  {name}: {f.name} {orig_size[0]}x{orig_size[1]} → 512x512")

    print(f"\n完成: {total} 张图已处理到 {DST_DIR}")
    print(f"每张图配同名 .txt caption（触发词: {TRIGGER}）")
    print("\n下一步: 运行 scripts/train_xiaor_image_lora.py 开始训练")


if __name__ == "__main__":
    main()
