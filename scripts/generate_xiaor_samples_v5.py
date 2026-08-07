"""prompt 修正验证：年龄锚定 + 强化 negative，重生成侧脸/生活/全身。

针对 v4 样图问题（侧脸/全身幼态化、生活照换人）的快速验证。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

BASE_MODEL = r"D:\RoleMatrix\models\base\RealisticVision_V5.1"
LORA_PATH = r"D:\RoleMatrix\models\lora_xiaor_real_v4"
OUT_DIR = Path(r"D:\RoleMatrix\.tmp\xiaor_samples_v5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 年龄锚定：显式锁定"20 岁大学女生"（对抗幼态化）
FIXED = (
    "little_r, young adult woman, 20 years old, college student, "
    "slender soft face, soft oval face, black thick-rimmed rectangular glasses, "
    "shaggy layered hair, wispy bangs, fair skin, petite, calm cool vibe"
)
# negative 强化年龄下限（child/loli/preteen 拉黑）
NEGATIVE = (
    "old woman, child, loli, elementary school student, preteen, teenage girl, "
    "baby face, chibi, petite child, muscular, skinny, athletic, heavy makeup, "
    "fashion model, revealing clothes, large breasts, blonde hair, colored hair, "
    "short hair, pixie cut, curly hair, high heels, business suit, strong expression, "
    "angry, aggressive, sharp face, pointed chin, male, anime, cartoon, illustration, "
    "drawing, 3d render, worst quality, low quality, bad anatomy, deformed, "
    "distorted face, blurry, watermark, text"
)

SCENES = [
    ("侧脸照", f"{FIXED}, side profile, reading a book, library, bookshelf background, natural lighting, quiet atmosphere, candid, realistic, detailed face", 512, 512),
    ("生活照", f"{FIXED}, candid daily life, looking at phone, night street, city lights, realistic lifestyle photography, detailed face", 512, 512),
    ("全身照", f"{FIXED}, full body, standing, white hoodie with bold black letters, black pleated skirt, white socks, chunky sneakers, realistic photo, detailed face, clear facial features", 448, 672),
]
SEEDS = [42, 2026]


def main() -> int:
    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, safety_checker=None,
        local_files_only=True,
    ).to("cuda")
    lora_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    sd = load_file(Path(LORA_PATH) / "adapter_model.safetensors")
    sd = {k.replace("base_model.model.", ""): v for k, v in sd.items()}
    set_peft_model_state_dict(pipe.unet, sd)
    pipe.unet.eval()
    print(f"模型加载 {time.time()-t0:.1f}s", flush=True)

    for scene, prompt, w, h in SCENES:
        for n, seed in enumerate(SEEDS, 1):
            img = pipe(
                prompt=prompt, negative_prompt=NEGATIVE,
                width=w, height=h,
                num_inference_steps=30, guidance_scale=7.5,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            img.save(OUT_DIR / f"{scene}_{n}.png")
            print(f"已生成 {scene}_{n}（{time.time()-t0:.1f}s）", flush=True)
    print(f"完成: {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
