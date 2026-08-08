"""小R v6 三类样图生成：正脸 / 侧脸 / 全身，供用户专业模型确认。

Prompt 基于 v6 训练 caption 的 FIXED（左眼下小痣 + 碎刘海 + 粗黑框眼镜）。
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
LORA_PATH = r"D:\RoleMatrix\models\lora_xiaor_real_v6"
OUT_DIR = Path(r"D:\RoleMatrix\.tmp\xiaor_samples_v6")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED = (
    "little_r, 1girl, solo, young asian woman, 20 years old, slender soft face, "
    "soft oval face, black long hair, messy bangs, wispy bangs, tousled hair, "
    "black thick rectangular glasses, pale skin, soft blush, pink lips, light makeup, "
    "beauty mark under left eye"
)
NEGATIVE = (
    "old woman, child, loli, elementary school student, preteen, teenage girl, "
    "baby face, muscular, skinny, heavy makeup, blonde hair, colored hair, short hair, "
    "pixie cut, curly hair, high heels, business suit, strong expression, angry, "
    "aggressive, sharp face, pointed chin, male, caucasian, western face, "
    "anime, cartoon, illustration, drawing, 3d render, worst quality, low quality, "
    "bad anatomy, deformed, distorted face, blurry, watermark, text"
)

SCENES = [
    ("正脸", f"{FIXED}, front view, looking at viewer, hand on cheek, white oversized hoodie, indoor, shelf, plushies, warm lighting, realistic", 512, 512),
    ("侧脸_图书馆", f"{FIXED}, side profile, looking down, reading, library, bookshelf, bright soft lighting, quiet atmosphere, realistic", 512, 512),
    ("侧脸_娃娃机", f"{FIXED}, side profile, night street, illuminated display case, neon lights, cool and warm light, pensive, realistic", 512, 512),
    ("全身_校园", f"{FIXED}, full body, standing, hands in pockets, cream white hoodie with black text, black pleated mini skirt, white mid-calf socks, white chunky sneakers, black shoulder bag, outdoors, campus, sunny, dappled sunlight, tree shadows, realistic", 448, 672),
]
SEEDS = [2026, 777]


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
