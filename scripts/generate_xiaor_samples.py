"""小R v4 四类样图生成：正脸 / 侧脸 / 生活 / 全身，每类 2 张不同 seed。

产物：.tmp/xiaor_samples/{类型}_{n}.png + prompts.txt（记录每张的 prompt）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

BASE_MODEL = r"D:\RoleMatrix\models\base\RealisticVision_V5.1"
LORA_PATH = r"D:\RoleMatrix\models\lora_xiaor_real_v4"
OUT_DIR = Path(r"D:\RoleMatrix\.tmp\xiaor_samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED = (
    "little_r, young east asian woman, slender soft face, soft oval face, baby face, "
    "black thick-rimmed rectangular glasses, shaggy layered hair, wispy bangs, "
    "fair skin, petite, calm cool vibe"
)
NEGATIVE = (
    "old woman, child, loli, muscular, skinny, athletic, heavy makeup, fashion model, "
    "revealing clothes, large breasts, blonde hair, colored hair, short hair, pixie cut, "
    "curly hair, high heels, business suit, strong expression, angry, aggressive, "
    "sharp face, pointed chin, male, anime, cartoon, illustration, drawing, 3d render, "
    "worst quality, low quality, bad anatomy, deformed, blurry, watermark, text"
)

# 四类场景：名称 -> (prompt 后缀, 出图尺寸)
SCENES = [
    ("正脸照", f"{FIXED}, portrait, selfie, looking at camera, indoor, warm lighting, cozy bedroom, phone selfie, realistic photo", 512, 512),
    ("侧脸照", f"{FIXED}, side profile, reading a book, library, bookshelf background, natural lighting, quiet atmosphere, candid, realistic", 512, 512),
    ("生活照", f"{FIXED}, candid daily life, looking at phone, night street, city lights, realistic lifestyle photography", 512, 512),
    ("全身照", f"{FIXED}, full body, standing, white hoodie with bold black letters, black pleated skirt, white socks, chunky sneakers, realistic photo", 448, 672),
]
SEEDS = [42, 2026]


def main() -> int:
    t0 = __import__("time").time()
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, safety_checker=None,
        local_files_only=True,
    ).to("cuda")

    # peft 手动加载 v4 LoRA（已验证的加载方式）
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
    print(f"模型加载完成 {__import__('time').time()-t0:.1f}s", flush=True)

    prompts_log = []
    for scene, prompt, w, h in SCENES:
        for n, seed in enumerate(SEEDS, 1):
            gen = torch.Generator("cuda").manual_seed(seed)
            img = pipe(
                prompt=prompt, negative_prompt=NEGATIVE,
                width=w, height=h,
                num_inference_steps=28, guidance_scale=7.0,
                generator=gen,
            ).images[0]
            out = OUT_DIR / f"{scene}_{n}.png"
            img.save(out)
            prompts_log.append(f"{out.name}\n  {prompt}\n  seed={seed}")
            print(f"已生成 {out.name}（{__import__('time').time()-t0:.1f}s）", flush=True)

    (OUT_DIR / "prompts.txt").write_text(
        "\n\n".join(prompts_log) + "\n\nNEGATIVE:\n" + NEGATIVE, encoding="utf-8"
    )
    print(f"全部完成，输出目录: {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
