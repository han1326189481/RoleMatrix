"""评估小R 真人 LoRA：同一 prompt 下 无LoRA vs 有LoRA 出图对比。

产物：.tmp/lora_eval/{prompt编号}_{base|lora}.png
"""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import StableDiffusionPipeline

BASE_MODEL = r"D:\RoleMatrix\models\base\RealisticVision_V5.1"
LORA_PATH = r"D:\RoleMatrix\models\lora_xiaor_real_v1"
OUT_DIR = Path(r"D:\RoleMatrix\.tmp\lora_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    ("portrait", "photo of xiaor girl, portrait, selfie, indoor, soft natural light, looking at camera, detailed face, high quality"),
    ("life", "photo of xiaor girl, casual daily life, cozy room background, warm lighting, realistic photo, high quality"),
]
NEGATIVE = "worst quality, low quality, anime, cartoon, illustration, drawing, 3d render, bad anatomy, deformed"


def main() -> None:
    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, safety_checker=None,
        local_files_only=True,
    ).to("cuda")
    print(f"基模加载完成 {time.time()-t0:.1f}s", flush=True)

    for i, (tag, prompt) in enumerate(PROMPTS, 1):
        # 无 LoRA
        img = pipe(
            prompt=prompt, negative_prompt=NEGATIVE,
            width=512, height=512, num_inference_steps=25, guidance_scale=7.0,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
        img.save(OUT_DIR / f"{i}_{tag}_base.png")
        print(f"[{i}] 无LoRA已保存 {tag}", flush=True)

        # 有 LoRA
        pipe.load_lora_weights(LORA_PATH)
        img2 = pipe(
            prompt=prompt, negative_prompt=NEGATIVE,
            width=512, height=512, num_inference_steps=25, guidance_scale=7.0,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
        img2.save(OUT_DIR / f"{i}_{tag}_lora.png")
        pipe.unload_lora_weights()
        print(f"[{i}] 有LoRA已保存 {tag}", flush=True)

    print(f"评估图生成完成，总耗时 {time.time()-t0:.1f}s，输出: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
