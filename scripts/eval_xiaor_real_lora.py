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
LORA_PATH = r"D:\RoleMatrix\models\lora_xiaor_real_v5"
OUT_DIR = Path(r"D:\RoleMatrix\.tmp\lora_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# v5 特征前缀：保留 east asian 人种锚定 + 年龄锚定（v6 修正结论）
FIXED = (
    "little_r, young east asian woman, 20 years old, college student, "
    "slender soft face, soft oval face, baby face, "
    "black thick-rimmed rectangular glasses, shaggy layered hair, wispy bangs, "
    "fair skin, petite, calm cool vibe"
)
PROMPTS = [
    ("portrait", f"{FIXED}, portrait, selfie, looking at camera, indoor, warm lighting, cozy bedroom, phone selfie, realistic photo"),
    ("life", f"{FIXED}, casual daily life, cozy room, warm lighting, realistic photo"),
]
NEGATIVE = (
    "old woman, child, loli, muscular, skinny, athletic, heavy makeup, fashion model, "
    "revealing clothes, large breasts, blonde hair, colored hair, short hair, pixie cut, "
    "curly hair, high heels, business suit, strong expression, angry, aggressive, "
    "sharp face, pointed chin, male, anime, cartoon, illustration, drawing, 3d render, "
    "worst quality, low quality, bad anatomy, deformed, blurry, watermark, text"
)


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

        # 有 LoRA（peft 手动加载：diffusers load_lora_weights 不认 peft 前缀）
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file

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
