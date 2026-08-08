"""P0 验证：token 截断是否导致关键身份特征丢失。

对比两组 prompt（同一 v7 LoRA）：
- A 精简版：Identity Tokens，控制在 CLIP 77 token 内
- B 对照组：v7 完整版（100+ token，之前一直超长）

验证：眼镜框型/圆脸/小痣/碎刘海 是否在 A 中恢复。
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
LORA_PATH = r"D:\RoleMatrix\models\lora_xiaor_real_v7"
OUT_DIR = Path(r"D:\RoleMatrix\.tmp\p0_verify")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# A 精简版：身份特征全部前移，目标 ≤77 token
PROMPT_A_FRONT = (
    "little_r, young adult East Asian woman, soft rounded oval face, full cheeks, "
    "long messy black hair, wispy bangs, thick black rectangular glasses, "
    "droopy gentle eyes, beauty mark under left eye, neutral aloof mood, "
    "white oversized hoodie, sleeves over hands, indoor shelf plushies warm light"
)
PROMPT_A_SIDE = (
    "little_r, young adult East Asian woman, soft rounded oval face, full cheeks, "
    "long messy black hair, wispy bangs, thick black rectangular glasses, "
    "droopy gentle eyes, beauty mark under left eye, side profile looking down, "
    "library bookshelf, reading book, natural light, candid"
)

# B 对照组：v7 完整版（超长，特征靠后）
PROMPT_B = (
    "little_r, 1girl, solo, young asian woman, 20 years old, beautiful, pretty, "
    "delicate features, clean skin, slender soft face, soft oval face, "
    "black long hair, messy bangs, wispy bangs, tousled hair, "
    "black thick rectangular glasses, pale skin, soft blush, pink lips, light makeup, "
    "beauty mark under left eye, front view, looking at viewer, hand on cheek, "
    "white oversized hoodie, indoor, shelf, plushies, warm lighting, realistic"
)

NEGATIVE = (
    "old woman, child, loli, muscular, skinny, heavy makeup, blonde hair, "
    "colored hair, short hair, male, caucasian, western face, sharp jawline, "
    "anime, cartoon, worst quality, low quality, bad anatomy, deformed, blurry"
)


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

    # 打印 token 数验证
    for name, p in [
        ("A_正脸", PROMPT_A_FRONT), ("A_侧脸", PROMPT_A_SIDE), ("B_对照组", PROMPT_B),
    ]:
        tokens = pipe.tokenizer(p, truncation=False).input_ids
        print(f"{name}: {len(tokens)} tokens", flush=True)

    jobs = [
        ("A_正脸", PROMPT_A_FRONT, 512, 512, 2026),
        ("A_侧脸", PROMPT_A_SIDE, 512, 512, 2026),
        ("B_正脸", PROMPT_B, 512, 512, 2026),
    ]
    for name, prompt, w, h, seed in jobs:
        img = pipe(
            prompt=prompt, negative_prompt=NEGATIVE,
            width=w, height=h, num_inference_steps=30, guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        img.save(OUT_DIR / f"{name}.png")
        print(f"已生成 {name}（{time.time()-t0:.1f}s）", flush=True)
    print(f"完成: {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
