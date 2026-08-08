"""方案 B 完善验证：768px + 关键特征加权 是否解决面部/眼镜扭曲。

对比（同 seed 2026，长 prompt）：
- 512 原始
- 768
- 768 + 关键特征重复加权（眼镜/眼睛/小痣）
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
from PIL import Image

BASE_MODEL = r"D:\RoleMatrix\models\base\RealisticVision_V5.1"
LORA_PATH = r"D:\RoleMatrix\models\lora_xiaor_real_v7"
OUT_DIR = Path(r"D:\RoleMatrix\.tmp\poc_b_fix")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LONG_PROMPT = (
    "little_r, young adult East Asian woman, 20 years old, soft rounded oval face, "
    "slightly full cheeks, soft jawline, long messy black hair, wispy bangs, "
    "tousled hair, face-framing strands, thick black rectangular glasses with metal rivets, "
    "slightly droopy sleepy eyes, beauty mark under left eye, pale skin, natural skin texture, "
    "neutral aloof expression, oversized cream white hoodie, sleeves over hands, "
    "loose long sleeves, black pleated mini skirt, white mid-calf socks, "
    "white chunky sneakers, black shoulder bag, university library, window seat, "
    "golden hour sunlight, reading a book, holding book, bookshelf background, "
    "warm indoor lighting, quiet atmosphere, casual candid photo, realistic"
)
# 关键特征加权：重复出现 = attention 权重加倍
KEY_FEATURES = [
    "thick black rectangular glasses with metal rivets",
    "slightly droopy sleepy eyes",
    "beauty mark under left eye",
]
WEIGHTED_PROMPT = LONG_PROMPT + ", " + ", ".join(KEY_FEATURES * 2)

NEGATIVE = (
    "old woman, child, loli, muscular, skinny, heavy makeup, blonde hair, "
    "colored hair, short hair, male, caucasian, western face, sharp jawline, "
    "anime, cartoon, worst quality, low quality, bad anatomy, deformed, blurry"
)


def encode_long(tokenizer, text_encoder, prompt, max_len=77, chunk=75):
    ids = tokenizer.encode(prompt)
    if ids and ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
    if ids and ids[-1] == tokenizer.eos_token_id:
        ids = ids[:-1]
    embeds = []
    for i in range(0, len(ids), chunk):
        seg = [tokenizer.bos_token_id] + ids[i : i + chunk] + [tokenizer.eos_token_id]
        if len(seg) < max_len:
            seg = seg + [tokenizer.eos_token_id] * (max_len - len(seg))
        seg_t = torch.tensor([seg], device=text_encoder.device)
        embeds.append(text_encoder(seg_t)[0])
    return torch.cat(embeds, dim=1)


def encode_short(tokenizer, text_encoder, prompt):
    return text_encoder(
        tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to("cuda")
    )[0]


def sample(pipe, prompt_embeds, neg_embeds, size=512, steps=30, seed=0):
    scheduler = pipe.scheduler
    torch.manual_seed(seed)
    latents = torch.randn((1, 4, size // 8, size // 8), device="cuda", dtype=prompt_embeds.dtype)
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(steps)
    for t in scheduler.timesteps:
        t = t.to(latents.device)
        with torch.no_grad():
            nu = pipe.unet(latents, t, encoder_hidden_states=neg_embeds).sample
            nt = pipe.unet(latents, t, encoder_hidden_states=prompt_embeds).sample
        latents = scheduler.step(nu + 7.5 * (nt - nu), t, latents).prev_sample
    with torch.no_grad():
        latents = latents / pipe.vae.config.scaling_factor
        img = pipe.vae.decode(latents.to(pipe.vae.dtype)).sample
    img = (img / 2 + 0.5).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    return Image.fromarray((img * 255).clip(0, 255).astype("uint8"))


def main() -> int:
    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, safety_checker=None, local_files_only=True,
    ).to("cuda")
    cfg = LoraConfig(r=64, lora_alpha=128, target_modules=["to_q", "to_k", "to_v", "to_out.0"], lora_dropout=0.0)
    pipe.unet = get_peft_model(pipe.unet, cfg)
    sd = load_file(Path(LORA_PATH) / "adapter_model.safetensors")
    sd = {k.replace("base_model.model.", ""): v for k, v in sd.items()}
    set_peft_model_state_dict(pipe.unet, sd)
    pipe.unet.eval()

    neg = encode_short(pipe.tokenizer, pipe.text_encoder, NEGATIVE)
    pos_long = encode_long(pipe.tokenizer, pipe.text_encoder, LONG_PROMPT)
    pos_w = encode_long(pipe.tokenizer, pipe.text_encoder, WEIGHTED_PROMPT)

    jobs = [
        ("512_long", pos_long, 512, 2026),
        ("768_long", pos_long, 768, 2026),
        ("768_weighted", pos_w, 768, 2026),
        ("768_weighted_s777", pos_w, 768, 777),
    ]
    for name, pos, size, seed in jobs:
        img = sample(pipe, pos, neg, size=size, seed=seed)
        img.save(OUT_DIR / f"{name}.png")
        print(f"{name} 完成（{time.time()-t0:.1f}s）", flush=True)
    print(f"完成: {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
