"""自检：long_full 面部扭曲的根因定位。

对照组设计（同 LoRA v7）：
- 长 prompt（147 token，方案 B）: seed 42 / 777 / 2026 —— 是否普遍崩
- 短 prompt（平衡版 ~55 token）: seed 42 / 777 / 2026 —— 对照
- 修复分块版（后续块不带重复 bos）: seed 2026 —— 验证中间 bos/eos 干扰
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
OUT_DIR = Path(r"D:\RoleMatrix\.tmp\poc_b_diag")
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
SHORT_PROMPT = (
    "little_r, young adult East Asian woman, soft rounded face, full cheeks, "
    "long messy black hair, wispy bangs, thick black rectangular glasses, "
    "droopy gentle eyes, beauty mark under left eye, pale skin, "
    "realistic photo, natural lighting, casual selfie"
)
NEGATIVE = (
    "old woman, child, loli, muscular, skinny, heavy makeup, blonde hair, "
    "colored hair, short hair, male, caucasian, western face, sharp jawline, "
    "anime, cartoon, worst quality, low quality, bad anatomy, deformed, blurry"
)


def encode_long(tokenizer, text_encoder, prompt, max_len=77, chunk=75, skip_mid_bos=False):
    """分块过 CLIP 拼接 embedding。skip_mid_bos=True 时后续块不带重复 bos。"""
    ids = tokenizer.encode(prompt)
    if ids and ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
    if ids and ids[-1] == tokenizer.eos_token_id:
        ids = ids[:-1]
    embeds = []
    for i, start in enumerate(range(0, len(ids), chunk)):
        seg = ids[start : start + chunk]
        if i == 0:
            seg_ids = [tokenizer.bos_token_id] + seg + [tokenizer.eos_token_id]
        elif skip_mid_bos:
            seg_ids = seg + [tokenizer.eos_token_id]
        else:
            seg_ids = [tokenizer.bos_token_id] + seg + [tokenizer.eos_token_id]
        if len(seg_ids) < max_len:
            seg_ids = seg_ids + [tokenizer.eos_token_id] * (max_len - len(seg_ids))
        seg_t = torch.tensor([seg_ids], device=text_encoder.device)
        embeds.append(text_encoder(seg_t)[0])
    return torch.cat(embeds, dim=1)


def encode_short(tokenizer, text_encoder, prompt):
    return text_encoder(
        tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to("cuda")
    )[0]


def sample(pipe, prompt_embeds, neg_embeds, steps=30, seed=0):
    scheduler = pipe.scheduler
    torch.manual_seed(seed)
    latents = torch.randn((1, 4, 512 // 8, 512 // 8), device="cuda", dtype=prompt_embeds.dtype)
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

    # 1) 长 prompt × 多 seed（验证是否普遍崩）
    for seed in [42, 777, 2026]:
        pos = encode_long(pipe.tokenizer, pipe.text_encoder, LONG_PROMPT)
        img = sample(pipe, pos, neg, seed=seed)
        img.save(OUT_DIR / f"long_seed{seed}.png")
        print(f"long_seed{seed} 完成", flush=True)

    # 2) 短 prompt × 多 seed（对照）
    for seed in [42, 777, 2026]:
        pos = encode_short(pipe.tokenizer, pipe.text_encoder, SHORT_PROMPT)
        img = sample(pipe, pos, neg, seed=seed)
        img.save(OUT_DIR / f"short_seed{seed}.png")
        print(f"short_seed{seed} 完成", flush=True)

    # 3) 修复分块（后续块无重复 bos）seed 2026
    pos = encode_long(pipe.tokenizer, pipe.text_encoder, LONG_PROMPT, skip_mid_bos=True)
    img = sample(pipe, pos, neg, seed=2026)
    img.save(OUT_DIR / "long_nomidbos_seed2026.png")
    print(f"long_nomidbos 完成", flush=True)

    print(f"全部完成 {time.time()-t0:.1f}s -> {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
