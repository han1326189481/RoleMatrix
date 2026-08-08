"""POC-B：多段 CLIP embedding 拼接，突破 77 token 限制。

原理（sd-webui/ComfyUI 的做法）：prompt 每 75 token 切块，各过 CLIP
得到 [1,77,768]，拼接为 [1, N*77, 768] 喂 UNet cross-attention
（key/value 长度可变，SD1.5 支持任意长度文本条件）。

对比：
- 截断版：同一超长 prompt 走默认 77 截断（场景词丢失）
- 长文本版：分块拼接，全部 token 生效
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
OUT_DIR = Path(r"D:\RoleMatrix\.tmp\poc_b")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 超长 prompt（~200 token）：身份特征 + 完整场景 + 光影 + 服装
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
NEGATIVE = (
    "old woman, child, loli, muscular, skinny, heavy makeup, blonde hair, "
    "colored hair, short hair, male, caucasian, western face, sharp jawline, "
    "anime, cartoon, worst quality, low quality, bad anatomy, deformed, blurry"
)


def encode_long(
    tokenizer, text_encoder, prompt: str, max_len: int = 77, chunk: int = 75
) -> torch.Tensor:
    """把长 prompt 分块过 CLIP，拼接 embedding。"""
    ids = tokenizer.encode(prompt)
    # 去掉首尾 special token
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
        embeds.append(text_encoder(seg_t)[0])  # [1, 77, 768]
    return torch.cat(embeds, dim=1)  # [1, N*77, 768]


def sample(pipe, prompt_embeds, neg_embeds, steps: int, seed: int):
    """手动采样（DPMSolver），支持任意长度 text embedding。"""
    scheduler = pipe.scheduler
    torch.manual_seed(seed)
    latents = torch.randn(
        (1, 4, 512 // 8, 512 // 8), device="cuda", dtype=prompt_embeds.dtype
    )
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(steps)
    for t in scheduler.timesteps:
        t = t.to(latents.device)
        with torch.no_grad():
            noise_uncond = pipe.unet(
                latents, t, encoder_hidden_states=neg_embeds
            ).sample
            noise_text = pipe.unet(
                latents, t, encoder_hidden_states=prompt_embeds
            ).sample
        noise_pred = noise_uncond + 7.5 * (noise_text - noise_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    with torch.no_grad():
        latents = latents / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents.to(pipe.vae.dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


def main() -> int:
    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, safety_checker=None,
        local_files_only=True,
    ).to("cuda")
    cfg = LoraConfig(r=64, lora_alpha=128, target_modules=["to_q", "to_k", "to_v", "to_out.0"], lora_dropout=0.0)
    pipe.unet = get_peft_model(pipe.unet, cfg)
    sd = load_file(Path(LORA_PATH) / "adapter_model.safetensors")
    sd = {k.replace("base_model.model.", ""): v for k, v in sd.items()}
    set_peft_model_state_dict(pipe.unet, sd)
    pipe.unet.eval()
    print(f"模型加载 {time.time()-t0:.1f}s", flush=True)

    # token 统计
    all_tokens = pipe.tokenizer.encode(LONG_PROMPT)
    print(f"超长 prompt 总 token: {len(all_tokens)}", flush=True)

    # 长文本版：分块拼接
    pos_long = encode_long(pipe.tokenizer, pipe.text_encoder, LONG_PROMPT)
    neg = pipe.text_encoder(
        pipe.tokenizer(NEGATIVE, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to("cuda")
    )[0]
    print(f"长文本 embedding: {pos_long.shape}（{pos_long.shape[1]//77} 块拼接）", flush=True)
    img = sample(pipe, pos_long, neg, steps=30, seed=2026)
    from PIL import Image
    img = img.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    img = (img * 255).clip(0, 255).astype("uint8")
    Image.fromarray(img).save(OUT_DIR / "long_full.png")
    print(f"长文本版已生成 {time.time()-t0:.1f}s", flush=True)

    # 截断版对照：同 prompt 只取前 77 token（默认行为）
    pos_trunc = pipe.text_encoder(
        pipe.tokenizer(LONG_PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to("cuda")
    )[0]
    img2 = sample(pipe, pos_trunc, neg, steps=30, seed=2026)
    img2 = img2.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    img2 = (img2 * 255).clip(0, 255).astype("uint8")
    Image.fromarray(img2).save(OUT_DIR / "truncated_77.png")
    print(f"截断版已生成 {time.time()-t0:.1f}s", flush=True)
    print(f"完成: {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
