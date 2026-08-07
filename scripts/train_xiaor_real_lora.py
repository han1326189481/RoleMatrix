"""小R 真人形象 LoRA 训练（SD1.5 / Realistic Vision v5.1，8GB 显存适配）。

流程：加载 Realistic Vision v5.1（fp16-no-ema）→ peft 给 UNet 注入 LoRA →
      DDPM 噪声预测损失训练 → 保存 adapter 到 models/lora_xiaor_real_v1/

设计要点（对应 rule/HARDWARE_AND_DEV_RULES.md）：
- 8GB 显存：batch_size=1 + gradient_accumulation=4 + fp16 混合精度
- VAE 用 fp32（训练只 encode，避免 fp16 数值误差）
- 512x512 center-crop，不做水平翻转（真人脸不对称）
- 触发词：xiaor（caption 统一 "photo of xiaor girl"）
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import DDPMScheduler, StableDiffusionPipeline
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from rolematrix.logger import get_logger, setup_logging

log = get_logger("train.xiaor_real")

BASE_MODEL = r"D:\RoleMatrix\models\base\RealisticVision_V5.1"
IMAGE_DIR = Path(r"D:\RoleMatrix\data\lora_xiaor_real_v5\img")
OUTPUT_DIR = Path(r"D:\RoleMatrix\models\lora_xiaor_real_v5")
RESOLUTION = 512
LORA_RANK = 64
LORA_ALPHA = 128
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
GRAD_ACCUM = 4
TOTAL_STEPS = 2200
WARMUP_STEPS = 150
REPEAT_PER_IMAGE = 30  # 24 张 × 30 = 720/epoch，约 3 epoch = 2160 steps（上限 2200）
SEED = 42


class CaptionDataset(Dataset):
    """读 img/*.png + 同名 .txt caption（无 txt 用默认触发词）。"""

    def __init__(self, image_dir: Path, repeat: int = 1) -> None:
        self.items: list[tuple[str, str]] = []
        for img in sorted(image_dir.glob("*.png")):
            cap_file = img.with_suffix(".txt")
            caption = (
                cap_file.read_text(encoding="utf-8").strip()
                if cap_file.exists()
                else "little_r, young east asian woman, slightly chubby, soft round face, black rectangular glasses"
            )
            self.items.append((str(img), caption))
        self.items = self.items * repeat
        random.Random(SEED).shuffle(self.items)
        self.transform = transforms.Compose([
            transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # SD 像素归一化 [-1,1]
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, caption = self.items[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), caption


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    setup_logging()

    log.info("加载基模 %s ...", BASE_MODEL)
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=True,
    )

    # VAE 用 fp32（训练 encode 更稳）；unet/text_encoder fp16
    pipe.vae.to("cuda", dtype=torch.float32)
    pipe.vae.eval()
    pipe.text_encoder.to("cuda", dtype=torch.float16)
    pipe.text_encoder.eval()
    unet = pipe.unet.to("cuda", dtype=torch.float16)
    unet.train()

    # peft LoRA 注入 UNet attention
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    tokenizer = pipe.tokenizer
    vae = pipe.vae

    dataset = CaptionDataset(IMAGE_DIR, repeat=REPEAT_PER_IMAGE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    log.info("数据集: %d 张图（含重复）", len(dataset))

    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
    # 余弦退火 + 线性预热
    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, args.steps - WARMUP_STEPS)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))).item())

    scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    total_steps = args.steps
    step = 0
    t0 = time.time()
    running_loss = 0.0
    scaler = torch.amp.GradScaler("cuda")

    log.info("开始训练 %d steps ...", total_steps)
    while step < total_steps:
        for pixel_values, captions in loader:
            if step >= total_steps:
                break
            pixel_values = pixel_values.to("cuda", dtype=torch.float32)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=torch.float16)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (latents.shape[0],),
                device="cuda",
            ).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            text_inputs = tokenizer(
                captions, padding="max_length", max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            encoder_hidden_states = pipe.text_encoder(
                text_inputs.input_ids.to("cuda")
            )[0]

            with torch.autocast("cuda", dtype=torch.float16):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            running_loss += loss.item()

            if (step + 1) % GRAD_ACCUM == 0 or step + 1 == total_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler_lr.step()

            step += 1
            if step % 50 == 0 or step == total_steps:
                vram = torch.cuda.memory_allocated() / 1024**3
                log.info(
                    "step %d/%d  loss=%.4f  lr=%.2e  VRAM=%.2fGB  耗时=%.0fs",
                    step, total_steps, running_loss / 50, scheduler_lr.get_last_lr()[0],
                    vram, time.time() - t0,
                )
                running_loss = 0.0

    # 保存 adapter（peft 格式）+ 训练记录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(OUTPUT_DIR))
    with open(OUTPUT_DIR / "training_args.json", "w", encoding="utf-8") as f:
        import json
        json.dump({
            "base_model": BASE_MODEL,
            "resolution": RESOLUTION,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "steps": step,
            "num_images": len(dataset) // REPEAT_PER_IMAGE,
            "trigger_word": "xiaor",
            "duration_sec": round(time.time() - t0, 1),
        }, f, ensure_ascii=False, indent=2)
    log.info("训练完成，adapter 已保存: %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
