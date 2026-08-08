"""RoleMatrix 小R 形象 LoRA 训练（SD1.5 真人底模）

硬件: RTX 4060 Laptop 8GB VRAM / 16GB RAM
底模: stablediffusionapi/realistic-vision-v51 (SD1.5 真人微调, ~2GB)
数据: data/image_train/ (28 张 512x512 + caption .txt)
输出: models/lora_xiaor_image_v1/ (LoRA 权重, ~50MB)

训练逻辑（diffusers + peft + bitsandbytes）:
1. 加载底模组件（UNet/VAE/text_encoder/tokenizer/scheduler）
2. 冻结 VAE 和 text_encoder，UNet 加 LoRA adapter（只训 attention 层）
3. 训练循环: 图片→VAE latent→加噪→UNet 预测 noise→MSE loss→backward
4. 保存 LoRA 权重（save_attn_procs → pytorch_lora_weights.safetensors）

显存优化（8GB VRAM）:
- bf16 训练（RTX 4060 原生支持）
- gradient_checkpointing（省激活值显存）
- 8bit AdamW（bitsandbytes，省优化器显存）
- VAE/text_encoder 用 torch.no_grad()（不计算梯度）
- 只训练 UNet attention 层（to_q/k/v/out.0）

训练参数:
- LoRA rank=32, alpha=16（人物 LoRA 推荐）
- lr=1e-4（人物 LoRA 常用）
- batch=1, grad_accum=4（等效 batch=4）
- epochs=30（28×30=840 forward steps，接近经验值）
- resolution=512（SD1.5 原生分辨率）

前置条件:
1. 底模已下载（hf download stablediffusionapi/realistic-vision-v51）
2. 数据已准备（python scripts/prepare_image_train_data.py）
3. 训练时关掉 Ollama 同模型（避免显存冲突）

训练后推理:
    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
    pipe.load_lora_weights(OUTPUT_DIR)  # 加载小R LoRA
    image = pipe("xiaor girl, selfie photo, smiling").images[0]
"""
from __future__ import annotations

import os
from pathlib import Path

# === 路径硬约束：所有缓存写死到 D 盘，禁止 C 盘（rule/HARDWARE_AND_DEV_RULES.md 2.3 节）===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = r"D:\RoleMatrix\.hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\RoleMatrix\.hf_cache\hub"
os.environ["HF_DATASETS_CACHE"] = r"D:\RoleMatrix\.hf_cache\datasets"
os.environ["TRANSFORMERS_CACHE"] = r"D:\RoleMatrix\.hf_cache"
os.environ["HF_ENDPOINT"] = r"https://hf-mirror.com"  # 国内镜像

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import bitsandbytes as bnb
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer

# ============================================================
# 配置
# ============================================================
# 本地路径（已通过 .tmp/download_base_model.py 下载到 D 盘，避免联网）
BASE_MODEL = r"D:\RoleMatrix\models\base\realistic-vision-v51"
DATA_DIR = Path(r"D:\RoleMatrix\data\image_train")
OUTPUT_DIR = Path(r"D:\RoleMatrix\models\lora_xiaor_image_v2")
RESOLUTION = 512

# LoRA 参数（人物 LoRA 推荐 rank=64，增加容量学习稳定人物特征）
LORA_R = 64
LORA_ALPHA = 32  # alpha = rank/2，平衡学习率
LORA_DROPOUT = 0.05
# SD1.5 UNet CrossAttention 层（人物 LoRA 只训 attention，不训 conv）
TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]

# 训练参数（v2 调整：epochs 30→15 避免过拟合，v1 的 loss 0.0027 过低）
EPOCHS = 15  # 28×15=420 forward steps
BATCH_SIZE = 1
GRAD_ACCUM = 4  # 等效 batch=4
LR = 1e-4  # 人物 LoRA 常用学习率
SAVE_STEPS = 50  # 每 50 步存一个 checkpoint（便于对比不同训练阶段）
LOG_STEPS = 10
MAX_GRAD_NORM = 1.0


# ============================================================
# 数据集
# ============================================================
class ImageTextDataset(Dataset):
    """加载图片 + 同名 .txt caption。"""

    def __init__(self, data_dir: Path, tokenizer: CLIPTokenizer, resolution: int):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.image_paths = sorted(
            list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
        )
        if not self.image_paths:
            raise RuntimeError(f"数据目录无图片: {data_dir}")
        print(f"  数据集: {len(self.image_paths)} 张图 ({data_dir})")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        txt_path = img_path.with_suffix(".txt")
        caption = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""

        img = Image.open(img_path).convert("RGB")
        # 保险起见再 resize 一次（数据准备脚本已 crop 到 512x512）
        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution), Image.LANCZOS)

        # to tensor, normalize to [-1, 1]（SD1.5 标准）
        arr = np.array(img).astype("float32") / 255.0
        arr = (arr - 0.5) / 0.5
        pixel_values = torch.from_numpy(arr).permute(2, 0, 1)  # HWC → CHW

        # encode caption → input_ids
        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        return {"pixel_values": pixel_values, "input_ids": input_ids}


# ============================================================
# 主训练流程
# ============================================================
def main() -> None:
    print("=" * 60)
    print("RoleMatrix 小R 形象 LoRA 训练 (SD1.5 真人底模)")
    print("=" * 60)
    print(f"硬件: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"底模: {BASE_MODEL}")
    print(f"数据: {DATA_DIR}")
    print(f"输出: {OUTPUT_DIR}")
    print(f"LoRA: rank={LORA_R}, alpha={LORA_ALPHA}, lr={LR}")
    print(f"训练: epochs={EPOCHS}, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
    print(f"分辨率: {RESOLUTION}x{RESOLUTION}")
    print("=" * 60)

    # 1. 加载底模组件
    print("\n[1/6] 加载底模组件...")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        BASE_MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    vae = AutoencoderKL.from_pretrained(
        BASE_MODEL, subfolder="vae", torch_dtype=torch.bfloat16
    )
    unet = UNet2DConditionModel.from_pretrained(
        BASE_MODEL, subfolder="unet", torch_dtype=torch.bfloat16
    )
    print(f"  加载完成, VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 2. 冻结 + 移到 GPU
    print("\n[2/6] 冻结 VAE/text_encoder, 移到 GPU...")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    vae.to("cuda")
    text_encoder.to("cuda")
    unet.to("cuda")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 3. 添加 LoRA adapter
    print("\n[3/6] UNet 添加 LoRA adapter...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
    )
    unet.add_adapter(lora_config)

    # 显存优化：gradient checkpointing（省激活值显存，代价是训练慢 ~20%）
    unet.enable_gradient_checkpointing()

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"  LoRA 已添加, 可训练参数: {n_trainable/1e6:.2f}M")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 4. 数据集 + 优化器
    print("\n[4/6] 加载数据集 + 优化器...")
    dataset = ImageTextDataset(DATA_DIR, tokenizer, RESOLUTION)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    # 8bit AdamW（省优化器显存，8GB VRAM 必备）
    optimizer = bnb.optim.AdamW8bit(
        trainable_params,
        lr=LR,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    # 5. 训练循环
    print("\n[5/6] 启动训练...")
    est_forward = len(dataset) * EPOCHS
    est_optim = est_forward // GRAD_ACCUM
    print(f"  预计 forward steps: {est_forward}")
    print(f"  预计 optimizer steps: {est_optim}")
    print(f"  checkpoint 保存: 每 {SAVE_STEPS} 步")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unet.train()
    torch.cuda.reset_peak_memory_stats()

    global_step = 0
    optimizer_step = 0
    accum_loss = 0.0

    for epoch in range(EPOCHS):
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to("cuda", dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to("cuda")

            # encode image to latent（VAE 不计算梯度）
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # sample noise + timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device="cuda",
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # encode text（text_encoder 不计算梯度）
                encoder_hidden_states = text_encoder(input_ids)[0]

            # predict noise（UNet 计算梯度）
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # MSE loss（预测 noise vs 真实 noise）
            loss = F.mse_loss(noise_pred, noise)
            loss_scaled = loss / GRAD_ACCUM
            loss_scaled.backward()
            accum_loss += loss.item()

            global_step += 1

            # optimizer step（每 GRAD_ACCUM 步更新一次）
            if global_step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

            # 日志
            if global_step % LOG_STEPS == 0:
                vram = torch.cuda.memory_allocated() / 1024**3
                peak_vram = torch.cuda.max_memory_allocated() / 1024**3
                avg_loss = accum_loss / min(global_step, LOG_STEPS)
                print(
                    f"  epoch {epoch+1}/{EPOCHS} | step {global_step}/{est_forward} "
                    f"| opt {optimizer_step} | loss {loss.item():.4f} "
                    f"| VRAM {vram:.2f}GB (peak {peak_vram:.2f}GB)"
                )
                accum_loss = 0.0

            # 定期保存 checkpoint
            if global_step % SAVE_STEPS == 0:
                ckpt_dir = OUTPUT_DIR / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                unet.save_attn_procs(str(ckpt_dir))
                print(f"  >> 保存 checkpoint: {ckpt_dir}")

    # 6. 保存最终 LoRA
    print("\n[6/6] 保存最终 LoRA...")
    unet.save_attn_procs(str(OUTPUT_DIR))

    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n训练完成！")
    print(f"  LoRA 已保存到: {OUTPUT_DIR}")
    print(f"  最终 VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"  峰值 VRAM: {peak_vram:.2f} GB / 8 GB")
    print(f"  总 forward steps: {global_step}")
    print(f"  总 optimizer steps: {optimizer_step}")
    print("=" * 60)
    print("推理测试命令:")
    print(f'  & "D:\\RoleMatrix\\.venv\\Scripts\\python.exe" scripts/test_xiaor_image_lora.py')
    print("  或直接用:")
    print("  from diffusers import StableDiffusionPipeline")
    print(f'  pipe = StableDiffusionPipeline.from_pretrained("{BASE_MODEL}", torch_dtype=torch.float16)')
    print(f'  pipe.load_lora_weights(r"{OUTPUT_DIR}")')
    print('  pipe.to("cuda")')
    print('  pipe("xiaor girl, selfie photo, smiling").images[0].save("test.png")')


if __name__ == "__main__":
    main()
