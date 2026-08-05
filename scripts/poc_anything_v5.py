"""Anything-v5 POC：验证 8GB 显存（RTX 4060 Laptop）本地生图可行性。

流程：下载模型（hf-mirror）→ 加载 → 生成一张小R 形象测试图 → 报告耗时/显存。
产物：.tmp/poc_anything_v5.png + 控制台指标。
"""
from __future__ import annotations

import time

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

OUT = r"D:\RoleMatrix\.tmp\poc_anything_v5.png"

# 小R 形象（对应 personas/default.yaml）：戴圆框眼镜、圆脸、低马尾、大号卫衣、害羞宅女
PROMPT = (
    "anime style, selfie photo of a shy cute girl with round glasses, "
    "soft round face, blushing, low ponytail, messy soft hair, "
    "wearing an oversized hoodie with long sleeves covering hands, "
    "cozy bedroom background, natural lighting, casual life photo, "
    "detailed face, high quality, 4k"
)
NEGATIVE = (
    "worst quality, low quality, bad anatomy, bad hands, text, "
    "watermark, signature, blurry, deformed"
)

if __name__ == "__main__":
    t0 = time.time()
    print("加载 Anything-v5 (genai-archive/anything-v5) ...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        "genai-archive/anything-v5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    print(f"模型加载完成，耗时 {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        width=512,
        height=512,
        num_inference_steps=20,
        guidance_scale=7.5,
    ).images[0]
    gen_time = time.time() - t1
    image.save(OUT)
    vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"出图完成：生成耗时 {gen_time:.1f}s，总耗时 {time.time() - t0:.1f}s", flush=True)
    print(f"峰值显存：{vram:.2f} GB / 8 GB", flush=True)
    print(f"已保存：{OUT}", flush=True)
