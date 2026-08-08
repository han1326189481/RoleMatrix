"""POC-A：LLM 提炼长描述 → ≤77 token SD 提示词 → 生图。

验证"识图/语言模型与生图模型配合"：qwen2.5:7b 把长中文描述
提炼成 SD 英文 tag（含小R 身份特征模板），确保全部关键特征
都在 CLIP 77 token 内。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx
import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

OLLAMA = "http://127.0.0.1:11434/api/generate"
BASE_MODEL = r"D:\RoleMatrix\models\base\RealisticVision_V5.1"
LORA_PATH = r"D:\RoleMatrix\models\lora_xiaor_real_v7"
OUT_DIR = Path(r"D:\RoleMatrix\.tmp\poc_a")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 身份特征模板（LLM 必须保留，防止提炼时丢掉小R 身份）
IDENTITY = (
    "little_r, young adult East Asian woman, soft rounded face, full cheeks, "
    "long messy black hair, wispy bangs, thick black rectangular glasses, "
    "droopy gentle eyes, beauty mark under left eye, pale skin"
)

SYSTEM = (
    "你是 Stable Diffusion 提示词工程师。把用户的中文描述提炼成英文逗号分隔的 tag，"
    "规则：1) 必须保留我给你的身份模板（原样放在最前面）2) 场景/动作/光影用简短 tag 表达"
    "3) 总 token 数尽量不超过 70 4) 只输出 tag 本身，不要解释。"
    f"身份模板：{IDENTITY}"
)

NEGATIVE = (
    "old woman, child, loli, muscular, skinny, heavy makeup, blonde hair, "
    "colored hair, short hair, male, caucasian, western face, sharp jawline, "
    "anime, cartoon, worst quality, low quality, bad anatomy, deformed, blurry"
)


def distill(description: str) -> str:
    with httpx.Client(timeout=180) as c:
        r = c.post(
            OLLAMA,
            json={
                "model": "qwen2.5:7b",
                "prompt": f"{SYSTEM}\n\n用户描述：{description}",
                "stream": False,
                "options": {"temperature": 0.2},
            },
        )
    return r.json().get("response", "").strip()


def main() -> int:
    # 一段"长描述"（模拟用户大段提示词）
    desc = (
        "小R坐在大学图书馆靠窗的位置，黄昏的阳光透过窗户洒进来，"
        "她低头看书，神情平静慵懒，手缩在米白色宽松卫衣的袖子里，"
        "桌上放着一杯热奶茶和笔记本，背景是书架，氛围安静温暖。"
    )
    print(f"输入描述({len(desc)}字): {desc}\n", flush=True)

    tags = distill(desc)
    print(f"LLM 提炼结果: {tags}\n", flush=True)

    # 组装 prompt
    prompt = tags
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

    tokens = pipe.tokenizer(prompt, truncation=False).input_ids
    print(f"最终 prompt token 数: {len(tokens)}", flush=True)
    img = pipe(
        prompt=prompt, negative_prompt=NEGATIVE,
        width=512, height=512, num_inference_steps=30, guidance_scale=7.5,
        generator=torch.Generator("cuda").manual_seed(2026),
    ).images[0]
    out = OUT_DIR / "poc_a_result.png"
    img.save(out)
    print(f"已生成: {out}（{time.time()-start_time():.1f}s）", flush=True)
    return 0


def start_time() -> float:
    return _t0


_t0 = time.time()

if __name__ == "__main__":
    sys.exit(main())
