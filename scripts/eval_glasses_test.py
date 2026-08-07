"""v3 眼镜特征测试：prompt 不含 glasses 时，LoRA 是否自动生成眼镜。

关键判断：若 LoRA 学到"眼镜=小R签名"，则无 glasses prompt 也会倾向生成眼镜；
base（无 LoRA）则不会。用 opencv 确定性检测。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionPipeline

BASE_MODEL = r"D:\RoleMatrix\models\base\RealisticVision_V5.1"
LORA_PATH = r"D:\RoleMatrix\models\lora_xiaor_real_v3"
OUT_DIR = Path(r"D:\RoleMatrix\.tmp\lora_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)
HC = Path(r"D:\RoleMatrix\.tmp\haarcascades")

# prompt 不含 glasses —— 测试 LoRA 是否自带眼镜特征
PROMPT_NO_GLASSES = (
    "little_r, young east asian woman, slightly chubby, soft round face, baby face, "
    "long black layered hair, wispy bangs, fair skin, portrait, looking at camera, "
    "indoor, warm lighting, realistic photo"
)
NEGATIVE = (
    "old woman, child, loli, muscular, skinny, athletic, heavy makeup, "
    "blonde hair, colored hair, short hair, male, anime, cartoon, illustration, "
    "worst quality, low quality, bad anatomy, deformed, blurry"
)


def detect_glasses(path: Path) -> int:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(str(HC / "haarcascade_eye_tree_eyeglasses.xml"))
    return len(cascade.detectMultiScale(gray, 1.1, 5, minSize=(15, 15)))


def main() -> int:
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, safety_checker=None,
        local_files_only=True,
    ).to("cuda")
    gen = torch.Generator("cuda").manual_seed(42)

    # base（无 LoRA）—— 对照组
    img = pipe(
        prompt=PROMPT_NO_GLASSES, negative_prompt=NEGATIVE,
        width=512, height=512, num_inference_steps=25, guidance_scale=7.0,
        generator=gen,
    ).images[0]
    img.save(OUT_DIR / "noglasses_base.png")
    print(f"无LoRA: 眼镜检测={detect_glasses(OUT_DIR / 'noglasses_base.png')}", flush=True)

    # lora —— 实验组：peft 手动加载（diffusers load_lora_weights 不认 peft 的 base_model.model 前缀）
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
        prompt=PROMPT_NO_GLASSES, negative_prompt=NEGATIVE,
        width=512, height=512, num_inference_steps=25, guidance_scale=7.0,
        generator=gen,
    ).images[0]
    img2.save(OUT_DIR / "noglasses_lora.png")
    print(f"有LoRA: 眼镜检测={detect_glasses(OUT_DIR / 'noglasses_lora.png')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
