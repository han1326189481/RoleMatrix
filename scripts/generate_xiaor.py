"""小R 生图流水线：方案 B 长 prompt + 768px + 关键特征加权 + 多 seed 抽卡。

用法：
  1) 只生成（自己挑）：
     python scripts/generate_xiaor.py --seeds 6
     python scripts/generate_xiaor.py --seeds 3 --out outputs\我的测试 --prompt "自定义完整描述"

  2) 生成 + 自动评审选优（qwen2.5vl 当评委，拷贝最高分到 out\\best\\）：
     python scripts/generate_xiaor.py --seeds 6 --pick

参数：
  --seeds N         抽卡数量（默认 6；或用 --seed-list 1,2,3 指定）
  --out DIR         输出目录（默认 outputs\\xiaor_<时间戳>）
  --size PX         生成分辨率（默认 768，512 脸部细节不足）
  --steps N         采样步数（默认 30）
  --cfg N           guidance scale（默认 7.5）
  --prompt TEXT     自定义 prompt（默认用小R 平衡版模板 + 关键特征加权）
  --no-weight       关闭关键特征加权（身份特征不重复）
  --pick            生成后自动评审选优（调用本地 qwen2.5vl:7b）
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from PIL import Image
import httpx

BASE_MODEL = r"D:\RoleMatrix\models\base\RealisticVision_V5.1"
LORA_PATH = r"D:\RoleMatrix\models\lora_xiaor_real_v7"
DEFAULT_OUT = Path(r"D:\RoleMatrix\outputs")

# ============================================================
# 小R 平衡版 prompt 模板（v7 验证：身份特征前置 + 场景细节）
# 身份特征放在第一块前部，防止被 77 token 截断
# ============================================================
XIAOR_PROMPT = (
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
# 关键身份特征：加权 = 重复出现（对抗长 prompt 注意力稀释）
KEY_FEATURES = [
    "thick black rectangular glasses with metal rivets",
    "slightly droopy sleepy eyes",
    "beauty mark under left eye",
]
NEGATIVE = (
    "old woman, child, loli, muscular, skinny, heavy makeup, blonde hair, "
    "colored hair, short hair, male, caucasian, western face, sharp jawline, "
    "anime, cartoon, worst quality, low quality, bad anatomy, deformed, blurry"
)

# 评审 prompt（与 judge_face.py 一致）
JUDGE_PROMPT = """你是一名严格的 AI 人像质量评审员，专门检查 AI 生成人像的面部缺陷。
仔细放大观察图中人物的脸，逐项严格检查并输出 JSON（不要输出其他文字）：
{
  "face": "正常|轻微扭曲|明显扭曲",
  "eyes": "正常|轻微异常|明显异常",
  "glasses": "无眼镜|清晰完整|轻微变形|明显变形",
  "face_score": 0-100,
  "detail_score": 0-100,
  "overall_score": 0-100,
  "defects": ["缺陷列表，如'右眼镜片变形'、'左眼大小不一'，无则空数组"]
}
注意：眼镜佩戴者的镜片变形、眼睛不对称、五官糊成一团都是扣分项。"""


# ============================================================
# 文本编码：方案 B（分块过 CLIP 拼接，突破 77 token 限制）
# ============================================================
def encode_prompt(tokenizer, text_encoder, prompt: str, max_len: int = 77, chunk: int = 75) -> torch.Tensor:
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


# ============================================================
# 模型加载（LoRA v7 + Realistic Vision）
# ============================================================
def load_pipeline() -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, safety_checker=None, local_files_only=True,
    ).to("cuda")
    cfg = LoraConfig(
        r=64, lora_alpha=128, target_modules=["to_q", "to_k", "to_v", "to_out.0"], lora_dropout=0.0,
    )
    pipe.unet = get_peft_model(pipe.unet, cfg)
    sd = load_file(Path(LORA_PATH) / "adapter_model.safetensors")
    sd = {k.replace("base_model.model.", ""): v for k, v in sd.items()}
    set_peft_model_state_dict(pipe.unet, sd)
    pipe.unet.eval()
    return pipe


# ============================================================
# 采样（支持变长 embedding，uncond/text 分别前向）
# ============================================================
def sample(
    pipe: StableDiffusionPipeline, prompt_embeds: torch.Tensor, neg_embeds: torch.Tensor,
    size: int = 768, steps: int = 30, cfg: float = 7.5, seed: int = 0,
) -> Image.Image:
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
        latents = scheduler.step(nu + cfg * (nt - nu), t, latents).prev_sample
    with torch.no_grad():
        latents = latents / pipe.vae.config.scaling_factor
        img = pipe.vae.decode(latents.to(pipe.vae.dtype)).sample
    img = (img / 2 + 0.5).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    return Image.fromarray((img * 255).clip(0, 255).astype("uint8"))


# ============================================================
# 评审（调用本地 Ollama qwen2.5vl）
# ============================================================
def judge_image(model: str, img_path: Path) -> dict:
    with httpx.Client(timeout=300) as c:
        r = c.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": model,
                "prompt": JUDGE_PROMPT,
                "images": [base64.b64encode(img_path.read_bytes()).decode()],
                "stream": False,
                "format": "json",
            },
        )
    raw = r.json().get("response", "")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw, "overall_score": 0}


def main() -> int:
    ap = argparse.ArgumentParser(description="小R 生图流水线")
    ap.add_argument("--seeds", type=int, default=6, help="抽卡数量（默认 6）")
    ap.add_argument("--seed-list", type=str, default="", help="指定 seed 列表，如 1,2,3（优先于 --seeds）")
    ap.add_argument("--out", type=str, default="", help="输出目录（默认 outputs\\xiaor_<时间戳>）")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--prompt", type=str, default="", help="自定义 prompt（默认小R 模板）")
    ap.add_argument("--no-weight", action="store_true", help="关闭关键特征加权")
    ap.add_argument("--pick", action="store_true", help="生成后自动评审选优")
    ap.add_argument("--judge-model", default="qwen2.5vl:7b")
    args = ap.parse_args()

    # 输出目录
    out_dir = Path(args.out) if args.out else DEFAULT_OUT / f"xiaor_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # prompt 组装
    prompt = args.prompt or XIAOR_PROMPT
    if not args.no_weight:
        prompt = prompt + ", " + ", ".join(KEY_FEATURES * 2)
        print(f"[prompt] 已加权关键特征（眼镜/眼睛/小痣 x2），总 token 约 {len(prompt.split(','))}", flush=True)

    # seeds
    seeds = [int(s) for s in args.seed_list.split(",") if s.strip()] if args.seed_list else list(range(1, args.seeds + 1))

    t0 = time.time()
    print(f"[加载] 基模 + LoRA v7 ...", flush=True)
    pipe = load_pipeline()
    neg = encode_prompt(pipe.tokenizer, pipe.text_encoder, NEGATIVE)
    pos = encode_prompt(pipe.tokenizer, pipe.text_encoder, prompt)
    print(f"[加载完成] {time.time()-t0:.1f}s，prompt {pos.shape[1]} tokens", flush=True)

    paths: list[Path] = []
    for seed in seeds:
        t1 = time.time()
        img = sample(pipe, pos, neg, size=args.size, steps=args.steps, cfg=args.cfg, seed=seed)
        p = out_dir / f"seed{seed}_{args.size}px.png"
        img.save(p)
        paths.append(p)
        print(f"[生成] {p.name} ({time.time()-t1:.1f}s)", flush=True)
    print(f"[完成] {len(paths)} 张 -> {out_dir}", flush=True)

    # 评审选优
    if args.pick:
        print(f"[评审] 评委={args.judge_model} ...", flush=True)
        results: list[tuple[float, Path, dict]] = []
        for p in paths:
            v = judge_image(args.judge_model, p)
            results.append((float(v.get("overall_score", 0) or 0), p, v))
            print(f"  {p.name}: face={v.get('face')} score={v.get('overall_score')} "
                  f"defects={v.get('defects', [])}", flush=True)
        results.sort(reverse=True)
        best_dir = out_dir / "best"
        best_dir.mkdir(exist_ok=True)
        print("\n===== 优劣排序 =====", flush=True)
        for i, (score, p, v) in enumerate(results, 1):
            print(f"{i}. {p.name}  overall={score}  face={v.get('face')}  "
                  f"detail={v.get('detail_score')}  defects={v.get('defects')}", flush=True)
            if i == 1:
                shutil.copy2(p, best_dir / f"BEST_{p.name}")
        print(f"[选优] 最高分图已拷贝 -> {best_dir}\\BEST_*.png", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
