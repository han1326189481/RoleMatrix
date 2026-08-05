"""RoleMatrix 小R LoRA 微调脚本 (transformers + bitsandbytes)

硬件: RTX 4060 Laptop 8GB VRAM / 16GB RAM
策略: 加载 unsloth 4bit 预量化权重 (避免下载 15GB 原权重)
基模: unsloth/Qwen2.5-7B-Instruct-bnb-4bit (~5GB)
数据: data/datasets/cleaned/rolematrix_sft_final.jsonl (1765 条)

错误记录区（训练过程中发现的问题）:
- #1: unsloth 安装时把 torch 从 2.5.1+cu121 升级到 2.11.0+cpu，CUDA 不可用 → 重装 torch 2.6.0+cu124
- #2: torchvision 0.26.0 要求 torch==2.11.0，与 torch 2.6.0 冲突 → 卸载 torchvision
- #3: unsloth_zoo 依赖 triton，triton-windows 在 Windows 上导入失败 → 放弃 unsloth，用纯 transformers
- #4: torchao 0.17.0 (unsloth 残留) 与 torch 2.6.0 不兼容 → 卸载 torchao
- #5: 启动训练时默认 python 是 3.9.0（无 torch），应该用 .venv\Scripts\python.exe（3.11.9 + torch 2.6.0+cu124）
- #6: 模型下载卡死——HF_HUB_ENABLE_HF_TRANSFER 已废弃 + 代理上 HF Hub 限流（无 HF_TOKEN）+ 16 条 TCP 连接 ESTABLISHED 但 0 MB/s 增量，17 分钟无更新
       → 修复：移除 HF_HUB_ENABLE_HF_TRANSFER，设置 HF_ENDPOINT=https://hf-mirror.com（国内镜像），先用 huggingface-cli 单独下载模型再启动训练
- 最终方案: torch 2.6.0+cu124 + transformers 5.5.0 + peft 0.20.0 + trl 1.9.2 + bnb 0.50.0
- #7: 训练速度骤降——前 100 步 3.6s/step，100 步后骤降到 47s/step。根因：Windows Memory Compression
       启动后抢占 CPU，导致 DataLoader（num_workers=0）跟不上，GPU 97% 利用率是假象（在等数据）
       → 修复：max_seq_length 1024→512（数据集 P99=340，512 覆盖 99.5%），epochs 3→2，关闭 Memory Compression
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# === 路径硬约束：所有缓存写死到 D 盘，禁止 C 盘（详见 rule/HARDWARE_AND_DEV_RULES.md 2.3 节）===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = r"D:\RoleMatrix\.hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\RoleMatrix\.hf_cache\hub"
os.environ["HF_DATASETS_CACHE"] = r"D:\RoleMatrix\.hf_cache\datasets"
os.environ["TRANSFORMERS_CACHE"] = r"D:\RoleMatrix\.hf_cache"
os.environ["OLLAMA_MODELS"] = r"D:\RoleMatrix\.ollama\models"
os.environ["PIP_CACHE_DIR"] = r"D:\RoleMatrix\.pip_cache"
os.environ["HF_ENDPOINT"] = r"https://hf-mirror.com"  # 国内镜像

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ============================================================
# 配置
# ============================================================
# 本地 4bit 预量化权重（已手动下载到 D 盘，避免联网下载卡死）
BASE_MODEL = r"D:\RoleMatrix\models\base\Qwen2.5-7B-Instruct-bnb-4bit"
DATA_PATH = Path(r"D:\RoleMatrix\data\datasets\cleaned\rolematrix_sft_final.jsonl")
OUTPUT_DIR = Path(r"D:\RoleMatrix\models\lora_xiaor_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 512  # 数据集 P99=340，512 覆盖 99.5% 样本，比 1024 快 4 倍

# LoRA 参数
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# 训练参数
EPOCHS = 2  # 已有 checkpoint-300 fallback（≈1 epoch），2 epoch 足够
BATCH_SIZE = 1
GRAD_ACCUM = 4  # 等效 batch=4
LR = 2e-4


# ============================================================
# 数据加载
# ============================================================
def load_dataset(tokenizer) -> Dataset:
    """加载 ShareGPT 格式数据，用 chat template 转成文本"""
    records = []
    for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        convs = r.get("conversations", [])
        if len(convs) < 2:
            continue
        messages = []
        for c in convs:
            role_map = {"system": "system", "human": "user", "gpt": "assistant"}
            role = role_map.get(c["from"], "user")
            messages.append({"role": role, "content": c["value"]})

        # 应用 chat template 转成文本
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        records.append({"text": text})

    print(f"加载 {len(records)} 条对话")
    print(f"\n样本示例（前200字）:\n{records[0]['text'][:200]}...")
    return Dataset.from_list(records)


# ============================================================
# 主训练流程
# ============================================================
def main() -> None:
    print("=" * 60)
    print("RoleMatrix 小R LoRA 微调 (transformers + bnb 4bit)")
    print("=" * 60)
    print(f"硬件: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # 1. 加载 tokenizer
    print("\n[1/5] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载数据
    print("\n[2/5] 加载数据集...")
    dataset = load_dataset(tokenizer)

    # 3. 加载 4bit 预量化模型
    print("\n[3/5] 加载模型 (4bit 预量化)...")
    print(f"  基模: {BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    print(f"  模型加载完成, VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 4. 准备模型 + LoRA
    print("\n[4/5] 配置 LoRA...")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. 训练
    print("\n[5/5] 启动训练...")
    print(f"  epochs: {EPOCHS}, batch: {BATCH_SIZE}, grad_accum: {GRAD_ACCUM}")
    print(f"  等效 batch_size: {BATCH_SIZE * GRAD_ACCUM}")
    est_steps = len(dataset) * EPOCHS // (BATCH_SIZE * GRAD_ACCUM)
    print(f"  预计 steps: ~{est_steps}")
    print("=" * 60)

    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        seed=42,
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\n训练前 VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 启动训练
    trainer.train()

    # 保存
    print("\n保存 LoRA adapter...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\n训练完成！LoRA adapter 已保存到: {OUTPUT_DIR}")
    print(f"最终 VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
