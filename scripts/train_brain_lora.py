"""大脑 LoRA 微调脚本（instruction 格式）

训练目标：让本地 Qwen2.5-7B-LoRA 学会输出结构化 JSON 决策
输入格式：messages = [system, user(用户消息+情绪+历史), assistant(JSON 决策)]
输出：LoRA adapter 保存到 models/lora_brain_v1

数据：data/datasets/cleaned/rolematrix_brain_train.jsonl (DeepSeek 标注)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# === 路径硬约束（详见 rule/HARDWARE_AND_DEV_RULES.md 2.3 节）===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = r"D:\RoleMatrix\.hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\RoleMatrix\.hf_cache\hub"
os.environ["HF_DATASETS_CACHE"] = r"D:\RoleMatrix\.hf_cache\datasets"
os.environ["TRANSFORMERS_CACHE"] = r"D:\RoleMatrix\.hf_cache"
os.environ["OLLAMA_MODELS"] = r"D:\RoleMatrix\.ollama\models"
os.environ["PIP_CACHE_DIR"] = r"D:\RoleMatrix\.pip_cache"
os.environ["HF_ENDPOINT"] = r"https://hf-mirror.com"

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
BASE_MODEL = r"D:\RoleMatrix\models\base\Qwen2.5-7B-Instruct-bnb-4bit"
DATA_PATH = Path(r"D:\RoleMatrix\data\datasets\cleaned\rolematrix_brain_train.jsonl")
OUTPUT_DIR = Path(r"D:\RoleMatrix\models\lora_brain_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 768  # instruction 格式 system+user+assistant JSON 较长

# LoRA 参数（与第一轮相同，保证可叠加）
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# 训练参数
EPOCHS = 3  # 结构化输出比风格学习更难，多训几轮
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 2e-4


# ============================================================
# 数据加载
# ============================================================
def load_dataset(tokenizer) -> Dataset:
    """加载 instruction 格式数据（messages 列表）"""
    records = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            messages = r.get("messages", [])
            if len(messages) < 2:
                continue

            # 应用 chat template 转成文本
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            records.append({"text": text})

    print(f"加载 {len(records)} 条 instruction 样本")
    if records:
        print(f"\n样本示例（前300字）:\n{records[0]['text'][:300]}...")
    return Dataset.from_list(records)


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("RoleMatrix 大脑 LoRA 微调 (instruction 格式, 结构化 JSON 输出)")
    print("=" * 60)
    print(f"硬件: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # 1. tokenizer
    print("\n[1/5] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 数据
    print("\n[2/5] 加载数据集...")
    dataset = load_dataset(tokenizer)

    # 3. 模型
    print("\n[3/5] 加载模型 (4bit 预量化)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 4. LoRA
    print("\n[4/5] 配置 LoRA...")
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. 训练
    print("\n[5/5] 启动训练...")
    print(f"  epochs: {EPOCHS}, batch: {BATCH_SIZE}, grad_accum: {GRAD_ACCUM}")
    est_steps = len(dataset) * EPOCHS // (BATCH_SIZE * GRAD_ACCUM)
    print(f"  预计 steps: ~{est_steps}")

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
        model=model, args=sft_config,
        train_dataset=dataset, processing_class=tokenizer,
    )

    print(f"\n训练前 VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    trainer.train()

    print("\n保存 LoRA adapter...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\n训练完成！LoRA adapter 已保存到: {OUTPUT_DIR}")
    print(f"最终 VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
