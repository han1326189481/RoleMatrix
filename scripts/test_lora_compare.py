"""小R LoRA 微调效果对比测试

对比同一 base 模型在「无 LoRA」vs「有 LoRA」时的输出差异，
验证微调是否让模型习得小R人设（计算机专业女大学生、高情商、真实口吻）。

原理：PEFT 的 disable_adapter() 上下文管理器可在推理时临时关闭 LoRA，
无需重新加载模型，同 prompt 同 seed 直接对比。
"""
from __future__ import annotations

import os
import sys
import time
import json

# === 路径硬约束（详见 rule/HARDWARE_AND_DEV_RULES.md 2.3 节）===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = r"D:\RoleMatrix\.hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\RoleMatrix\.hf_cache\hub"
os.environ["TRANSFORMERS_CACHE"] = r"D:\RoleMatrix\.hf_cache"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = r"D:\RoleMatrix\models\base\Qwen2.5-7B-Instruct-bnb-4bit"
LORA_PATH = r"D:\RoleMatrix\models\lora_xiaor_v1"

# 小R人设 system prompt（从训练数据里提取的核心人设）
SYSTEM_PROMPT = (
    "你是小R，一个21岁的计算机专业女大学生。性格可爱、高情商，说话自然真实。"
    "你不太懂复杂技术，遇到技术问题会坦诚求助。聊天时用短句、口语化，"
    "不用括号描述动作，长话会分几段说。"
)

# 测试用例（覆盖小R人设的关键场景）
TEST_CASES = [
    {
        "name": "技术求助场景",
        "user": "我的电脑突然蓝屏了怎么办啊",
        "expect": "应体现『不太懂技术但想帮忙/求助』的人设，而非专业解答",
    },
    {
        "name": "情绪安慰场景",
        "user": "今天好累啊，写了一整天代码头都要炸了",
        "expect": "高情商回应，关心+共情，口语化",
    },
    {
        "name": "日常闲聊",
        "user": "你在干嘛呢",
        "expect": "自然短句，口语化，像真人女生聊天",
    },
    {
        "name": "人设一致性",
        "user": "你是谁啊",
        "expect": "应保持小R身份（计算机专业女大学生）",
    },
    {
        "name": "长回复分段测试",
        "user": "给我讲讲你今天在学校发生的事情吧",
        "expect": "应该用短句分段，不是一大段",
    },
]


def generate(model, tokenizer, messages, max_new_tokens=200):
    """生成回复，返回文本和耗时"""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    torch.manual_seed(42)  # 固定 seed 保证可比性
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), elapsed


def main():
    print("=" * 70)
    print("小R LoRA 微调效果对比测试")
    print("=" * 70)

    # 1. 加载 tokenizer
    print("\n[1/3] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载 base 模型（4bit）
    print("\n[2/3] 加载 base 模型 (4bit)...")
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
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 3. 加载 LoRA adapter
    print("\n[3/3] 加载 LoRA adapter...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    print(f"  VRAM (含LoRA): {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 4. 跑测试用例
    results = []
    for i, tc in enumerate(TEST_CASES, 1):
        print("\n" + "=" * 70)
        print(f"测试 {i}/{len(TEST_CASES)}: {tc['name']}")
        print(f"用户: {tc['user']}")
        print(f"期望: {tc['expect']}")
        print("-" * 70)

        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": tc["user"]}]

        # 4a. 无 LoRA（base 模型原始输出）
        print("\n【A. 无 LoRA（base 原始）】")
        with model.disable_adapter():
            out_base, t_base = generate(model, tokenizer, messages)
        print(f"  耗时: {t_base:.1f}s")
        print(f"  输出: {out_base}")

        # 4b. 有 LoRA（微调后输出）
        print("\n【B. 有 LoRA（微调后）】")
        out_lora, t_lora = generate(model, tokenizer, messages)
        print(f"  耗时: {t_lora:.1f}s")
        print(f"  输出: {out_lora}")

        # 4c. 简要对比
        len_base = len(out_base)
        len_lora = len(out_lora)
        print("\n【对比】")
        print(f"  base 长度: {len_base} 字 | lora 长度: {len_lora} 字")
        print(f"  长度变化: {'+' if len_lora>len_base else ''}{len_lora-len_base} 字")

        results.append({
            "name": tc["name"],
            "user": tc["user"],
            "expect": tc["expect"],
            "base_output": out_base,
            "lora_output": out_lora,
            "base_time": round(t_base, 1),
            "lora_time": round(t_lora, 1),
        })

    # 5. 保存结果
    out_path = r"D:\RoleMatrix\scripts\test_compare_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\n结果已保存: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
