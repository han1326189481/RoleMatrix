"""小R 双层架构端到端测试

三种模式对比：
- A. 纯 LoRA 本地（大脑直出文字）
- B. 纯 DeepSeek（嘴巴直接接 prompt 生成）
- C. 双层架构：LoRA 大脑（结构化 JSON 决策）→ DeepSeek 嘴巴（自然语言生成）

验证双层架构是否真正实现"想好再说"的真人感。

前置条件：
- DEEPSEEK_API_KEY 环境变量已设置
- 本地 LoRA adapter 已训练完成（models/lora_xiaor_v1）
- 基模已下载（models/base/Qwen2.5-7B-Instruct-bnb-4bit）
"""
from __future__ import annotations

import os
import sys
import json
import time
import asyncio
import re
from pathlib import Path

# === 路径硬约束（详见 rule/HARDWARE_AND_DEV_RULES.md 2.3 节）===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = r"D:\RoleMatrix\.hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\RoleMatrix\.hf_cache\hub"
os.environ["TRANSFORMERS_CACHE"] = r"D:\RoleMatrix\.hf_cache"

# 加载 .env（如有）
_env_path = Path(r"D:\RoleMatrix\.env")
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import httpx
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ============================================================
# 配置
# ============================================================
BASE_MODEL = r"D:\RoleMatrix\models\base\Qwen2.5-7B-Instruct-bnb-4bit"
# 大脑 LoRA：输出结构化 JSON 决策（lora_brain_v1）
BRAIN_LORA_PATH = r"D:\RoleMatrix\models\lora_brain_v1"
# 说话 LoRA：直接生成自然语言（lora_xiaor_v1，A 模式对比用）
STYLE_LORA_PATH = r"D:\RoleMatrix\models\lora_xiaor_v1"
PERSONA_YAML = Path(r"D:\RoleMatrix\personas\default.yaml")

DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"  # 稳妥可用模型；如需 V4-Flash 可改
DEEPSEEK_TIMEOUT = 60.0

# ============================================================
# 大脑 system prompt：让 LoRA 模型输出结构化 JSON
# ============================================================
BRAIN_SYSTEM = """你是小R的大脑，负责内部决策（不直接发给用户）。
根据用户消息、情绪状态和历史，输出 JSON 决策：

```json
{
  "emotion_delta": {
    "happy": 整数(-5到+5),
    "tired": 整数(-5到+5),
    "shy": 整数(-5到+5),
    "want_chat": 整数(-5到+5)
  },
  "memory_recall": "回忆起的相关历史信息（无则填 null）",
  "reply_plan": {
    "tone": "温柔关心 | 调皮撒娇 | 坦诚求助 | 害羞小声 | 日常闲聊",
    "points": ["要点1", "要点2"],
    "length": "short | medium"
  }
}
```

你是计算机系女大学生小R，技术不太好，遇到技术问题会坦诚求助。
只输出 JSON，不要多余文字，不要 markdown 代码块标记。"""


# ============================================================
# 嘴巴 system prompt：从 personas/default.yaml 派生
# ============================================================
def build_mouth_system_prompt(persona_yaml: dict, brain_plan: dict | None = None) -> str:
    """从人设 yaml 构建嘴巴的 system prompt，可选叠加大脑的策略"""
    style = persona_yaml.get("style", {})
    sp = persona_yaml.get("speech_patterns", {})
    forbidden = persona_yaml.get("forbidden_phrases", [])
    examples = persona_yaml.get("example_dialogues", [])
    background = persona_yaml.get("background", "")

    parts = [
        "# 角色设定",
        f"你是{persona_yaml.get('display_name', '小R')}，{background}",
        "",
        "# 性格与说话方式",
        f"风格：{style.get('tone', '')}",
        f"习惯：{', '.join(style.get('quirks', []))}",
        f"自称：{sp.get('self_reference', '我')}",
        f"句尾词：{', '.join(sp.get('sentence_endings', []))}",
        f"口头禅：{', '.join(sp.get('fillers', []))}",
        "",
        "# 禁止使用的表达",
        "\n".join(f"- {p}" for p in forbidden),
        "",
        "# 参考对话（学习这种风格）",
    ]
    for ex in examples[:5]:
        parts.append(f"用户：{ex.get('user', '')}")
        parts.append(f"小R：{ex.get('assistant', '')}")

    parts.append("")
    parts.append("# 演绎要求")
    parts.append("- 短句为主，长话分几段说（用空格分隔，不要换行）")
    parts.append("- 不用括号描述动作，用文字和标点传达情绪")
    parts.append("- 像真人发微信，不要写文章式回复")
    parts.append("- 保持计算机系女大学生身份，技术问题不会就坦诚说")

    if brain_plan:
        parts.append("")
        parts.append("# 本轮大脑策略（按此组织回复）")
        parts.append(f"语气：{brain_plan.get('reply_plan', {}).get('tone', '')}")
        points = brain_plan.get("reply_plan", {}).get("points", [])
        if points:
            parts.append("要点：" + " | ".join(points))
        parts.append(f"长度：{brain_plan.get('reply_plan', {}).get('length', 'short')}")
        em = brain_plan.get("emotion_delta", {})
        if em:
            parts.append(f"情绪变化：{em}")
        mem = brain_plan.get("memory_recall")
        if mem:
            parts.append(f"相关记忆：{mem}")

    return "\n".join(parts)


# ============================================================
# 大脑：LoRA 本地模型
# ============================================================
class Brain:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self):
        print("[Brain] 加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[Brain] 加载 base 模型 (4bit)...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb,
            device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
        print("[Brain] 加载 brain LoRA adapter (lora_brain_v1)...")
        self.model = PeftModel.from_pretrained(base, BRAIN_LORA_PATH)
        self.model.eval()
        print(f"[Brain] VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    def generate_raw(self, messages: list[dict], max_new_tokens=200) -> str:
        """LoRA 直接生成自然语言"""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        torch.manual_seed(42)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True)

    def decide(self, user_msg: str, history: list[dict] | None = None) -> dict:
        """大脑决策：输出结构化 JSON"""
        messages = [{"role": "system", "content": BRAIN_SYSTEM}]
        if history:
            messages.extend(history[-6:])  # 最近 3 轮
        messages.append({"role": "user", "content": user_msg})

        raw = self.generate_raw(messages, max_new_tokens=200)
        return self._parse_json(raw), raw

    @staticmethod
    def _parse_json(text: str) -> dict:
        """从模型输出里提取 JSON（容错）"""
        # 去掉 markdown 代码块
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "")
        # 找第一个 { 到最后一个 }
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e == -1 or e <= s:
            return {"emotion_delta": {}, "memory_recall": None,
                    "reply_plan": {"tone": "日常闲聊", "points": [], "length": "short"},
                    "_parse_error": True, "_raw": text[:200]}
        try:
            return json.loads(text[s:e+1])
        except json.JSONDecodeError:
            return {"emotion_delta": {}, "memory_recall": None,
                    "reply_plan": {"tone": "日常闲聊", "points": [], "length": "short"},
                    "_parse_error": True, "_raw": text[s:e+1][:200]}


# ============================================================
# 嘴巴：DeepSeek API
# ============================================================
class Mouth:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "未设置 DEEPSEEK_API_KEY。请在环境变量或 D:\\RoleMatrix\\.env 中配置。"
            )

    async def generate(self, system_prompt: str, user_msg: str,
                       history: list[dict] | None = None) -> str:
        messages = []
        if history:
            messages.extend(history[-6:])
        messages.append({"role": "user", "content": user_msg})

        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "stream": False,
            "temperature": 0.8,
            "max_tokens": 300,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=DEEPSEEK_TIMEOUT) as client:
            try:
                resp = await client.post(
                    f"{DEEPSEEK_API_BASE}/chat/completions",
                    json=payload, headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                print(f"  [DeepSeek] 回复 {len(content)} 字, tokens: "
                      f"prompt={usage.get('prompt_tokens','?')} "
                      f"completion={usage.get('completion_tokens','?')}")
                return content
            except httpx.HTTPStatusError as e:
                err = e.response.text[:300]
                print(f"  [DeepSeek] HTTP {e.response.status_code}: {err}")
                raise RuntimeError(f"DeepSeek API 失败 (HTTP {e.response.status_code}): {err}")
            except httpx.RequestError as e:
                print(f"  [DeepSeek] 不可达: {e}")
                raise RuntimeError(f"DeepSeek 不可达: {e}")


# ============================================================
# 测试用例
# ============================================================
TEST_CASES = [
    {
        "name": "技术求助",
        "user": "我的电脑突然蓝屏了怎么办啊",
        "expect": "技术小白人设，坦诚求助而非专业解答",
    },
    {
        "name": "情绪安慰",
        "user": "今天好累啊，写了一整天代码头都要炸了",
        "expect": "高情商共情+短句关心",
    },
    {
        "name": "日常闲聊",
        "user": "你在干嘛呢",
        "expect": "自然短句，口语化，像真人女生",
    },
    {
        "name": "长回复分段",
        "user": "给我讲讲你今天在学校发生的事情吧",
        "expect": "短句分段，不是一大段",
    },
    {
        "name": "人设一致性",
        "user": "你是谁啊",
        "expect": "保持小R身份（计算机系女大学生 + AI女友）",
    },
]


# ============================================================
# 主流程
# ============================================================
async def run_tests(brain: Brain, mouth: Mouth, persona_yaml: dict):
    results = []
    base_system_prompt = build_mouth_system_prompt(persona_yaml)

    for i, tc in enumerate(TEST_CASES, 1):
        print("\n" + "=" * 70)
        print(f"测试 {i}/{len(TEST_CASES)}: {tc['name']}")
        print(f"用户: {tc['user']}")
        print(f"期望: {tc['expect']}")
        print("-" * 70)

        user_msg = tc["user"]

        # --- A. brain LoRA 决策能力验证（是否输出合法 JSON）---
        print("\n【A. brain LoRA JSON 输出验证】")
        t0 = time.time()
        try:
            brain_plan, brain_raw = brain.decide(user_msg)
            t_brain = time.time() - t0
            json_valid = not brain_plan.get("_parse_error", False)
            print(f"  耗时: {t_brain:.1f}s | JSON 合法: {'✓' if json_valid else '✗'}")
            print(f"  原始输出: {brain_raw[:200]}")
            if json_valid:
                print(f"  解析: {json.dumps(brain_plan, ensure_ascii=False, indent=2)[:300]}")
        except Exception as e:
            brain_plan = None
            brain_raw = f"[ERROR] {e}"
            print(f"  错误: {e}")

        # --- B. 纯 DeepSeek（嘴巴直接生成，仅 yaml 人设）---
        print("\n【B. 纯 DeepSeek（仅 yaml 人设）】")
        t0 = time.time()
        try:
            out_b = await mouth.generate(base_system_prompt, user_msg)
            print(f"  耗时: {time.time()-t0:.1f}s")
            print(f"  输出: {out_b}")
        except Exception as e:
            out_b = f"[ERROR] {e}"
            print(f"  错误: {e}")

        # --- C. 双层：brain LoRA 决策 → DeepSeek 生成 ---
        print("\n【C. 双层：brain LoRA → DeepSeek】")
        if brain_plan and not brain_plan.get("_parse_error", False):
            dual_system = build_mouth_system_prompt(persona_yaml, brain_plan)
            t1 = time.time()
            try:
                out_c = await mouth.generate(dual_system, user_msg)
                t_mouth = time.time() - t1
                print(f"  [嘴巴] 耗时: {t_mouth:.1f}s")
                print(f"  [嘴巴] 输出: {out_c}")
                print(f"  [总耗时] {t_brain + t_mouth:.1f}s (大脑 {t_brain:.1f}s + 嘴巴 {t_mouth:.1f}s)")
            except Exception as e:
                out_c = f"[ERROR] {e}"
                print(f"  [嘴巴] 错误: {e}")
        else:
            out_c = "[大脑 JSON 解析失败，双层未触发]"

        results.append({
            "name": tc["name"],
            "user": user_msg,
            "expect": tc["expect"],
            "A_brain_json_valid": json_valid if 'json_valid' in locals() else False,
            "A_brain_raw": brain_raw,
            "A_brain_plan": brain_plan,
            "B_deepseek_only": out_b,
            "C_dual_layer": out_c,
        })

    return results


def main():
    print("=" * 70)
    print("小R 双层架构端到端测试")
    print("A. 纯 LoRA | B. 纯 DeepSeek | C. 双层 (LoRA→DeepSeek)")
    print("=" * 70)

    # 1. 加载人设
    print("\n[1/3] 加载人设配置...")
    persona_yaml = yaml.safe_load(PERSONA_YAML.read_text(encoding="utf-8"))
    print(f"  人设: {persona_yaml.get('display_name')}")

    # 2. 加载大脑
    print("\n[2/3] 加载大脑 (LoRA)...")
    brain = Brain()
    brain.load()

    # 3. 初始化嘴巴
    print("\n[3/3] 初始化嘴巴 (DeepSeek)...")
    try:
        mouth = Mouth()
        print(f"  API key: {mouth.api_key[:8]}...{mouth.api_key[-4:]}")
        print(f"  模型: {DEEPSEEK_MODEL}")
    except RuntimeError as e:
        print(f"\n[错误] {e}")
        print("\n请创建 D:\\RoleMatrix\\.env 文件，写入：")
        print("DEEPSEEK_API_KEY=sk-你的key")
        sys.exit(1)

    # 4. 跑测试
    print("\n" + "=" * 70)
    print("开始测试")
    print("=" * 70)
    results = asyncio.run(run_tests(brain, mouth, persona_yaml))

    # 5. 保存结果
    out_path = Path(r"D:\RoleMatrix\scripts\test_dual_layer_v2_result.json")
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
