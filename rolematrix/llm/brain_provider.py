"""本地 LoRA 大脑 Provider。

封装 4bit base + brain LoRA adapter，做成长驻进程的模型单例。
职责：根据用户消息输出结构化 JSON 决策（emotion_delta / memory_recall / reply_plan），
供 DeepSeek 嘴巴根据决策组织最终回复。

设计要点：
1. 模型单例：进程内只加载一次 4bit 权重（~5GB），避免每次请求重载
2. 空 JSON fallback：大脑输出 `{}` 或非法 JSON 时降级为默认策略（日常闲聊）
3. 同步推理：transformers generate 是同步阻塞，包一层 to_thread.run 协程化
4. 路径硬约束：环境变量在模块顶部设置，所有缓存写死 D 盘
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from functools import lru_cache
from typing import Any

# === 路径硬约束（详见 rule/HARDWARE_AND_DEV_RULES.md 2.3 节）===
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", r"D:\RoleMatrix\.hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", r"D:\RoleMatrix\.hf_cache\hub")
os.environ.setdefault("HF_DATASETS_CACHE", r"D:\RoleMatrix\.hf_cache\datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\RoleMatrix\.hf_cache")
os.environ.setdefault("OLLAMA_MODELS", r"D:\RoleMatrix\.ollama\models")

import torch  # noqa: E402

from ..logger import get_logger  # noqa: E402

log = get_logger("llm.brain")

# ============================================================
# 默认配置（可被 config.yaml 覆盖）
# ============================================================
DEFAULT_BASE_MODEL = r"D:\RoleMatrix\models\base\Qwen2.5-7B-Instruct-bnb-4bit"
DEFAULT_BRAIN_LORA = r"D:\RoleMatrix\models\lora_brain_v1"

# ============================================================
# 大脑 system prompt：让 LoRA 输出结构化 JSON 决策
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
  },
  "web_search_query": "需要联网搜索时的关键词，纯闲聊填 null。只在用户问事实性问题（天气/新闻/某人是谁/技术概念/实时信息）时才填",
  "save_to_collection": "看到值得收藏的图片时填 {source, tags, reason}，否则 null。用户发的好看表情包、搜到的有趣图都可以存",
  "send_meme": "想发个表情包回应用户时填 {tag, reason}，否则 null。比如用户说开心的事，可以发个'开心'标签的图"
}
```

判断原则：
- web_search_query: 每天 ≤ 5 次，只在真的需要外部信息时填，纯闲聊不填
- save_to_collection: 偶尔触发，符合小R 喜好（动漫/甜食/可爱风）才存
- send_meme: 关系熟了之后偶尔用，不要太频繁（默认 null）

你是计算机系女大学生小R，技术不太好，遇到技术问题会坦诚求助。
只输出 JSON，不要多余文字，不要 markdown 代码块标记。

示例：
用户："今天上海天气怎么样"
→ {"web_search_query": "上海今天天气", "send_meme": null, "save_to_collection": null, ...}

用户："在吗"
→ 所有新字段填 null

用户："哈哈我今天好开心"
→ {"send_meme": {"tag": "开心", "reason": "用户开心，发个表情一起乐"}, "web_search_query": null, ...}"""

# ============================================================
# 空 JSON fallback：大脑输出异常时的兜底策略
# ============================================================
DEFAULT_PLAN: dict[str, Any] = {
    "emotion_delta": {"happy": 0, "tired": 0, "shy": 0, "want_chat": 1},
    "memory_recall": None,
    "reply_plan": {
        "tone": "日常闲聊",
        "points": ["简短自然回应"],
        "length": "short",
    },
}


def _default_plan(reason: str) -> dict[str, Any]:
    """生成带 _fallback 标记的默认策略"""
    plan = json.loads(json.dumps(DEFAULT_PLAN))  # 深拷贝
    plan["_fallback"] = True
    plan["_fallback_reason"] = reason
    return plan


def _parse_brain_json(text: str) -> dict[str, Any]:
    """从大脑原始输出提取 JSON，失败返回 None（不抛错）。

    支持的输入形式：
    - 纯 JSON：'{"emotion_delta": ...}'
    - 带 markdown 代码块：'```json\\n{...}\\n```'
    - JSON 前后有多余文字：'好的：{...}'
    - 空对象：'{}'（解析成功但为空 dict）

    宽容处理：即使缺少 reply_plan 字段，只要包含任意有效业务字段
    （web_search_query/save_to_collection/send_meme/emotion_delta/memory_recall），
    也接受并自动补全默认 reply_plan，让新增的 web search 等能力能正常工作。

    空对象 '{}' 不是失败：大脑训练时被告知"日常闲聊时所有字段填 null"，
    因此 '{}' 是符合预期的"无特别决策"，归一化为默认计划（带 _fallback 标记）。
    """
    if not text:
        return None
    # 去掉 markdown 代码块标记
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    # 找第一个 { 到最后一个 }
    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    try:
        plan = json.loads(cleaned[s : e + 1])
        # 非 dict 视为异常
        if not isinstance(plan, dict):
            return None
        # 空对象：大脑认为"日常闲聊，无需特别决策"（训练时被告知新字段填 null），
        # 归一化为默认计划而不是判失败，确保双层模式继续走嘴巴层正常回复。
        if not plan:
            log.info("大脑输出空决策 '{}'，按日常闲聊处理")
            return _default_plan("大脑输出空决策（日常闲聊）")

        # 容错：如果缺 reply_plan 但有其他业务字段，自动补全默认 reply_plan
        # （让大脑新增的 web_search_query 等字段能正常工作）
        if "reply_plan" not in plan:
            has_business_field = any(
                k in plan
                for k in (
                    "web_search_query",
                    "save_to_collection",
                    "send_meme",
                    "emotion_delta",
                    "memory_recall",
                )
            )
            if has_business_field:
                plan["reply_plan"] = {
                    "tone": "日常闲聊",
                    "points": ["简短自然回应"],
                    "length": "short",
                }
                log.info(
                    "大脑输出缺 reply_plan，已自动补全默认值（含其他业务字段）"
                )
            else:
                return None
        return plan
    except json.JSONDecodeError:
        return None


class BrainProvider:
    """本地 LoRA 大脑 provider（模型单例）。

    模型只在首次调用时加载（lazy init），后续请求复用同一实例。
    避免每次请求重新加载 5GB 权重。

    实现 LLMProvider Protocol 的子集：
    - decide() 是大脑专用方法（输出结构化 JSON）
    - chat() 是 LLMProvider 标准接口（用于单层 LoRA 直出，作 fallback）
    """

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        lora_path: str = DEFAULT_BRAIN_LORA,
    ) -> None:
        self.base_model = base_model
        self.lora_path = lora_path
        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()  # 加载过程加锁，避免并发重复加载

    # ---------- LLMProvider Protocol（用于单层模式作 fallback）----------

    @property
    def default_model(self) -> str:
        return "lora-brain-v1"

    async def chat(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        image_base64: str | None = None,
    ) -> str:
        """LLMProvider 标准接口：直出自然语言（不输出 JSON）。

        仅在 dual 模式 brain 失败、需要降级到单层 LoRA 时用。
        """
        await self._ensure_loaded()
        msgs = [{"role": "system", "content": system_prompt}] + messages
        return await self._generate_async(msgs, max_new_tokens=200)

    # ---------- 大脑专用接口 ----------

    async def decide(
        self,
        user_msg: str,
        emotion_context: dict[str, int] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """大脑决策：输出结构化 JSON。

        Args:
            user_msg: 用户最新消息
            emotion_context: 当前情绪向量（可选，注入到 user prompt）
            history: 最近对话历史（取最近 6 条）

        Returns:
            决策 JSON dict。出错时返回带 _fallback 标记的默认策略。
        """
        user_msg = (user_msg or "").strip()
        if not user_msg:
            # 空消息不加载 5GB 模型，直接返回默认计划
            return _default_plan("用户消息为空")

        await self._ensure_loaded()

        # 组装 messages
        user_content = f"用户消息：{user_msg}"
        if emotion_context:
            user_content += f"\n情绪状态：{json.dumps(emotion_context, ensure_ascii=False)}"
        else:
            user_content += "\n情绪状态：{}"
        # 历史摘要注入到 user_content（取最近 6 条，仅保留 role/content 摘要）
        hist_brief = history[-6:] if history else []
        user_content += f"\n历史：{json.dumps(hist_brief, ensure_ascii=False)}"

        messages = [{"role": "system", "content": BRAIN_SYSTEM}]
        if history:
            messages.extend(history[-6:])
        messages.append({"role": "user", "content": user_content})

        # 生成（同步阻塞，包到 thread 里）
        try:
            raw = await self._generate_async(messages, max_new_tokens=250)
        except Exception as e:
            log.error("大脑生成失败: %s", e)
            return _default_plan(f"生成异常: {e}")

        # 解析 JSON
        plan = _parse_brain_json(raw)
        if plan is None:
            log.warning("大脑输出 JSON 解析失败，使用 fallback。原始输出: %s", raw[:200])
            return _default_plan("JSON 解析失败")
        return plan

    # ---------- 内部加载与生成 ----------

    async def _ensure_loaded(self) -> None:
        """懒加载模型（首次调用时加载，进程内单例）。"""
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:  # double-check
                return
            await asyncio.to_thread(self._load_model)

    def _load_model(self) -> None:
        """同步加载 4bit base + LoRA（在 to_thread 里跑，不阻塞事件循环）。"""
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        log.info("加载 tokenizer: %s", self.base_model)
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log.info("加载 4bit base 模型: %s", self.base_model)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        log.info("加载 brain LoRA adapter: %s", self.lora_path)
        model = PeftModel.from_pretrained(base, self.lora_path)
        model.eval()

        self._model = model
        self._tokenizer = tokenizer
        vram_gb = torch.cuda.memory_allocated() / 1024**3
        log.info("大脑加载完成，VRAM 占用: %.2f GB", vram_gb)

    async def _generate_async(
        self, messages: list[dict[str, str]], max_new_tokens: int = 200
    ) -> str:
        """协程包装的同步 generate。"""
        return await asyncio.to_thread(
            self._generate_sync, messages, max_new_tokens
        )

    def _generate_sync(
        self, messages: list[dict[str, str]], max_new_tokens: int = 200
    ) -> str:
        """同步生成（在 to_thread 里跑）。"""
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        torch.manual_seed(42)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


# ============================================================
# 模块级单例（进程内复用，避免重复加载 5GB 权重）
# ============================================================
_brain_singleton: BrainProvider | None = None


def get_brain(
    base_model: str = DEFAULT_BASE_MODEL,
    lora_path: str = DEFAULT_BRAIN_LORA,
) -> BrainProvider:
    """获取大脑单例（进程内只加载一次）。

    注意：base_model 和 lora_path 只在首次调用时生效，
    后续调用会忽略参数直接返回已加载的单例。要换 LoRA 必须重启进程。
    """
    global _brain_singleton
    if _brain_singleton is None:
        _brain_singleton = BrainProvider(base_model, lora_path)
    return _brain_singleton


def reset_brain() -> None:
    """重置单例（测试用，会触发下次重新加载）。"""
    global _brain_singleton
    if _brain_singleton is not None:
        # 显存释放
        if _brain_singleton._model is not None:
            del _brain_singleton._model
            torch.cuda.empty_cache()
    _brain_singleton = None
