"""双层架构编排器：大脑 (LoRA) → 嘴巴 (provider)。

核心流程：
1. 大脑根据用户消息 + 情绪 + 历史输出结构化 JSON 决策
2. 把决策拼接到嘴巴的 system prompt
3. 嘴巴根据决策组织自然语言回复

降级策略：
- 大脑失败（生成异常或 JSON 解析失败）→ 返回 fallback 决策，继续走嘴巴
- 嘴巴失败 → 抛异常给上层（chat.py 会降级到 Ollama）
- 完全失败（大脑+嘴巴都失败）→ 上层降级到单层 Ollama

不重复实现 LLMProvider 接口，作为 chat.py 的协程工具模块被调用。
"""
from __future__ import annotations

import json
from typing import Any

from ..config import get_settings
from ..llm import get_provider
from ..llm.brain_provider import BrainProvider, get_brain
from ..logger import get_logger

log = get_logger("bridge.dual_layer")


def build_mouth_system_prompt(
    base_system_prompt: str,
    brain_plan: dict[str, Any] | None,
    search_results: list[dict[str, Any]] | None = None,
) -> str:
    """把大脑决策拼到嘴巴的 system prompt 后面。

    Args:
        base_system_prompt: 原始人设 + 情绪 prompt
        brain_plan: 大脑输出的决策 JSON（含 emotion_delta / reply_plan 等）
        search_results: web search 结果（可选，注入到 prompt 让小R 引用）

    Returns:
        增强后的 system prompt
    """
    if not brain_plan:
        return base_system_prompt

    parts = [base_system_prompt, "", "# 本轮大脑策略（按此组织回复）"]

    reply_plan = brain_plan.get("reply_plan") or {}
    tone = reply_plan.get("tone")
    if tone:
        parts.append(f"语气：{tone}")

    points = reply_plan.get("points") or []
    if points:
        parts.append("要点：" + " | ".join(str(p) for p in points))
        # 多个要点时，强制每个要点单独成段
        if len(points) >= 2:
            parts.append(f"⚠️ 本次有 {len(points)} 个要点，必须分成 {len(points)} 段独立消息发送，")
            parts.append("段与段之间用一个空行（两个连续换行符）分隔。")

    length = reply_plan.get("length")
    if length:
        parts.append(f"长度：{length}")

    em = brain_plan.get("emotion_delta")
    if em and isinstance(em, dict) and any(v != 0 for v in em.values()):
        parts.append(f"情绪变化：{em}")

    mem = brain_plan.get("memory_recall")
    if mem:
        parts.append(f"相关记忆：{mem}")

    # 注入 web search 结果
    if search_results:
        from ..tools.web_search import format_results_for_prompt
        parts.append(format_results_for_prompt(search_results))

    # 强化分段约束（解决测试中 length=short 仍输出长段的问题）
    parts.append("")
    parts.append("# 演绎要求（必须遵守）")
    parts.append("- 严格按长度策略：short ≤ 40字，medium ≤ 100字")
    parts.append("- 短句为主，长话分几段说，每段是一条独立消息")
    parts.append("- 段落之间必须留一个空行，即按两次回车键分隔段落")
    parts.append("- 严禁使用 Markdown 硬换行（行尾两个空格+单换行）")
    parts.append("- 严禁使用单换行分隔段落，必须是空行（两个连续换行符）")
    parts.append("- 不用括号描述动作")
    parts.append("")
    parts.append("# 分段示例")
    parts.append("用户：你今天怎么样？")
    parts.append("正确回复（注意段落之间有空行）：")
    parts.append("今天还行啦，刚写完一段代码有点累")
    parts.append("")  # 这是一个空行，表示段落分隔
    parts.append("你呢，今天过得怎么样呀？")
    parts.append("")
    parts.append("# 反例（错误，会被合并成一条消息）")
    parts.append('错误回复（Markdown硬换行，行尾两个空格+单换行）："还行  \\n你呢"')
    parts.append('错误回复（单换行）："还行\\n你呢"')
    parts.append("正确做法：在两段之间留一个完全空的行（没有任何字符）。")

    return "\n".join(parts)


async def dual_chat(
    user_msg: str,
    base_system_prompt: str,
    history: list[dict[str, str]] | None = None,
    emotion_context: dict[str, int] | None = None,
    image_base64: str | None = None,
    session_key: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """双层架构主入口：brain 决策 → web search → mouth 生成。

    Args:
        user_msg: 用户最新消息
        base_system_prompt: 人设 + 情绪的 system prompt
        history: 对话历史
        emotion_context: 当前情绪向量（注入到大脑 user prompt）
        image_base64: 图片（双层模式不处理图片，由 chat.py 提前路由到 Ollama）
        session_key: 会话标识，用于 web search 限流

    Returns:
        (reply_text, brain_plan)：
        - reply_text: 嘴巴生成的最终回复
        - brain_plan: 大脑的决策 JSON（含 _fallback 标记如果降级了）
    """
    settings = get_settings()
    history = history or []

    # 1. 大脑决策
    brain = get_brain(
        base_model=settings.llm.brain_base_model,
        lora_path=settings.llm.brain_lora_path,
    )
    try:
        brain_plan = await brain.decide(
            user_msg=user_msg,
            emotion_context=emotion_context,
            history=history,
        )
        is_fallback = brain_plan.get("_fallback", False)
        if is_fallback:
            log.warning("大脑降级: %s", brain_plan.get("_fallback_reason"))
        else:
            log.info(
                "大脑决策 tone=%s points=%s search_query=%s",
                brain_plan.get("reply_plan", {}).get("tone"),
                brain_plan.get("reply_plan", {}).get("points"),
                brain_plan.get("web_search_query"),
            )
    except Exception as e:
        log.error("大脑 decide 异常: %s", e)
        from ..llm.brain_provider import _default_plan
        brain_plan = _default_plan(f"decide 异常: {e}")

    # 2. web search（如果大脑决策含 web_search_query）
    search_results: list[dict[str, Any]] = []
    search_query = brain_plan.get("web_search_query")
    if search_query and isinstance(search_query, str) and search_query.strip():
        try:
            from ..tools.web_search import search as web_search
            from ..tools.collection_store import insert_search_history
            search_results = await web_search(
                query=search_query.strip(),
                session_key=session_key or "default",
            )
            # 记录搜索历史到收藏库
            try:
                await insert_search_history(
                    query=search_query.strip(),
                    session_key=session_key,
                    results=search_results,
                )
            except Exception as e:
                log.warning("记录搜索历史失败: %s", e)
        except Exception as e:
            log.warning("web search 执行失败: %s", e)

    # 3. 嘴巴生成（用大脑策略 + 搜索结果增强 system prompt）
    mouth_system = build_mouth_system_prompt(
        base_system_prompt, brain_plan, search_results=search_results
    )

    provider = get_provider(settings.llm.provider)
    model = settings.llm.cloud_model if settings.llm.provider == "deepseek" else settings.llm.local_model

    try:
        reply = await provider.chat(
            model=model,
            system_prompt=mouth_system,
            messages=history + [{"role": "user", "content": user_msg}],
            image_base64=None,  # 双层模式不处理图片
        )
    except Exception as e:
        log.error("嘴巴 provider=%s 失败: %s", settings.llm.provider, e)
        raise  # 上层 chat.py 会降级到 Ollama

    return reply, brain_plan


def reset_brain_singleton() -> None:
    """重置大脑单例（测试用）。"""
    from ..llm.brain_provider import reset_brain
    reset_brain()
