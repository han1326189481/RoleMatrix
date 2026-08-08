"""聊天接口：支持单层（Ollama/DeepSeek 二选一）和双层（brain LoRA → mouth）架构。

通过 config.yaml 的 llm.mode 配置：
- single：按 llm.provider 走单层（ollama 或 deepseek）
- dual：双层架构，brain LoRA 决策 → mouth 生成

图片始终走本地 minicpm-v（DeepSeek-V4-Flash 不支持图片）。

回复分段：将长段回复切成多段返回，前端按真实打字速度分段发送，
避免一次性吐出大段文字造成的不真实感。
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import get_settings
from ..logger import get_logger
from .dispatcher import get_runtime
from ..llm import get_provider

log = get_logger("bridge.chat")

router = APIRouter(prefix="/chat", tags=["chat"])

# 会话历史（内存，按 session_key 隔离）
_histories: dict[str, list[dict[str, str]]] = {}

# 单段最大字数：超过这个长度会被进一步切分
_MAX_SEGMENT_CHARS = 60
# 单段最小字数：短于此值会与下一段合并
_MIN_SEGMENT_CHARS = 4


async def _auto_collect_user_image(image_base64: str, session_key: str) -> None:
    """收藏闭环：把用户发的图片（base64）自动存入收藏库。

    失败仅记日志，绝不阻断对话主流程。
    """
    import base64
    import tempfile

    try:
        header, _, b64 = image_base64.partition(",")
        ext_map = {
            "png": ".png", "jpeg": ".jpg", "jpg": ".jpg",
            "gif": ".gif", "webp": ".webp",
        }
        ext = ".jpg"
        header_lower = header.lower()
        for key, value in ext_map.items():
            if key in header_lower:
                ext = value
                break
        data = base64.b64decode(b64 if b64 else image_base64)
        if not data:
            return
        from ..tools.image_downloader import save_local_file

        # base64 无法直接入库（insert_item 需要文件实体），写临时文件后转存
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            await save_local_file(
                src_path=tmp_path,
                source="user_image",
                tags=[],
                description="用户发送的图片",
                session_key=session_key,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:  # noqa: BLE001
        log.warning("自动收藏用户图片失败 session=%s: %s", session_key, e)


def split_reply(text: str) -> list[str]:
    """把一段长回复切成多段，模拟真人分次发送。

    切分优先级：
    1. 显式换行
    2. 句末标点（。！？!?）
    3. 子句分隔（，；,;）
    4. 兜底按字数硬切

    太短的段会向后合并，避免出现零碎的单字。
    """
    text = (text or "").strip()
    if not text:
        return []

    # 第一步：按换行粗切
    raw_lines = [s.strip() for s in re.split(r"\s*\n\s*", text) if s.strip()]

    # 第二步：在每个句末标点后插入切分点，但保留标点本身
    sent_end = "。！？!?…"
    sub_sep = "，；,;"

    segments: list[str] = []
    for line in raw_lines:
        # 先按句末标点切
        buf = ""
        for ch in line:
            buf += ch
            if ch in sent_end:
                segments.append(buf.strip())
                buf = ""
        if buf.strip():
            segments.append(buf.strip())

    # 第三步：过长的段按子句分隔再切
    refined: list[str] = []
    for seg in segments:
        if len(seg) <= _MAX_SEGMENT_CHARS:
            refined.append(seg)
            continue
        # 长段按子句分隔切分
        buf = ""
        for ch in seg:
            buf += ch
            if ch in sub_sep and len(buf) >= _MAX_SEGMENT_CHARS // 2:
                refined.append(buf.strip())
                buf = ""
        if buf.strip():
            refined.append(buf.strip())

    # 第四步：兜底——还超长的段硬切
    final: list[str] = []
    for seg in refined:
        if len(seg) <= _MAX_SEGMENT_CHARS:
            final.append(seg)
            continue
        for i in range(0, len(seg), _MAX_SEGMENT_CHARS):
            final.append(seg[i:i + _MAX_SEGMENT_CHARS])

    # 第五步：合并过短的段（与下一段合并）
    merged: list[str] = []
    for seg in final:
        if not seg:
            continue
        if merged and len(merged[-1]) < _MIN_SEGMENT_CHARS:
            merged[-1] = merged[-1] + seg
        else:
            merged.append(seg)

    return merged if merged else [text]


class ChatRequest(BaseModel):
    message: str
    image_base64: str | None = None
    session_key: str = "webchat"
    model: str | None = None  # 覆盖默认模型


class ChatResponse(BaseModel):
    reply: str  # 完整原文（兼容旧前端）
    segments: list[str] = Field(default_factory=list)  # 分段后的数组
    emotion: dict[str, int]
    persona: dict[str, Any] = Field(default_factory=dict)
    model_used: str
    provider: str  # 标记用了哪个 provider
    mode: str = "single"  # single | dual
    brain_plan: dict[str, Any] = Field(default_factory=dict)  # 双层模式的大脑决策
    duration_ms: int


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """核心聊天端点：人格注入 + 情绪更新 + LLM 调用。

    根据 settings.llm.mode 决定走单层还是双层：
    - single：按 provider 走单层（Ollama 或 DeepSeek）
    - dual：双层架构（brain LoRA 决策 → mouth 生成）
    """
    t0 = time.time()
    rt = get_runtime()
    settings = get_settings()
    session = req.session_key

    # 1. 人格 + 情绪注入
    persona = rt.persona_registry.get(None)
    rt.emotion.apply_event(session, {"want_chat": -20, "happy": 5})
    await rt.persist_emotion(session)

    # 记忆接通：把用户消息写入 session 层记忆（事件总线订阅者消费）
    from ..eventbus import Event, bus
    await bus.publish(
        Event("message.received", {
            "session_key": session,
            "prompt": req.message or "[图片]",
            "channel": "webchat",
        })
    )

    persona_block = persona.to_prompt_block()
    emotion_block = rt.emotion.to_prompt_context(session)
    system_prompt = f"{persona_block}\n\n{emotion_block}"
    # 注入长期记忆（profile/long/daily 三层，无记忆时为空）
    from ..memory import build_memory_context
    memory_block = await build_memory_context(session)
    if memory_block:
        system_prompt += memory_block

    # 2. 选择 provider 和模型
    # 图片始终走本地（DeepSeek-V4-Flash 不支持图片，双层模式也不处理图片）
    has_image = bool(req.image_base64)
    if has_image:
        provider_name = "ollama"
        model = req.model or settings.llm.local_vision_model
        mode = "single"  # 图片强制单层
        # 收藏闭环：用户发的图片自动存入收藏库（失败仅日志，不阻断）
        await _auto_collect_user_image(req.image_base64, session)
    else:
        provider_name = settings.llm.provider
        mode = settings.llm.mode
        if provider_name == "deepseek":
            model = req.model or settings.llm.cloud_model
        else:
            model = req.model or settings.llm.local_model

    # 3. 获取历史
    history = _histories.setdefault(session, [])

    # 4. 调用 LLM（根据 mode 走单层或双层）
    brain_plan: dict[str, Any] = {}
    try:
        if mode == "dual" and not has_image:
            # 双层架构：brain LoRA → mouth
            from .dual_layer import dual_chat
            # 获取当前情绪向量给大脑
            emotion_vec = rt.emotion.get(session).as_vector()
            reply, brain_plan = await dual_chat(
                user_msg=req.message,
                base_system_prompt=system_prompt,
                history=history,
                emotion_context=emotion_vec,
                image_base64=None,
                session_key=session,
            )
        else:
            # 单层：直接走 provider
            provider = get_provider(provider_name)
            reply = await provider.chat(
                model=model,
                system_prompt=system_prompt,
                messages=history + [{"role": "user", "content": req.message}],
                image_base64=req.image_base64,
            )
    except Exception as e:
        log.error("LLM 调用失败 mode=%s provider=%s: %s", mode, provider_name, e)
        # 降级策略：
        # - dual 失败 → 降级到单层 DeepSeek
        # - deepseek 失败 → 降级到 Ollama
        # - ollama 失败 → 抛 500
        if mode == "dual":
            log.warning("双层架构失败，降级到单层 provider=%s", provider_name)
            mode = "single"  # 标记已降级
            provider = get_provider(provider_name)
            try:
                reply = await provider.chat(
                    model=model,
                    system_prompt=system_prompt,
                    messages=history + [{"role": "user", "content": req.message}],
                    image_base64=None,
                )
            except Exception as e2:
                log.error("双层降级后单层也失败: %s", e2)
                if provider_name == "deepseek":
                    provider_name = "ollama"
                    model = settings.llm.local_model
                    provider = get_provider(provider_name)
                    reply = await provider.chat(
                        model=model,
                        system_prompt=system_prompt,
                        messages=history + [{"role": "user", "content": req.message}],
                        image_base64=None,
                    )
                else:
                    raise HTTPException(500, f"LLM 调用失败: {e2}") from e2
        elif provider_name == "deepseek":
            log.warning("DeepSeek 失败，降级到本地 Ollama")
            provider_name = "ollama"
            model = settings.llm.local_model
            provider = get_provider(provider_name)
            reply = await provider.chat(
                model=model,
                system_prompt=system_prompt,
                messages=history + [{"role": "user", "content": req.message}],
                image_base64=None,
            )
        else:
            raise HTTPException(500, f"LLM 调用失败: {e}") from e

    # 5. 回复后情绪更新
    # 如果大脑输出了 emotion_delta，应用它；否则用默认衰减
    emotion_delta = brain_plan.get("emotion_delta") if brain_plan else None
    if emotion_delta and isinstance(emotion_delta, dict):
        # 限制范围 -5 ~ +5
        clamped = {k: max(-5, min(5, v)) for k, v in emotion_delta.items() if isinstance(v, int)}
        rt.emotion.apply_event(session, clamped)
    rt.emotion.apply_event(session, {"want_chat": -10})
    await rt.persist_emotion(session)

    # 6. 更新历史
    history.append({"role": "user", "content": req.message or "[图片]"})
    history.append({"role": "assistant", "content": reply})
    if len(history) > 20:
        _histories[session] = history[-20:]

    elapsed = int((time.time() - t0) * 1000)
    segments = split_reply(reply)

    # 记忆接通：把回复写入 session 层记忆（供当日 consolidate 摘要）
    await bus.publish(
        Event("message.sent", {
            "session_key": session,
            "assistant_texts": segments,
            "model": model,
        })
    )

    log.info("chat 完成 session=%s mode=%s provider=%s model=%s reply=%d字 切分=%d段 耗时=%dms",
             session, mode, provider_name, model, len(reply), len(segments), elapsed)

    return ChatResponse(
        reply=reply,
        segments=segments,
        emotion=rt.emotion.get(session).as_vector(),
        persona={
            "name": persona.display_name or persona.name,
            "greeting": persona.greeting,
        },
        model_used=model,
        provider=provider_name,
        mode=mode,
        brain_plan=brain_plan,
        duration_ms=elapsed,
    )


@router.post("/reset")
async def reset_session(session_key: str = "webchat") -> dict:
    """清空指定会话的历史。"""
    _histories.pop(session_key, None)
    return {"ok": True, "session_key": session_key}
