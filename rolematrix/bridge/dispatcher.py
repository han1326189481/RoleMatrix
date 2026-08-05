"""Hook 分发器：根据 hook 名调用 persona/emotion 模块，构造 hook Result。

Result 字段对齐 OpenClaw hook 返回类型（见 _ref/openclaw/src/plugins/）：
- before_prompt_build: {prependSystemContext, appendSystemContext, prependContext, appendContext, ...}
- heartbeat_prompt_contribution: {prependContext, appendContext}
- agent_turn_prepare: {prependContext, appendContext}
- llm_output / before_agent_reply: 观察类返回 {}
"""
from __future__ import annotations

from ..config import get_settings
from ..emotion import EmotionEngine
from ..eventbus import Event, bus
from ..logger import get_logger
from ..memory.stub import register_memory_handlers
from ..persona import PersonaLoader, PersonaRegistry
from ..storage.db import init_db, load_all_emotion_states, upsert_emotion_state
from .contracts import HookRequest

log = get_logger("bridge.dispatcher")


class Runtime:
    """桥接层持有的运行时依赖（模块级单例）。"""

    def __init__(self) -> None:
        settings = get_settings()
        root = settings.project_root
        self.persona_loader = PersonaLoader(root / settings.persona.config_dir)
        self.persona_registry = PersonaRegistry(
            self.persona_loader, settings.persona.default_persona
        )
        self.emotion = EmotionEngine()
        self.bus = bus
        # 接通事件总线订阅（Memory stub）
        register_memory_handlers()

    async def init_storage(self) -> None:
        """启动时初始化 SQLite + 恢复情绪状态。"""
        await init_db()
        for session_key, data in await load_all_emotion_states():
            self.emotion.load_state(session_key, data)
        log.info("运行时存储初始化完成")

    async def persist_emotion(self, session_key: str) -> None:
        """持久化一个 session 的情绪状态（失败仅日志，不阻断）。"""
        try:
            await upsert_emotion_state(session_key, self.emotion.snapshot(session_key))
        except Exception as e:  # noqa: BLE001
            log.error("情绪持久化失败 session=%s: %s", session_key, e)


# 模块级单例，server 复用
runtime: Runtime | None = None


def get_runtime() -> Runtime:
    global runtime
    if runtime is None:
        runtime = Runtime()
    return runtime


def _session_key(req: HookRequest) -> str:
    return req.ctx.session_key or req.ctx.chat_id or "_default"


async def dispatch(req: HookRequest) -> dict:
    """根据 hook 名分发到对应 handler，返回 hook Result。"""
    handler = _HANDLERS.get(req.hook)
    if handler is None:
        log.debug("未处理的 hook: %s（返回空 result）", req.hook)
        return {}
    try:
        return await handler(req)
    except Exception as e:  # noqa: BLE001
        log.error("hook %s 处理失败: %s", req.hook, e, exc_info=e)
        return {}  # 出错不阻断 OpenClaw 主流程


async def handle_before_prompt_build(req: HookRequest) -> dict:
    """核心注入点：人格描述 + 情绪状态 注入到 prompt。"""
    rt = get_runtime()
    session = _session_key(req)
    persona = rt.persona_registry.get(req.ctx.agent_id)

    # 收到用户消息 -> 情绪微调
    rt.emotion.apply_event(session, {"want_chat": -20, "happy": 5})
    await rt.persist_emotion(session)

    persona_block = persona.to_prompt_block()
    emotion_block = rt.emotion.to_prompt_context(session)

    # 异步通知记忆模块（P0 仅广播事件，不实际存储）
    await rt.bus.publish(
        Event(
            "message.received",
            {
                "session_key": session,
                "agent_id": req.ctx.agent_id,
                "channel": req.ctx.channel,
                "sender_id": req.ctx.sender_id,
                "prompt": req.event.get("prompt", ""),
                "persona": persona.name,
            },
        )
    )

    return {
        # 人格描述较稳定，prepend 到 system prompt（可被 provider cache，省 token）
        "prependSystemContext": persona_block,
        # 情绪每轮变，append 到 user context
        "appendContext": emotion_block,
    }


async def handle_heartbeat_prompt_contribution(req: HookRequest) -> dict:
    """心跳轮：情绪驱动是否贡献主动聊天 prompt。"""
    rt = get_runtime()
    session = _session_key(req)
    state = rt.emotion.apply_decay(session)
    await rt.persist_emotion(session)

    if state.want_chat < 50:
        return {}  # 不想聊天，不贡献

    persona = rt.persona_registry.get(req.ctx.agent_id)
    greeting = persona.greeting or "在吗？"
    # 情境化提示，让主动聊天也像真人
    hint = (
        f"[主动开口的冲动] 想找用户说话，可以用很自然的方式开口，"
        f"比如顺着此刻的心境说点什么。参考语气：{greeting}"
    )
    return {"appendContext": hint}


async def handle_agent_turn_prepare(req: HookRequest) -> dict:
    """同轮 context 补充，P0 占位。"""
    return {}


async def handle_llm_output(req: HookRequest) -> dict:
    """观察类：记录回复，回复后 want_chat 略降。"""
    rt = get_runtime()
    session = _session_key(req)
    rt.emotion.apply_event(session, {"want_chat": -10})
    await rt.persist_emotion(session)
    await rt.bus.publish(
        Event(
            "message.sent",
            {
                "session_key": session,
                "agent_id": req.ctx.agent_id,
                "assistant_texts": req.event.get("assistantTexts", []),
                "model": req.event.get("model"),
            },
        )
    )
    return {}


async def handle_before_agent_reply(req: HookRequest) -> dict:
    """短路 OpenClaw 默认 agent，由小R 双层架构生成回复。

    返回 {handled: true, reply: {text: ...}} 接管回复；
    失败时返回 {} 放行给 OpenClaw 默认 agent（gpt-5.5）。
    """
    rt = get_runtime()
    settings = get_settings()
    session = _session_key(req)

    # hook event: { cleanedBody: string }
    user_msg = req.event.get("cleanedBody", "")
    if not user_msg:
        return {}  # 无消息内容，放行

    # 人格 + 情绪注入
    persona = rt.persona_registry.get(req.ctx.agent_id)
    rt.emotion.apply_event(session, {"want_chat": -20, "happy": 5})

    persona_block = persona.to_prompt_block()
    emotion_block = rt.emotion.to_prompt_context(session)
    system_prompt = f"{persona_block}\n\n{emotion_block}"

    # 会话历史（与 chat router 共享，保持 webchat/openclaw session 隔离）
    from .chat import _histories
    history = _histories.setdefault(session, [])

    # 内存历史为空时，从数据库加载会话记忆（服务重启后的热启动）
    if not history:
        try:
            from ..storage import query_session_memory
            mem_items = await query_session_memory(session, limit=10)
            if mem_items:
                for m in mem_items:
                    role = m.get("role") or "user"
                    content = m.get("content") or ""
                    if role in ("user", "assistant") and content:
                        history.append({"role": role, "content": content})
                log.info("从数据库加载 %d 条会话记忆 session=%s", len(history), session)
        except Exception as e:  # noqa: BLE001
            log.warning("加载会话记忆失败 session=%s: %s", session, e)

    # 调用双层架构
    brain_plan: dict = {}
    try:
        if settings.llm.mode == "dual":
            from .dual_layer import dual_chat
            emotion_vec = rt.emotion.get(session).as_vector()
            reply, brain_plan = await dual_chat(
                user_msg=user_msg,
                base_system_prompt=system_prompt,
                history=history,
                emotion_context=emotion_vec,
                image_base64=None,
                session_key=session,
            )
        else:
            # 单层降级
            provider_name = settings.llm.provider
            if provider_name == "deepseek":
                model = settings.llm.cloud_model
            else:
                model = settings.llm.local_model
            from ..llm import get_provider
            provider = get_provider(provider_name)
            reply = await provider.chat(
                model=model,
                system_prompt=system_prompt,
                messages=history + [{"role": "user", "content": user_msg}],
                image_base64=None,
            )
    except Exception as e:  # noqa: BLE001
        log.error("before_agent_reply 生成回复失败: %s", e, exc_info=e)
        return {}  # 失败放行给默认 agent

    # 情绪更新
    emotion_delta = brain_plan.get("emotion_delta") if brain_plan else None
    if emotion_delta and isinstance(emotion_delta, dict):
        clamped = {
            k: max(-5, min(5, v))
            for k, v in emotion_delta.items()
            if isinstance(v, int)
        }
        rt.emotion.apply_event(session, clamped)
    rt.emotion.apply_event(session, {"want_chat": -10})
    await rt.persist_emotion(session)

    # 规范化分段标记：
    # LLM 可能输出字面字符串 "\n\n"（4个字符：反斜杠n反斜杠n）而非真换行符
    # 统一转换为真换行符 \n\n，让 TS 侧 splitReplySegments 能正确拆分
    raw_reply = reply
    if "\\n\\n" in raw_reply:
        reply = raw_reply.replace("\\n\\n", "\n\n")
        log.info(
            "检测到字面 \\n\\n 字符串，已转换为真换行符。段数=%d",
            len([s for s in reply.split("\n\n") if s.strip()]),
        )
    # 也处理单个字面 \n（少见，但兜底）
    if "\\n" in reply and "\n" not in reply:
        reply = reply.replace("\\n", "\n")

    # 处理 Markdown 硬换行：行尾两个空格+单换行 视为段落分隔
    # 这是 LLM 常见的"换段"写法，但实际只是 Markdown 视觉换行
    # 转换为 \n\n 让分段逻辑能正确拆分
    if "  \n" in reply and "\n\n" not in reply:
        reply = reply.replace("  \n", "\n\n")
        log.info(
            "检测到 Markdown 硬换行（行尾2空格+单换行），已转换为段落分隔符。段数=%d",
            len([s for s in reply.split("\n\n") if s.strip()]),
        )

    # 更新历史
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": reply})
    if len(history) > 20:
        _histories[session] = history[-20:]

    # 详细日志：打印 reply 的 repr 以便看到换行符真实形式
    segments = [s for s in reply.split("\n\n") if s.strip()]
    log.info(
        "before_agent_reply 短路 session=%s reply=%d字 段数=%d repr=%r",
        session, len(reply), len(segments), reply[:200],
    )

    return {
        "handled": True,
        "reply": {"text": reply},
        "reason": "rolematrix-bridge: 小R 双层架构接管",
    }


_HANDLERS = {
    "before_prompt_build": handle_before_prompt_build,
    "agent_turn_prepare": handle_agent_turn_prepare,
    "heartbeat_prompt_contribution": handle_heartbeat_prompt_contribution,
    "llm_output": handle_llm_output,
    "before_agent_reply": handle_before_agent_reply,
}
