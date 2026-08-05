"""记忆引擎：订阅事件总线，写入四层记忆（session/daily/long/profile）。

事件订阅（由 register_memory_handlers 在 Runtime 初始化时接通）：
- message.received: 收到用户消息 → 写入 session 层记忆
- message.sent: 回复发出 → 写入 session 层记忆

daily/long/profile 三层由 MemoryManager（memory/manager.py）管理：
- consolidate（心跳触发或 HTTP API）生成当日摘要与长期事实
- add_fact / HTTP /memory/note 显式写入画像与事实
"""
from __future__ import annotations

from ..eventbus import Event, bus
from ..logger import get_logger
from .manager import get_memory_manager

log = get_logger("memory.store")


async def on_message_received(event: Event) -> None:
    """收到用户消息：写入 session 层记忆。"""
    p = event.payload
    session_key = p.get("session_key") or "_default"
    prompt = p.get("prompt", "")
    if not prompt:
        return
    await get_memory_manager().record_message(
        session_key,
        role="user",
        content=prompt,
        metadata={
            "channel": p.get("channel"),
            "sender_id": p.get("sender_id"),
            "persona": p.get("persona"),
        },
    )


async def on_message_sent(event: Event) -> None:
    """回复发出：写入 session 层记忆。"""
    p = event.payload
    session_key = p.get("session_key") or "_default"
    texts = p.get("assistant_texts", []) or []
    if not texts:
        return
    # 合并多段回复为一条记忆（保留段落分隔）
    content = "\n".join(str(t) for t in texts if t)
    if not content:
        return
    await get_memory_manager().record_message(
        session_key,
        role="assistant",
        content=content,
        metadata={"model": p.get("model")},
    )


def register_memory_handlers() -> None:
    """在 Runtime 初始化时调用，接通事件总线订阅。"""
    bus.subscribe("message.received", on_message_received)
    bus.subscribe("message.sent", on_message_sent)
    log.debug("记忆引擎已订阅 message.received / message.sent")
