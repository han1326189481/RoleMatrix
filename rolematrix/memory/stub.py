"""记忆引擎 P1 实现。

订阅事件总线：
- message.received: 收到用户消息 → 写入 session 层记忆
- message.sent: 回复发出 → 写入 session 层记忆 + 触发 long 层摘要（P2）

四层记忆：
- session: 当前会话原始消息流（最近 N 条）
- daily: 跨会话的当日摘要（P2）
- long: 长期事实/偏好（P2）
- profile: 用户画像（P2）
"""
from __future__ import annotations

from ..eventbus import Event, bus
from ..logger import get_logger
from ..storage import insert_memory_item

log = get_logger("memory.store")


async def on_message_received(event: Event) -> None:
    """收到用户消息：写入 session 层记忆。"""
    p = event.payload
    session_key = p.get("session_key") or "_default"
    prompt = p.get("prompt", "")
    if not prompt:
        return
    try:
        await insert_memory_item(
            session_key=session_key,
            layer="session",
            role="user",
            content=prompt,
            metadata={
                "channel": p.get("channel"),
                "sender_id": p.get("sender_id"),
                "persona": p.get("persona"),
            },
        )
        log.debug(
            "[记忆] 写入 session user msg session=%s len=%d",
            session_key, len(prompt),
        )
    except Exception as e:  # noqa: BLE001
        log.error("[记忆] 写入用户消息失败: %s", e)


async def on_message_sent(event: Event) -> None:
    """回复发出：写入 session 层记忆。P2 触发 long 层摘要抽取。"""
    p = event.payload
    session_key = p.get("session_key") or "_default"
    texts = p.get("assistant_texts", []) or []
    if not texts:
        return
    # 合并多段回复为一条记忆（保留段落分隔）
    content = "\n".join(str(t) for t in texts if t)
    if not content:
        return
    try:
        await insert_memory_item(
            session_key=session_key,
            layer="session",
            role="assistant",
            content=content,
            metadata={"model": p.get("model")},
        )
        log.debug(
            "[记忆] 写入 session assistant msg session=%s len=%d",
            session_key, len(content),
        )
    except Exception as e:  # noqa: BLE001
        log.error("[记忆] 写入助手回复失败: %s", e)


def register_memory_handlers() -> None:
    """在 Runtime 初始化时调用，接通事件总线订阅。"""
    bus.subscribe("message.received", on_message_received)
    bus.subscribe("message.sent", on_message_sent)
    log.debug("记忆引擎已订阅 message.received / message.sent")
