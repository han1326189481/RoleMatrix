"""内部异步事件总线。

用于 RoleMatrix 各模块间解耦通信（例如 Memory 更新后通知 Emotion）。
桥接层从 OpenClaw 收到的 hook 由 dispatcher 直接调用各模块；
模块内部需要广播事件时走这里。
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Union

from .logger import get_logger

log = get_logger("eventbus")

AsyncHandler = Callable[["Event"], Union[Awaitable[None], None]]


@dataclass
class Event:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """异步事件总线：订阅 + 发布。"""

    def __init__(self) -> None:
        self._handlers: dict[str, list[AsyncHandler]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: AsyncHandler) -> None:
        self._handlers[event_type].append(handler)

    async def publish(self, event: Event) -> None:
        handlers = self._handlers.get(event.type, [])
        if not handlers:
            return
        log.debug("publish %s -> %d handler(s)", event.type, len(handlers))
        results = await asyncio.gather(
            *(h(event) for h in handlers), return_exceptions=True
        )
        for r in results:
            if isinstance(r, Exception):
                log.error("event handler error: %s", r, exc_info=r)


# 全局单例
bus = EventBus()
