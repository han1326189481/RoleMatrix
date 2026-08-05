"""记忆引擎：四层记忆（session/daily/long/profile）。

- session: 当前会话原始消息流（事件总线自动写入）
- daily:   当日对话摘要（consolidate 生成）
- long:    长期事实/偏好（consolidate 提取 + 显式写入）
- profile: 用户画像（显式写入 / HTTP API）

入口：
- register_memory_handlers(): 接通事件总线（Runtime 初始化时调用）
- get_memory_manager() / build_memory_context(): 读写各层 + 组装注入文本
"""
from __future__ import annotations

from .manager import MemoryManager, get_memory_manager
from .stub import register_memory_handlers


async def build_memory_context(session_key: str) -> str:
    """组装可注入 system prompt 的长期记忆文本块（无记忆返回 ""）。"""
    return await get_memory_manager().build_memory_context(session_key)


__all__ = [
    "MemoryManager",
    "build_memory_context",
    "get_memory_manager",
    "register_memory_handlers",
]
