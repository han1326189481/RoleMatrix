"""记忆引擎。

P1: 订阅事件总线，写入 SQLite 四层记忆（session/daily/long/profile）。
"""
from .stub import register_memory_handlers

__all__ = ["register_memory_handlers"]
