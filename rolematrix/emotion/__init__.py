"""情绪引擎：状态机 + 指数衰减 + 事件更新。"""
from .engine import EmotionEngine
from .models import EMOTION_DIMS, EmotionState

__all__ = ["EmotionEngine", "EmotionState", "EMOTION_DIMS"]
