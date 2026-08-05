"""情绪状态模型：7 维，每维 0-100。"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# 情绪维度名（顺序固定，便于向量计算）
EMOTION_DIMS: tuple[str, ...] = (
    "happy",
    "sad",
    "tired",
    "angry",
    "shy",
    "worried",
    "want_chat",
)


class EmotionState(BaseModel):
    happy: int = 50
    sad: int = 0
    tired: int = 20
    angry: int = 0
    shy: int = 10
    worried: int = 0
    want_chat: int = 60
    updated_at: datetime = Field(default_factory=datetime.now)

    def as_vector(self) -> dict[str, int]:
        """返回 7 维向量（不含 updated_at）。"""
        return {k: getattr(self, k) for k in EMOTION_DIMS}
