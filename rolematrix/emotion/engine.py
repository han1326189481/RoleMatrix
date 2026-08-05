"""情绪引擎。

- 内存为主存储（按 sessionKey 隔离），保证同步快速访问
- 指数衰减：向各自基准值收敛 v = baseline + (v - baseline) * 0.5^(t/halflife)
- 事件更新：+/- 调整并 clamp 到 [0, 100]
- to_prompt_context：转成可注入 prompt 的文本
- load_state / snapshot：供 SQLite 持久化层使用（重启不丢）
"""
from __future__ import annotations

from datetime import datetime

from ..config import EmotionInitialConfig, get_settings
from ..logger import get_logger
from .models import EMOTION_DIMS, EmotionState

log = get_logger("emotion.engine")


def _decay_toward_baseline(value: int, baseline: int, factor: float) -> int:
    """向基准值衰减：剩余 = baseline + (value - baseline) * factor。"""
    return round(baseline + (value - baseline) * factor)


class EmotionEngine:
    def __init__(self) -> None:
        self._states: dict[str, EmotionState] = {}
        self._baseline: dict[str, int] = get_settings().emotion.initial.model_dump()
        self._halflife_hours: int = get_settings().emotion.decay_halflife_hours

    def get(self, session_key: str) -> EmotionState:
        if session_key not in self._states:
            self._states[session_key] = EmotionState(**self._baseline)
        return self._states[session_key]

    def apply_decay(self, session_key: str) -> EmotionState:
        """按经过时间做指数衰减，向基准值收敛。"""
        state = self.get(session_key)
        now = datetime.now()
        elapsed_hours = (now - state.updated_at).total_seconds() / 3600
        if elapsed_hours <= 0:
            return state
        factor = 0.5 ** (elapsed_hours / max(self._halflife_hours, 1))
        decayed: dict[str, int | datetime] = {}
        for k in EMOTION_DIMS:
            decayed[k] = _decay_toward_baseline(
                getattr(state, k), self._baseline[k], factor
            )
        decayed["updated_at"] = now
        new_state = state.model_copy(update=decayed)
        self._states[session_key] = new_state
        return new_state

    def apply_event(
        self, session_key: str, changes: dict[str, int]
    ) -> EmotionState:
        """事件驱动的情绪变化，例如收到消息 happy+5。"""
        state = self.apply_decay(session_key)
        updated: dict[str, int | datetime] = {}
        for k, delta in changes.items():
            if k not in EMOTION_DIMS:
                continue
            cur = getattr(state, k)
            updated[k] = max(0, min(100, cur + delta))
        updated["updated_at"] = datetime.now()
        new_state = state.model_copy(update=updated)
        self._states[session_key] = new_state
        log.debug(
            "情绪更新 session=%s changes=%s -> %s",
            session_key,
            changes,
            new_state.as_vector(),
        )
        return new_state

    def to_prompt_context(self, session_key: str) -> str:
        """生成注入 prompt 的情绪上下文文本。

        改进版：从裸数字 "happy=65" 变为情境描写，
        让 LLM 能"感受到"情绪而非"读取"情绪。
        """
        state = self.apply_decay(session_key)
        v = state.as_vector()
        cues: list[str] = []

        # happy: 心情
        if v["happy"] >= 75:
            cues.append("心情特别好，嘴角忍不住上扬")
        elif v["happy"] >= 55:
            cues.append("心情不错，有点小开心")
        elif v["happy"] <= 25:
            cues.append("心情有点低落，提不起劲")
        elif v["happy"] <= 40:
            cues.append("情绪一般，没什么起伏")

        # sad: 难过
        if v["sad"] >= 60:
            cues.append("有点难过，想被哄")
        elif v["sad"] >= 30:
            cues.append("心里闷闷的，不太想说笑")

        # tired: 疲倦
        if v["tired"] >= 70:
            cues.append("好困…脑子转得慢")
        elif v["tired"] >= 45:
            cues.append("有点累，反应可能慢半拍")

        # angry: 生气
        if v["angry"] >= 60:
            cues.append("在生气，不太想理人（但还是会回）")
        elif v["angry"] >= 30:
            cues.append("有点不高兴，可能会闹小别扭")

        # shy: 害羞
        if v["shy"] >= 75:
            cues.append("害羞得脸红，说话会结巴")
        elif v["shy"] >= 50:
            cues.append("有点不好意思")
        elif v["shy"] <= 10:
            cues.append("今天挺放松的，不那么害羞")

        # worried: 担心
        if v["worried"] >= 60:
            cues.append("心里有点担心用户，会忍不住多问几句")
        elif v["worried"] >= 30:
            cues.append("有点放心不下")

        # want_chat: 想聊天
        if v["want_chat"] >= 75:
            cues.append("很想聊天，话会变多")
        elif v["want_chat"] >= 50:
            cues.append("挺想聊的，会主动接话")
        elif v["want_chat"] <= 20:
            cues.append("有点不想说话，回复会很短")
        elif v["want_chat"] <= 35:
            cues.append("聊得差不多了，想安静一下")

        if not cues:
            return ""
        return "[此刻的心境] " + "，".join(cues)

    def load_state(self, session_key: str, data: dict) -> None:
        """从持久化层恢复一个 session 的情绪状态。"""
        self._states[session_key] = EmotionState.model_validate(data)

    def snapshot(self, session_key: str) -> dict:
        """导出一个 session 的情绪状态用于持久化。"""
        return self.get(session_key).model_dump()
