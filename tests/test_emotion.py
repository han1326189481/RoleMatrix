"""情绪引擎单元测试。"""
from __future__ import annotations

from datetime import datetime, timedelta

from rolematrix.emotion import EmotionEngine, EMOTION_DIMS


def test_apply_event_clamps_to_0_100() -> None:
    """事件更新应 clamp 到 [0, 100]。"""
    eng = EmotionEngine()
    # happy 基准 50，+1000 应被 clamp 到 100
    state = eng.apply_event("s1", {"happy": 1000})
    assert state.happy == 100
    # happy -1000 应被 clamp 到 0
    state = eng.apply_event("s1", {"happy": -1000})
    assert state.happy == 0


def test_apply_event_updates_updated_at() -> None:
    """事件更新应刷新 updated_at。"""
    eng = EmotionEngine()
    before = datetime.now()
    state = eng.apply_event("s2", {"happy": 5})
    assert state.updated_at >= before


def test_decay_converges_toward_baseline() -> None:
    """衰减应让偏离基准的值向基准收敛。"""
    eng = EmotionEngine()
    # 注入一个很久以前的高 happy 状态
    old_time = (datetime.now() - timedelta(hours=48)).isoformat()
    eng.load_state("s3", {"happy": 90, "updated_at": old_time})
    decayed = eng.apply_decay("s3")
    # 基准 happy=50，48 小时（2 个半衰期），factor=0.25
    # 期望：50 + (90-50)*0.25 = 60
    assert 50 < decayed.happy < 90
    assert abs(decayed.happy - 60) <= 1  # 允许四舍五入误差


def test_decay_no_change_when_just_updated() -> None:
    """刚刚更新过的状态，衰减应近似无变化。"""
    eng = EmotionEngine()
    # 用 load_state 设绝对值（apply_event 是增量语义）
    eng.load_state("s4", {"happy": 80})
    state = eng.apply_decay("s4")
    # 间隔极短，elapsed≈0，直接返回原 state
    assert state.happy == 80


def test_snapshot_load_state_roundtrip() -> None:
    """snapshot 与 load_state 应可往返恢复状态。"""
    eng = EmotionEngine()
    # 用 load_state 设绝对值，避免 apply_event 的增量语义干扰
    eng.load_state("s5", {"happy": 70, "sad": 30})
    snap = eng.snapshot("s5")

    eng2 = EmotionEngine()
    eng2.load_state("s5", snap)
    snap2 = eng2.snapshot("s5")
    # updated_at 也是 datetime，往返应一致
    assert snap["happy"] == snap2["happy"] == 70
    assert snap["sad"] == snap2["sad"] == 30


def test_all_dims_present() -> None:
    """7 个情绪维度都应存在。"""
    assert len(EMOTION_DIMS) == 7
    assert set(EMOTION_DIMS) == {
        "happy", "sad", "tired", "angry", "shy", "worried", "want_chat"
    }


def test_to_prompt_context_includes_all_dims() -> None:
    """prompt 上下文文本应包含所有维度。"""
    eng = EmotionEngine()
    ctx = eng.to_prompt_context("s6")
    for dim in EMOTION_DIMS:
        assert dim in ctx
    assert "情绪状态" in ctx
