"""四层记忆（session/daily/long/profile）单元测试。

使用临时 SQLite 文件隔离，不污染 data/rolematrix.db。
consolidate 通过注入 fake provider 控制 LLM 行为，不发起真实网络请求。
"""
from __future__ import annotations

import pytest

from rolematrix.memory.manager import MemoryManager

S = "test-mem-session"


class _FakeProvider:
    """可控的 fake 嘴巴 provider。"""

    default_model = "fake-model"

    def __init__(self, reply: str | None = None, error: Exception | None = None):
        self.reply = reply
        self.error = error

    async def chat(self, model, system_prompt, messages, image_base64=None):
        if self.error is not None:
            raise self.error
        return self.reply or ""


@pytest.fixture(autouse=True)
async def _isolate_db(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """每个测试用独立临时数据库。"""
    db_file = tmp_path / "mem_test.db"
    monkeypatch.setattr("rolematrix.storage.db._DB_PATH", str(db_file))
    from rolematrix.storage.db import init_db

    await init_db()
    yield


def _new_manager() -> MemoryManager:
    return MemoryManager()


async def test_record_and_query_session() -> None:
    mgr = _new_manager()
    await mgr.record_message(S, "user", "今天好累啊")
    await mgr.record_message(S, "assistant", "辛苦了，歇会儿吧")
    await mgr.record_message(S, "user", "  ")  # 空内容应跳过
    items = await mgr.query_session(S, limit=10)
    assert len(items) == 2
    assert items[0]["role"] == "user"
    assert items[1]["role"] == "assistant"


async def test_add_fact_deduplicates() -> None:
    mgr = _new_manager()
    assert await mgr.add_fact(S, "long", "用户喜欢喝奶茶") is True
    assert await mgr.add_fact(S, "long", "用户喜欢喝奶茶") is False  # 去重
    items = await mgr.query_long(S)
    assert len(items) == 1


async def test_add_fact_invalid_layer_raises() -> None:
    mgr = _new_manager()
    with pytest.raises(ValueError):
        await mgr.add_fact(S, "bad-layer", "内容")
    # 空内容不应写入
    assert await mgr.add_fact(S, "long", "   ") is False


async def test_query_layers_roundtrip() -> None:
    mgr = _new_manager()
    await mgr.add_fact(S, "daily", "今天聊了数据库大作业")
    await mgr.add_fact(S, "long", "用户下周三是大作业截止日")
    await mgr.add_fact(S, "profile", "用户是计算机系学生")
    assert [m["content"] for m in await mgr.query_daily(S)] == ["今天聊了数据库大作业"]
    assert [m["content"] for m in await mgr.query_long(S)] == ["用户下周三是大作业截止日"]
    assert [m["content"] for m in await mgr.query_profile(S)] == ["用户是计算机系学生"]


async def test_build_memory_context() -> None:
    mgr = _new_manager()
    # 无记忆时返回空
    assert await mgr.build_memory_context(S) == ""
    await mgr.add_fact(S, "profile", "用户是计算机系学生")
    await mgr.add_fact(S, "long", "用户喜欢喝奶茶")
    ctx = await mgr.build_memory_context(S)
    assert "## 用户画像" in ctx
    assert "用户是计算机系学生" in ctx
    assert "## 关于用户的长期记忆" in ctx
    assert "用户喜欢喝奶茶" in ctx


async def test_consolidate_skips_when_no_messages() -> None:
    mgr = _new_manager()
    result = await mgr.consolidate(S, provider=_FakeProvider(reply="{}"))
    assert result["daily"] is None
    assert result["facts"] == []


async def test_consolidate_fallback_when_llm_fails() -> None:
    """LLM 失败时降级为拼接摘要，绝不抛错。"""
    mgr = _new_manager()
    await mgr.record_message(S, "user", "我下周要交数据库大作业了，好紧张")
    result = await mgr.consolidate(
        S, provider=_FakeProvider(error=RuntimeError("API 不可达"))
    )
    assert result["fallback"] is True
    assert result["daily"]  # 降级摘要非空
    assert "数据库大作业" in result["daily"]
    # daily 摘要已落库
    assert await mgr.has_daily_summary_today(S) is True


async def test_consolidate_extracts_summary_and_facts() -> None:
    """LLM 返回合法 JSON 时写入 daily 摘要 + long 事实。"""
    mgr = _new_manager()
    await mgr.record_message(S, "user", "我今天喝了一杯超好喝的奶茶")
    provider = _FakeProvider(
        reply='{"daily_summary": "用户今天喝了奶茶很开心", '
        '"facts": ["用户喜欢喝奶茶"]}'
    )
    result = await mgr.consolidate(S, provider=provider)
    assert result["fallback"] is False
    assert result["daily"] == "用户今天喝了奶茶很开心"
    assert result["facts"] == ["用户喜欢喝奶茶"]
    # 已落库
    long_items = await mgr.query_long(S)
    assert long_items[0]["content"] == "用户喜欢喝奶茶"


async def test_consolidate_idempotent_same_day() -> None:
    """同一天第二次 consolidate 应跳过（幂等）。"""
    mgr = _new_manager()
    await mgr.record_message(S, "user", "第一条消息")
    await mgr.consolidate(S, provider=_FakeProvider(error=RuntimeError("x")))
    result2 = await mgr.consolidate(S, provider=_FakeProvider(error=RuntimeError("x")))
    assert result2["skipped"] is True
    # daily 层只有一条摘要
    assert len(await mgr.query_daily(S, limit=50)) == 1
