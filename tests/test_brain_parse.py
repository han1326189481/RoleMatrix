"""大脑 JSON 解析与空决策兜底单元测试（不加载模型）。"""
from __future__ import annotations

from rolematrix.llm.brain_provider import (
    BrainProvider,
    _default_plan,
    _parse_brain_json,
)


def test_parse_valid_json() -> None:
    """合法 JSON 应原样解析。"""
    text = '{"emotion_delta": {"happy": 1}, "reply_plan": {"tone": "日常闲聊"}}'
    plan = _parse_brain_json(text)
    assert plan is not None
    assert plan["reply_plan"]["tone"] == "日常闲聊"
    assert "_fallback" not in plan


def test_parse_markdown_code_block() -> None:
    """带 markdown 代码块的 JSON 应能解析。"""
    text = '```json\n{"emotion_delta": {"happy": 1}}\n```'
    plan = _parse_brain_json(text)
    assert plan is not None
    assert plan["emotion_delta"]["happy"] == 1


def test_parse_json_with_surrounding_text() -> None:
    """JSON 前后有杂文字的应能提取。"""
    text = '好的：{"reply_plan": {"tone": "温柔关心"}} 就这样'
    plan = _parse_brain_json(text)
    assert plan is not None
    assert plan["reply_plan"]["tone"] == "温柔关心"


def test_parse_blank_returns_none() -> None:
    """空字符串/纯空白是真失败，返回 None。"""
    assert _parse_brain_json("") is None
    assert _parse_brain_json("   \n  ") is None
    assert _parse_brain_json("随便说点什么") is None


def test_parse_empty_object_becomes_default_plan() -> None:
    """空对象 '{}' 是大脑的'日常闲聊'决策，应归一化为默认计划而非判失败。

    回归：修复前 '{}' 被当 JSON 解析失败，导致双层模式对短输入
    （如"在吗""你在干嘛呢"）直接标记"双层未触发"。
    """
    for text in ("{}", "{}\n", "{} ", "{\n} "):
        plan = _parse_brain_json(text)
        assert plan is not None, f"输入 {text!r} 应被归一化而不是判失败"
        assert plan.get("_fallback") is True
        assert plan["reply_plan"]["tone"] == "日常闲聊"
        assert "日常闲聊" in plan.get("_fallback_reason", "")


def test_parse_missing_reply_plan_autofills() -> None:
    """含业务字段但缺 reply_plan 时应自动补全。"""
    plan = _parse_brain_json('{"web_search_query": "今天天气"}')
    assert plan is not None
    assert plan["web_search_query"] == "今天天气"
    assert plan["reply_plan"]["tone"] == "日常闲聊"


def test_default_plan_marks_fallback() -> None:
    """_default_plan 应带 _fallback 标记与原因。"""
    plan = _default_plan("测试原因")
    assert plan["_fallback"] is True
    assert plan["_fallback_reason"] == "测试原因"
    assert plan["reply_plan"]["tone"] == "日常闲聊"


async def test_decide_empty_message_returns_fallback_without_model() -> None:
    """空消息应直接返回默认计划，且不触发模型加载。"""
    brain = BrainProvider()
    # 不加载模型的路径：空消息在 _ensure_loaded 之前就返回
    plan = await brain.decide("  ")
    assert plan["_fallback"] is True
    assert "用户消息为空" in plan["_fallback_reason"]
    # 单例模型未被加载（_model 仍为 None，证明没走模型路径）
    assert brain._model is None
