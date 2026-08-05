"""收藏闭环单元测试：大脑决策消费 + 用户图片自动收藏。

使用临时收藏库目录隔离（不写 .xiaor_collection）。
"""
from __future__ import annotations

import base64

import pytest

from rolematrix.bridge.chat import _auto_collect_user_image
from rolematrix.bridge.dual_layer import (
    _handle_save_to_collection,
    _handle_send_meme,
    _split_tags,
)
from rolematrix.config import get_settings

# 1x1 PNG（红色像素）
PNG_1PX_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture(autouse=True)
async def _isolate_collection(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """每个测试用独立临时收藏库。"""
    settings = get_settings()
    monkeypatch.setattr(settings.collection, "root_dir", str(tmp_path / "collection"))
    from rolematrix.tools.collection_store import init_collection_db

    await init_collection_db()
    yield


def test_split_tags() -> None:
    assert _split_tags(["开心", "可爱"]) == ["开心", "可爱"]
    assert _split_tags("开心,可爱") == ["开心", "可爱"]
    assert _split_tags("开心，可爱；软萌") == ["开心", "可爱", "软萌"]
    assert _split_tags(None) == []
    assert _split_tags("") == []


async def test_send_meme_hits_by_tag(tmp_path) -> None:
    """send_meme 按 tag 命中收藏并递增使用计数。"""
    from rolematrix.tools.image_downloader import save_local_file

    img = tmp_path / "meme.png"
    img.write_bytes(base64.b64decode(PNG_1PX_B64))
    await save_local_file(str(img), source="user_image", tags=["开心", "可爱"])

    result = await _handle_send_meme({"tag": "开心", "reason": "用户开心"}, "s1")
    assert result is not None
    assert result["tag"] == "开心"
    assert result["file_path"].endswith(".png")
    assert result["usage_count"] == 1

    # 再次命中：usage 递增
    result2 = await _handle_send_meme({"tag": "可爱"}, "s1")
    assert result2 is not None
    assert result2["usage_count"] == 2


async def test_send_meme_miss_returns_none() -> None:
    """未命中标签/空决策返回 None（不抛错）。"""
    assert await _handle_send_meme({"tag": "不存在的标签"}, "s1") is None
    assert await _handle_send_meme({}, "s1") is None


async def test_save_to_collection_local_file(tmp_path) -> None:
    """save_to_collection 本地文件路径 → 收藏成功。"""
    img = tmp_path / "pic.png"
    img.write_bytes(base64.b64decode(PNG_1PX_B64))
    saved = await _handle_save_to_collection(
        {"source": str(img), "tags": "甜食", "reason": "看起来很好吃"}, "s1"
    )
    assert saved is not None
    assert saved["file_path"].endswith(".png")
    assert saved["tags"] == ["甜食"]


async def test_save_to_collection_invalid_source() -> None:
    """无效 source 返回 None（不抛错）。"""
    assert await _handle_save_to_collection({"source": "", "tags": []}, "s1") is None
    assert (
        await _handle_save_to_collection({"source": "不存在的路径"}, "s1") is None
    )


async def test_auto_collect_user_image() -> None:
    """用户发的图片（base64）自动进收藏库。"""
    data_uri = "data:image/png;base64," + PNG_1PX_B64
    await _auto_collect_user_image(data_uri, "s2")
    from rolematrix.tools.collection_store import count_items, list_recent

    assert await count_items() == 1
    items = await list_recent()
    assert items[0]["source"] == "user_image"


async def test_auto_collect_user_image_invalid_base64() -> None:
    """非法 base64 不抛错、不写入。"""
    await _auto_collect_user_image("data:image/png;base64,!!!not-base64!!!", "s2")
    from rolematrix.tools.collection_store import count_items

    assert await count_items() == 0
