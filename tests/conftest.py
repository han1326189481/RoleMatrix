"""pytest 公共配置：锁定项目根 + 重载配置。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _project_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """每个测试都锁定项目根目录，并清除配置缓存。"""
    monkeypatch.setenv("ROLEMATRIX_ROOT", str(PROJECT_ROOT))
    # 配置文件不存在时 get_settings 回退默认值，存在则加载
    from rolematrix.config import reload_settings

    reload_settings()
