"""人格引擎单元测试。"""
from __future__ import annotations

from pathlib import Path

from rolematrix.persona import PersonaLoader, PersonaRegistry

PERSONAS_DIR = Path(__file__).resolve().parent.parent / "personas"


def test_load_default_persona() -> None:
    """能正确加载 default.yaml。"""
    loader = PersonaLoader(PERSONAS_DIR)
    p = loader.load("default")
    assert p.name == "default"
    assert p.display_name == "小R"
    assert p.greeting  # 非空


def test_prompt_block_contains_key_fields() -> None:
    """to_prompt_block 应包含人格关键字段。"""
    loader = PersonaLoader(PERSONAS_DIR)
    p = loader.load("default")
    block = p.to_prompt_block()
    # default.yaml 是沉浸式角色卡，渲染标题为"角色扮演指令"
    assert "角色扮演指令" in block
    assert "小R" in block
    # 语气信息来自 style.tone（渲染为"整体语气：…"）
    assert "语气" in block


def test_fallback_to_default_when_missing() -> None:
    """加载不存在的人格应回退到 default。"""
    loader = PersonaLoader(PERSONAS_DIR)
    p = loader.load("does-not-exist")
    assert p.name == "default"


def test_registry_assign_and_get() -> None:
    """registry 应能绑定 agent->persona 并取回。"""
    loader = PersonaLoader(PERSONAS_DIR)
    reg = PersonaRegistry(loader, default_name="default")
    # 未绑定时返回 default
    p = reg.get("unknown-agent")
    assert p.name == "default"
    # 绑定后返回指定人格
    reg.assign("agent-1", "default")
    assert reg.get("agent-1").name == "default"
    assert reg.persona_name_for("agent-1") == "default"


def test_loader_caches() -> None:
    """同一个人格第二次加载应命中缓存（同一对象）。"""
    loader = PersonaLoader(PERSONAS_DIR)
    p1 = loader.load("default")
    p2 = loader.load("default")
    assert p1 is p2


def test_reload_clears_cache() -> None:
    """reload 后应重新读文件。"""
    loader = PersonaLoader(PERSONAS_DIR)
    p1 = loader.load("default")
    loader.reload("default")
    p2 = loader.load("default")
    assert p1 is not p2  # 不同对象，缓存已清


def test_list_personas() -> None:
    """list_personas 应至少包含 default。"""
    loader = PersonaLoader(PERSONAS_DIR)
    names = loader.list_personas()
    assert "default" in names
