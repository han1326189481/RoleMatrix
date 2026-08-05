"""人格 YAML 加载器，支持热加载（缓存 + 失效）。"""
from __future__ import annotations

from pathlib import Path

import yaml

from ..logger import get_logger
from .models import Persona

log = get_logger("persona.loader")


class PersonaLoader:
    def __init__(self, config_dir: str | Path) -> None:
        self.config_dir = Path(config_dir)
        self._cache: dict[str, Persona] = {}

    def load(self, name: str) -> Persona:
        if name in self._cache:
            return self._cache[name]
        path = self.config_dir / f"{name}.yaml"
        if not path.exists():
            # 找不到则回退到 default；再找不到就用空配置构造一个
            log.warning("人格 %s 不存在，回退 default", name)
            if name != "default":
                return self.load("default")
            return Persona(name="default", description="默认人格")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        persona = Persona.model_validate(data)
        self._cache[name] = persona
        log.info("加载人格 %s (%s)", name, persona.display_name or persona.name)
        return persona

    def reload(self, name: str | None = None) -> None:
        if name:
            self._cache.pop(name, None)
            log.info("人格 %s 已重载", name)
        else:
            self._cache.clear()
            log.info("所有人格已重载")

    def list_personas(self) -> list[str]:
        if not self.config_dir.exists():
            return []
        return sorted(p.stem for p in self.config_dir.glob("*.yaml"))
