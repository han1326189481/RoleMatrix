"""人格注册表：agentId -> persona 映射。"""
from __future__ import annotations

from .loader import PersonaLoader
from .models import Persona


class PersonaRegistry:
    """维护 OpenClaw agent 与 RoleMatrix 人格的绑定关系。"""

    def __init__(self, loader: PersonaLoader, default_name: str) -> None:
        self._loader = loader
        self._default = default_name
        self._mapping: dict[str, str] = {}

    def assign(self, agent_id: str, persona_name: str) -> None:
        self._mapping[agent_id] = persona_name

    def get(self, agent_id: str | None) -> Persona:
        name = self._mapping.get(agent_id or "", self._default)
        return self._loader.load(name)

    def persona_name_for(self, agent_id: str | None) -> str:
        return self._mapping.get(agent_id or "", self._default)
