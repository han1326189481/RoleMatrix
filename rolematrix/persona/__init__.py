"""人格引擎：YAML 配置热加载 + agent 绑定。"""
from .loader import PersonaLoader
from .models import Persona, PersonaCapabilities, PersonaReply, PersonaStyle
from .registry import PersonaRegistry

__all__ = [
    "PersonaLoader",
    "PersonaRegistry",
    "Persona",
    "PersonaStyle",
    "PersonaCapabilities",
    "PersonaReply",
]
