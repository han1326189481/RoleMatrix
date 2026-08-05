"""桥接层：TS 插件 <-> Python Runtime 的 HTTP 契约与分发。"""
from .contracts import SUPPORTED_HOOKS, HookContext, HookRequest
from .dispatcher import Runtime, dispatch, get_runtime
from .server import create_app

__all__ = [
    "SUPPORTED_HOOKS",
    "HookRequest",
    "HookContext",
    "create_app",
    "dispatch",
    "get_runtime",
    "Runtime",
]
