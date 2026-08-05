"""桥接契约：TS 插件 <-> Python 服务的请求/响应 schema。

字段对齐 OpenClaw hook 的 event 与 ctx（见 _ref/openclaw/src/plugins/hook-types.ts）。
TS 插件把 OpenClaw 触发的 hook 打包成 HookRequest POST 到 /hook，
Python 返回对应 hook 的 Result（字典形式，字段因 hook 而异）。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ---- 已支持的 hook 名 ----
SUPPORTED_HOOKS: tuple[str, ...] = (
    "before_prompt_build",
    "agent_turn_prepare",
    "heartbeat_prompt_contribution",
    "llm_output",
    "before_agent_reply",
)


class HookContext(BaseModel):
    """从 OpenClaw 的 PluginHookAgentContext 提取的关键字段。"""

    agent_id: str | None = None
    session_key: str | None = None
    session_id: str | None = None
    channel: str | None = None  # 如 "wechat" / "qq" / "telegram"
    chat_id: str | None = None
    sender_id: str | None = None
    sender_is_owner: bool | None = None
    trigger: str | None = None  # cron | heartbeat | user
    model_id: str | None = None
    model_provider_id: str | None = None


class HookRequest(BaseModel):
    """统一的 hook 转发请求体。"""

    hook: str
    event: dict = Field(default_factory=dict)  # hook 原始 event 字段
    ctx: HookContext = Field(default_factory=HookContext)
