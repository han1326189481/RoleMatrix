"""FastAPI Bridge Server：接收 TS 桥接插件转发的 OpenClaw hook。

同时提供简易聊天 UI（/chat-page）用于直接测试 RoleMatrix 核心功能，
不依赖 OpenClaw gateway，避免 Windows 环境的 EBADF 等问题。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .. import __version__
from ..logger import get_logger
from .chat import router as chat_router
from .contracts import SUPPORTED_HOOKS, HookRequest
from .dispatcher import dispatch, get_runtime

log = get_logger("bridge.server")

_CHAT_HTML_PATH = Path(__file__).parent / "chat.html"


class AssignRequest(BaseModel):
    agent_id: str
    persona: str


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 启动：初始化 SQLite + 恢复情绪状态
        rt = get_runtime()
        await rt.init_storage()
        log.info("运行时已就绪")

        # 初始化小R 私人收藏库（独立于主记忆库）
        try:
            from ..tools.collection_store import init_collection_db
            await init_collection_db()
            log.info("收藏库已就绪")
        except Exception as e:  # noqa: BLE001
            log.warning("收藏库初始化失败（不影响主流程）: %s", e)

        # 双层模式：预热大脑 LoRA，避免首条消息等待 5+s 加载
        from ..config import get_settings
        settings = get_settings()
        if settings.llm.mode == "dual":
            try:
                from .dual_layer import get_brain
                brain = get_brain(
                    base_model=settings.llm.brain_base_model,
                    lora_path=settings.llm.brain_lora_path,
                )
                await brain._ensure_loaded()
                log.info("大脑 LoRA 预热完成")
            except Exception as e:  # noqa: BLE001
                log.warning("大脑预热失败（将在首次调用时重试）: %s", e)

        yield
        # 关闭：无需额外清理（aiosqlite 每次操作独立连接）

    app = FastAPI(title="RoleMatrix Bridge", version=__version__, lifespan=lifespan)

    # ---- 简易聊天 UI ----
    app.include_router(chat_router)

    @app.get("/", response_class=HTMLResponse)
    async def chat_page() -> str:
        """聊天测试首页。"""
        return _CHAT_HTML_PATH.read_text(encoding="utf-8")

    # ---- Bridge hook 转发（OpenClaw → Python）----
    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/hooks")
    async def supported_hooks() -> dict:
        return {"hooks": list(SUPPORTED_HOOKS)}

    @app.post("/hook")
    async def handle_hook(req: HookRequest) -> dict:
        return await dispatch(req)

    @app.get("/personas")
    async def list_personas() -> dict:
        rt = get_runtime()
        return {"personas": rt.persona_loader.list_personas()}

    @app.post("/personas/assign")
    async def assign_persona(body: AssignRequest) -> dict:
        rt = get_runtime()
        rt.persona_registry.assign(body.agent_id, body.persona)
        log.info("绑定 agent=%s -> persona=%s", body.agent_id, body.persona)
        return {"ok": True}

    @app.get("/emotion/{session_key}")
    async def get_emotion(session_key: str) -> dict:
        rt = get_runtime()
        state = rt.emotion.apply_decay(session_key)
        return state.as_vector()

    # ---- 小R 私人收藏库管理端点（人工审核用）----
    @app.get("/collection/list")
    async def list_collection(limit: int = 20) -> dict:
        """列出最近收藏（按时间倒序）。"""
        from ..tools.collection_store import list_recent, count_items
        items = await list_recent(limit=limit)
        total = await count_items()
        return {"total": total, "items": items}

    return app
