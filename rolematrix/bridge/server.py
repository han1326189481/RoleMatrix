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


class MemoryNoteRequest(BaseModel):
    session_key: str
    layer: str  # daily | long | profile
    content: str
    metadata: dict | None = None


class CollectionFromUrlRequest(BaseModel):
    url: str
    tags: list[str] | None = None
    description: str | None = None
    session_key: str | None = None


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

    # ---- 四层记忆管理端点 ----
    @app.get("/memory/{session_key}")
    async def get_memory(session_key: str) -> dict:
        """查看一个 session 的四层记忆与注入文本预览。"""
        from ..memory import get_memory_manager
        mgr = get_memory_manager()
        return {
            "session": await mgr.query_session(session_key, limit=20),
            "daily": await mgr.query_daily(session_key),
            "long": await mgr.query_long(session_key),
            "profile": await mgr.query_profile(session_key),
            "context": await mgr.build_memory_context(session_key),
        }

    @app.post("/memory/note")
    async def memory_note(body: MemoryNoteRequest) -> dict:
        """显式写入一条事实/画像（daily/long/profile 层，自动去重）。"""
        from ..memory import get_memory_manager
        mgr = get_memory_manager()
        try:
            added = await mgr.add_fact(
                body.session_key, body.layer, body.content, metadata=body.metadata
            )
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "added": added}

    @app.post("/memory/consolidate/{session_key}")
    async def memory_consolidate(session_key: str) -> dict:
        """触发当日对话整理：生成 daily 摘要 + 提取 long 事实。"""
        from ..memory import get_memory_manager
        mgr = get_memory_manager()
        return await mgr.consolidate(session_key)

    # ---- 小R 私人收藏库管理端点（人工审核用）----
    @app.get("/collection/list")
    async def list_collection(limit: int = 20) -> dict:
        """列出最近收藏（按时间倒序）。"""
        from ..tools.collection_store import count_items, list_recent
        items = await list_recent(limit=limit)
        total = await count_items()
        return {"total": total, "items": items}

    @app.get("/collection/search")
    async def search_collection(tag: str, limit: int = 5) -> dict:
        """按标签搜索收藏（send_meme 的查询接口，供排查用）。"""
        from ..tools.collection_store import query_by_tag
        return {"items": await query_by_tag(tag, limit=limit)}

    @app.post("/collection/from-url")
    async def collect_from_url(body: CollectionFromUrlRequest) -> dict:
        """手动收藏一张图片 URL（小R 看到好看图时人工添加）。"""
        from ..tools.image_downloader import download_and_save
        result = await download_and_save(
            url=body.url,
            source="web_image",
            tags=body.tags,
            description=body.description,
            session_key=body.session_key,
        )
        if result:
            rel_path, file_hash = result
            return {"ok": True, "file_path": rel_path, "file_hash": file_hash}
        return {"ok": False, "error": "下载或保存失败（详见服务端日志）"}

    return app
