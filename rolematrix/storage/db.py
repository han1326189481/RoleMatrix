"""SQLite 持久化（aiosqlite）。

P0 用于情绪状态持久化：服务重启后情绪不丢失。
P1 扩展记忆存储：session / daily / long / profile 四层。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from ..config import get_settings
from ..logger import get_logger

log = get_logger("storage.db")

# 情绪表：每 session 一行，存 7 维 + updated_at
_EMOTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS emotion_state (
    session_key  TEXT PRIMARY KEY,
    happy        INTEGER NOT NULL,
    sad          INTEGER NOT NULL,
    tired        INTEGER NOT NULL,
    angry        INTEGER NOT NULL,
    shy          INTEGER NOT NULL,
    worried      INTEGER NOT NULL,
    want_chat    INTEGER NOT NULL,
    updated_at   TEXT NOT NULL
)
"""

# 记忆表：四层记忆存储（session/daily/long/profile）
_MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key  TEXT NOT NULL,
    layer        TEXT NOT NULL,
    role         TEXT,
    content      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    metadata     TEXT
)
"""

# 索引：按 session + layer 查询（最常用路径）
_MEMORY_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_memory_session_layer ON memory_items(session_key, layer)",
    "CREATE INDEX IF NOT EXISTS idx_memory_created_at ON memory_items(created_at DESC)",
]

_DB_PATH: str | None = None


def _resolve_db_path() -> str:
    global _DB_PATH
    if _DB_PATH is None:
        path = get_settings().storage.sqlite_path
        if not Path(path).is_absolute():
            path = str(get_settings().project_root / path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _DB_PATH = path
    return _DB_PATH


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


async def init_db() -> None:
    """初始化表结构（幂等）。在 server lifespan 启动时调用。"""
    async with aiosqlite.connect(_resolve_db_path()) as db:
        await db.execute(_EMOTION_SCHEMA)
        await db.execute(_MEMORY_SCHEMA)
        for idx in _MEMORY_INDEXES:
            await db.execute(idx)
        await db.commit()
    log.info("SQLite 已初始化: %s", _resolve_db_path())


async def load_all_emotion_states() -> list[tuple[str, dict[str, Any]]]:
    """启动时加载所有 session 的情绪状态。"""
    async with aiosqlite.connect(_resolve_db_path()) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM emotion_state")
        rows = await cursor.fetchall()
    result: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        d = dict(row)
        session_key = d.pop("session_key")
        result.append((session_key, d))
    return result


async def upsert_emotion_state(session_key: str, state: dict[str, Any]) -> None:
    """写入/更新一个 session 的情绪状态。"""
    async with aiosqlite.connect(_resolve_db_path()) as db:
        await db.execute(
            """
            INSERT INTO emotion_state
                (session_key, happy, sad, tired, angry, shy, worried, want_chat, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_key) DO UPDATE SET
                happy=excluded.happy, sad=excluded.sad, tired=excluded.tired,
                angry=excluded.angry, shy=excluded.shy, worried=excluded.worried,
                want_chat=excluded.want_chat, updated_at=excluded.updated_at
            """,
            (
                session_key,
                state["happy"], state["sad"], state["tired"], state["angry"],
                state["shy"], state["worried"], state["want_chat"],
                state["updated_at"],
            ),
        )
        await db.commit()


# ============================================================
# 记忆表 CRUD
# ============================================================

async def insert_memory_item(
    session_key: str,
    layer: str,
    content: str,
    role: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    """插入一条记忆。

    Args:
        session_key: 会话标识
        layer: 'session' | 'daily' | 'long' | 'profile'
        content: 消息原文或摘要
        role: 'user' | 'assistant' | 'summary' | 'fact'（可选）
        metadata: 附加元数据（情绪标签、权重等），JSON 序列化存储

    Returns:
        新插入行的 id
    """
    async with aiosqlite.connect(_resolve_db_path()) as db:
        cursor = await db.execute(
            """
            INSERT INTO memory_items (session_key, layer, role, content, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_key,
                layer,
                role,
                content,
                _now_iso(),
                json.dumps(metadata, ensure_ascii=False) if metadata else None,
            ),
        )
        await db.commit()
        return cursor.lastrowid or 0


async def query_session_memory(
    session_key: str, limit: int = 20
) -> list[dict[str, Any]]:
    """查询一个 session 的会话级记忆（最近 N 条，按时间正序）。"""
    async with aiosqlite.connect(_resolve_db_path()) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT * FROM memory_items
            WHERE session_key = ? AND layer = 'session'
            ORDER BY created_at DESC LIMIT ?
            """,
            (session_key, limit),
        )
        rows = await cursor.fetchall()
    # 反转为时间正序（旧→新），便于注入到 prompt
    result = [dict(row) for row in reversed(rows)]
    for r in result:
        if r.get("metadata"):
            try:
                r["metadata"] = json.loads(r["metadata"])
            except json.JSONDecodeError:
                pass
    return result


async def query_long_memory(
    session_key: str, limit: int = 5
) -> list[dict[str, Any]]:
    """查询长期记忆（daily/long 层），按时间倒序。"""
    async with aiosqlite.connect(_resolve_db_path()) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT * FROM memory_items
            WHERE session_key = ? AND layer IN ('daily', 'long')
            ORDER BY created_at DESC LIMIT ?
            """,
            (session_key, limit),
        )
        rows = await cursor.fetchall()
    result = [dict(row) for row in rows]
    for r in result:
        if r.get("metadata"):
            try:
                r["metadata"] = json.loads(r["metadata"])
            except json.JSONDecodeError:
                pass
    return result


async def count_memory_items(session_key: str | None = None) -> int:
    """统计记忆条数（可选按 session 过滤）。"""
    async with aiosqlite.connect(_resolve_db_path()) as db:
        if session_key:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM memory_items WHERE session_key = ?",
                (session_key,),
            )
        else:
            cursor = await db.execute("SELECT COUNT(*) FROM memory_items")
        row = await cursor.fetchone()
        return row[0] if row else 0
