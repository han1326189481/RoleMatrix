"""四层记忆管理器：session / daily / long / profile。

P1 完整实现（替换原 P2 占位）：
- session: 当前会话原始消息流（按条存储，最近 N 条）
- daily:   跨会话的当日摘要（consolidate 用嘴巴 LLM 生成，失败降级为拼接）
- long:    长期事实/偏好（consolidate 提取 + 显式写入，带去重）
- profile: 用户画像（显式写入 / HTTP API，带去重）

设计原则：
1. 记忆是增强能力，任何读写失败都只记日志，绝不阻断对话主流程
2. 所有写入去重（同 session+layer+content 不重复插入）
3. consolidate 幂等：同一天只生成一次 daily 摘要
4. consolidate 的 LLM 调用有超时保护，失败降级为纯拼接摘要
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any

from ..logger import get_logger
from ..storage.db import insert_memory_item, query_memory_items, query_session_memory

log = get_logger("memory.manager")

# 允许写入的事实层
FACT_LAYERS = ("daily", "long", "profile")

DAILY_SUMMARY_PROMPT = """你是小R的记忆整理助手。下面是小R和用户今天的对话记录。

请只输出一个 JSON 对象（不要任何多余文字）：
{
  "daily_summary": "一句话概括今天聊了什么（20-40字）",
  "facts": ["值得长期记住的用户事实1", "事实2"]
}

规则：
- daily_summary 概括当天对话主题与重要内容
- facts 只提取稳定事实：用户的喜好/身份/日程/关系进展/提到的重要事情，最多 3 条
- 没有值得长期记住的事实就输出空数组 []

示例输入：
用户：我今天真的好累，数据库大作业要交了
小R：辛苦了…要不要歇一下？
用户：下周三是截止日期，我得抓紧了

示例输出：
{"daily_summary": "用户今天赶数据库大作业很累，下周三是截止日期", "facts": ["用户正在赶数据库大作业，下周三截止"]}"""


class MemoryManager:
    """四层记忆的读写入口（进程内单例）。"""

    def __init__(self) -> None:
        # (session_key, date) 防抖：同一天只 consolidate 一次
        self._consolidated_days: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------
    # session 层：当前会话原始消息
    # ------------------------------------------------------------------
    async def record_message(
        self,
        session_key: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """写入一条会话消息（空内容跳过）。"""
        content = (content or "").strip()
        if not content:
            return 0
        try:
            return await insert_memory_item(
                session_key, "session", content, role=role, metadata=metadata
            )
        except Exception as e:  # noqa: BLE001
            log.error("[记忆] session 写入失败 session=%s: %s", session_key, e)
            return 0

    async def query_session(
        self, session_key: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        try:
            return await query_session_memory(session_key, limit=limit)
        except Exception as e:  # noqa: BLE001
            log.error("[记忆] session 查询失败 session=%s: %s", session_key, e)
            return []

    # ------------------------------------------------------------------
    # 事实层（daily / long / profile）：带去重写入 + 分层查询
    # ------------------------------------------------------------------
    async def add_fact(
        self,
        session_key: str,
        layer: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """写入一条事实。同 session+layer+content 已存在则跳过。

        Returns:
            True 表示新增，False 表示重复（或内容为空/写入失败）。
        """
        if layer not in FACT_LAYERS:
            raise ValueError(f"layer 必须是 {'/'.join(FACT_LAYERS)}，收到 {layer!r}")
        content = (content or "").strip()
        if not content:
            return False
        try:
            existing = await query_memory_items(session_key, [layer], limit=200)
            if any(item.get("content") == content for item in existing):
                return False
            await insert_memory_item(
                session_key, layer, content, role="fact", metadata=metadata
            )
            log.debug("[记忆] 新增 %s 事实 session=%s: %s", layer, session_key, content)
            return True
        except Exception as e:  # noqa: BLE001
            log.error("[记忆] %s 写入失败 session=%s: %s", layer, session_key, e)
            return False

    async def query_layer(
        self, session_key: str, layer: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        try:
            return await query_memory_items(session_key, [layer], limit=limit)
        except Exception as e:  # noqa: BLE001
            log.error("[记忆] %s 查询失败 session=%s: %s", layer, session_key, e)
            return []

    async def query_long(
        self, session_key: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        return await self.query_layer(session_key, "long", limit)

    async def query_profile(
        self, session_key: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        return await self.query_layer(session_key, "profile", limit)

    async def query_daily(
        self, session_key: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        return await self.query_layer(session_key, "daily", limit)

    # ------------------------------------------------------------------
    # prompt 注入：把长期记忆组装成文本块
    # ------------------------------------------------------------------
    async def build_memory_context(self, session_key: str) -> str:
        """组装可注入 system prompt 的长期记忆文本块（无记忆返回 ""）。"""
        parts: list[str] = []
        profile = await self.query_profile(session_key, limit=8)
        if profile:
            parts.append("## 用户画像（长期记住，不要搞错）")
            parts.extend(f"- {m['content']}" for m in profile)
        long_mem = await self.query_long(session_key, limit=8)
        if long_mem:
            parts.append("## 关于用户的长期记忆")
            parts.extend(f"- {m['content']}" for m in long_mem)
        daily = await self.query_daily(session_key, limit=3)
        if daily:
            parts.append("## 最近几天的日常")
            for m in daily:
                date = (m.get("created_at") or "")[:10]
                parts.append(f"- [{date}] {m['content']}")
        if not parts:
            return ""
        return "\n\n" + "\n".join(parts)

    # ------------------------------------------------------------------
    # consolidate：当日对话 → daily 摘要 + long 事实提取
    # ------------------------------------------------------------------
    async def has_daily_summary_today(self, session_key: str) -> bool:
        """今天是否已有 daily 摘要（consolidate 幂等判断）。"""
        today = datetime.now().strftime("%Y-%m-%d")
        daily = await self.query_daily(session_key, limit=50)
        return any((m.get("created_at") or "").startswith(today) for m in daily)

    async def consolidate(
        self,
        session_key: str,
        provider: Any | None = None,
    ) -> dict[str, Any]:
        """把今天的对话压缩为 daily 摘要 + 提取 long 事实。

        Args:
            session_key: 会话标识
            provider: 嘴巴 LLM provider（测试可注入）；默认取 config 的 provider

        Returns:
            {"daily": 摘要或None, "facts": [新增事实...], "fallback": bool,
             "skipped": bool}
            任何异常都不抛出（记忆是增强能力）。
        """
        today = datetime.now().strftime("%Y-%m-%d")
        result: dict[str, Any] = {
            "daily": None, "facts": [], "fallback": False, "skipped": False,
        }

        # 幂等：今天已有摘要则跳过（含内存防抖）
        if (session_key, today) in self._consolidated_days or await self.has_daily_summary_today(session_key):
            result["skipped"] = True
            return result
        self._consolidated_days.add((session_key, today))

        session_msgs = await self.query_session(session_key, limit=100)
        if not session_msgs:
            return result

        # 过滤今天的消息；跨天且今天无新消息时用最近 10 条兜底
        today_msgs = [
            m for m in session_msgs
            if (m.get("created_at") or "").startswith(today)
        ]
        if not today_msgs:
            today_msgs = session_msgs[-10:]

        convo = "\n".join(
            f"{'用户' if m.get('role') == 'user' else '小R'}：{m['content']}"
            for m in today_msgs
        )
        if len(convo) > 3000:
            convo = convo[-3000:]

        # 尝试 LLM 生成摘要 + 提取事实
        parsed: dict[str, Any] | None = None
        try:
            if provider is None:
                from ..config import get_settings
                from ..llm import get_provider
                settings = get_settings()
                provider = get_provider(settings.llm.provider)
            reply = await asyncio.wait_for(
                provider.chat(
                    model=provider.default_model,
                    system_prompt=DAILY_SUMMARY_PROMPT,
                    messages=[{"role": "user", "content": convo}],
                ),
                timeout=20.0,
            )
            parsed = self._parse_summary(reply)
            if parsed is None:
                raise ValueError("摘要 JSON 解析失败")
        except Exception as e:  # noqa: BLE001
            log.warning("consolidate LLM 失败，降级为拼接摘要: %s", e)
            result["fallback"] = True
            parsed = {
                "daily_summary": today_msgs[-1]["content"][:80],
                "facts": [],
            }

        # 写 daily 摘要
        summary = (parsed.get("daily_summary") or "").strip()
        if summary:
            await self.add_fact(
                session_key, "daily", summary,
                metadata={"date": today, "fallback": result["fallback"]},
            )
            result["daily"] = summary

        # 写 profile 画像事实（add_fact 内部去重）
        # consolidate 提取的是用户稳定事实（喜好/身份/日程），归入画像层
        for fact in parsed.get("facts") or []:
            fact = str(fact).strip()
            if not fact:
                continue
            if await self.add_fact(
                session_key, "profile", fact,
                metadata={"source": "consolidate", "date": today},
            ):
                result["facts"].append(fact)

        log.info(
            "[记忆] consolidate session=%s fallback=%s daily=%s facts=%d",
            session_key, result["fallback"], bool(summary), len(result["facts"]),
        )
        return result

    @staticmethod
    def _parse_summary(text: str) -> dict[str, Any] | None:
        """宽容解析摘要 JSON（容忍 markdown 代码块/前后杂文字）。"""
        if not text:
            return None
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e == -1 or e <= s:
            return None
        try:
            data = json.loads(text[s : e + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        facts = data.get("facts") or []
        if isinstance(facts, str):
            facts = re.split(r"[,;；，]", facts)
        return {
            "daily_summary": str(data.get("daily_summary", "")).strip(),
            "facts": [str(f).strip() for f in facts if str(f).strip()],
        }


# ------------------------------------------------------------------
# 进程内单例
# ------------------------------------------------------------------
_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """获取记忆管理器单例。"""
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager
