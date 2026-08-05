"""Ollama 本地 LLM Provider。

将 chat.py 中原有的 Ollama 调用逻辑迁移到这里，保持接口统一。
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from ..logger import get_logger

log = get_logger("llm.ollama")

DEFAULT_TEXT_MODEL = "qwen2.5:7b"
DEFAULT_VISION_MODEL = "minicpm-v:latest"
DEFAULT_TIMEOUT = 120.0


class OllamaProvider:
    """Ollama 本地 provider。"""

    @property
    def default_model(self) -> str:
        return DEFAULT_TEXT_MODEL

    @property
    def _base_url(self) -> str:
        return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

    async def chat(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        image_base64: str | None = None,
    ) -> str:
        """调用 Ollama /api/chat 接口。"""
        full_messages: list[dict[str, Any]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        # 历史消息
        for m in messages:
            full_messages.append({"role": m["role"], "content": m["content"]})

        # 当前消息（可能有图片）
        # Ollama /api/chat 标准：images 字段与 role/content 平级，值为 base64 字符串数组
        # 参考：https://github.com/ollama/ollama/blob/main/docs/api.md
        if image_base64:
            raw = image_base64
            if "," in raw and raw.startswith("data:"):
                raw = raw.split(",", 1)[1]
            last = full_messages[-1] if full_messages else None
            if last and last["role"] == "user":
                # 给最后一条 user 消息附加 images 字段（content 保持字符串）
                if not last.get("content"):
                    last["content"] = "请描述这张图片"
                last["images"] = [raw]
            else:
                full_messages.append({
                    "role": "user",
                    "content": "请描述这张图片",
                    "images": [raw],
                })
            model = model or DEFAULT_VISION_MODEL
        else:
            model = model or DEFAULT_TEXT_MODEL

        payload = {
            "model": model,
            "messages": full_messages,
            "stream": False,
            "options": {"temperature": 0.7},
        }

        log.info("调用 Ollama model=%s messages=%d img=%s",
                 model, len(full_messages), bool(image_base64))

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            try:
                resp = await client.post(f"{self._base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "").strip()
            except httpx.HTTPStatusError as e:
                log.error("Ollama HTTP %d: %s", e.response.status_code, e.response.text[:200])
                raise RuntimeError(f"Ollama 调用失败: HTTP {e.response.status_code}") from e
            except httpx.RequestError as e:
                log.error("Ollama 不可达: %s", e)
                raise RuntimeError(f"Ollama 不可达: {e}") from e
