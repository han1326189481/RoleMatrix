"""DeepSeek-V4-Flash 云端 LLM Provider。

API 兼容 OpenAI 格式，端点：https://api.deepseek.com/v1/chat/completions
模型：deepseek-v4-flash（284B MoE，13B 激活，1M 上下文，0.28元/百万token）

关键优势：
- 1M 上下文：可传全量记忆+人设，无需复杂压缩
- 价格极低：每轮对话约 0.001 元
- Agent 能力强：角色演绎效果远超本地 7B
"""
from __future__ import annotations

import os

import httpx

from ..logger import get_logger

log = get_logger("llm.deepseek")

API_BASE = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_TIMEOUT = 60.0


class DeepSeekProvider:
    """DeepSeek 云端 provider。

    API key 从环境变量 DEEPSEEK_API_KEY 读取。
    """

    @property
    def default_model(self) -> str:
        return DEFAULT_MODEL

    @property
    def _api_key(self) -> str:
        key = os.getenv("DEEPSEEK_API_KEY", "")
        if not key:
            raise RuntimeError(
                "未设置 DEEPSEEK_API_KEY 环境变量。"
                "请在环境变量中配置你的 DeepSeek API key。"
            )
        return key

    async def chat(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        image_base64: str | None = None,
    ) -> str:
        """调用 DeepSeek API 生成回复。

        DeepSeek 当前不直接支持图片输入（V4-Flash 是文本模型），
        若有图片会回退到本地视觉模型（调用方处理）。
        """
        # DeepSeek-V4-Flash 不支持图片，调用方应自行路由到 Ollama vision
        if image_base64:
            raise ValueError(
                "DeepSeek-V4-Flash 不支持图片输入，请用本地 minicpm-v 模型处理图片"
            )

        # 组装 OpenAI 兼容格式
        full_messages: list[dict[str, str]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "model": model or self.default_model,
            "messages": full_messages,
            "stream": False,
            "temperature": 0.8,  # 角色演绎稍高温度更生动
            "max_tokens": 512,  # 闲聊不需要长回复，省 token
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        log.info(
            "调用 DeepSeek model=%s messages=%d 总字符=%d",
            payload["model"],
            len(full_messages),
            sum(len(m["content"]) for m in full_messages),
        )

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            try:
                resp = await client.post(
                    f"{API_BASE}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                log.info(
                    "DeepSeek 回复 %d 字, tokens: prompt=%s completion=%s",
                    len(content),
                    usage.get("prompt_tokens", "?"),
                    usage.get("completion_tokens", "?"),
                )
                return content
            except httpx.HTTPStatusError as e:
                err_text = e.response.text[:300]
                log.error("DeepSeek HTTP %d: %s", e.response.status_code, err_text)
                raise RuntimeError(
                    f"DeepSeek API 调用失败 (HTTP {e.response.status_code}): {err_text}"
                ) from e
            except httpx.RequestError as e:
                log.error("DeepSeek 不可达: %s", e)
                raise RuntimeError(f"DeepSeek API 不可达: {e}") from e
