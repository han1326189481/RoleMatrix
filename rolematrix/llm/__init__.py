"""LLM Provider 抽象层：支持本地 Ollama 和云端 DeepSeek 切换。

设计目标：
1. 统一接口：chat.py 不关心底层是 Ollama 还是 DeepSeek
2. 配置驱动：config.yaml 的 llm.provider 决定用哪个
3. 可扩展：新增 provider 只需实现 LLMProvider 接口
"""
from __future__ import annotations

from typing import Any, Protocol


class LLMProvider(Protocol):
    """LLM 提供方接口。"""

    async def chat(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        image_base64: str | None = None,
    ) -> str:
        """调用 LLM 生成回复。

        Args:
            model: 模型名（如 "deepseek-v4-flash" 或 "qwen2.5:7b"）
            system_prompt: 系统 prompt（含人设+情绪）
            messages: 对话历史 [{"role","content"}, ...]
            image_base64: 可选的图片（视觉模型）

        Returns:
            assistant 的文本回复
        """
        ...

    @property
    def default_model(self) -> str:
        """该 provider 的默认模型。"""
        ...


def get_provider(provider_name: str) -> LLMProvider:
    """工厂方法：根据名称返回 provider 实例。"""
    if provider_name == "deepseek":
        from .deepseek import DeepSeekProvider
        return DeepSeekProvider()
    if provider_name == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider()
    raise ValueError(f"未知 provider: {provider_name}（支持: deepseek, ollama）")
