"""配置加载：环境变量 + YAML。

优先级：环境变量 ROLEMATRIX_CONFIG 指向的 YAML > 默认值。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


class StorageConfig(BaseModel):
    sqlite_path: str = "data/rolematrix.db"


class PersonaConfig(BaseModel):
    config_dir: str = "personas"
    default_persona: str = "default"


class EmotionInitialConfig(BaseModel):
    happy: int = 50
    sad: int = 0
    tired: int = 20
    angry: int = 0
    shy: int = 10
    worried: int = 0
    want_chat: int = 60


class EmotionConfig(BaseModel):
    decay_halflife_hours: int = 24
    initial: EmotionInitialConfig = Field(default_factory=EmotionInitialConfig)


class LoggingConfig(BaseModel):
    level: str = "INFO"


class LLMConfig(BaseModel):
    # deepseek | ollama
    provider: str = "ollama"
    # 云端模型（provider=deepseek 时使用）
    cloud_model: str = "deepseek-v4-flash"
    # 本地文本模型（provider=ollama 时使用）
    local_model: str = "qwen2.5:7b"
    # 本地视觉模型（图片输入时使用，始终本地）
    local_vision_model: str = "minicpm-v:latest"
    # 调用模式：single（单层，按 provider 走）| dual（双层：brain LoRA → mouth provider）
    mode: str = "single"
    # 大脑 LoRA 配置（mode=dual 时使用）
    brain_base_model: str = r"D:\RoleMatrix\models\base\Qwen2.5-7B-Instruct-bnb-4bit"
    brain_lora_path: str = r"D:\RoleMatrix\models\lora_brain_v1"


class SearchConfig(BaseModel):
    """Web search 配置（小R 联网搜索能力）"""
    # 是否启用 web search
    enabled: bool = True
    # 搜索 provider: ddgs (DuckDuckGo, 无 key)
    provider: str = "ddgs"
    # 超时秒数
    timeout_sec: float = 5.0
    # 每 session 每分钟搜索次数上限
    rate_limit_per_min: int = 2
    # query 缓存 TTL（秒），相同 query 1h 内不重复搜索
    cache_ttl_sec: int = 3600
    # 返回结果数
    max_results: int = 3


class CollectionConfig(BaseModel):
    """小R 私人收藏库配置（图片/表情包/搜索结果）"""
    # 收藏库根目录（独立于主记忆库）
    root_dir: str = r"D:\RoleMatrix\.xiaor_collection"
    # 图片存储子目录
    images_subdir: str = "images"
    # 收藏库 SQLite 文件名
    db_filename: str = "collection.db"
    # 允许的图片扩展名
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".gif", ".webp"]
    )
    # 单张图片大小上限（字节），默认 10MB
    max_image_size: int = 10 * 1024 * 1024


class Settings(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    collection: CollectionConfig = Field(default_factory=CollectionConfig)

    @property
    def project_root(self) -> Path:
        root = os.getenv("ROLEMATRIX_ROOT")
        return Path(root).resolve() if root else Path.cwd().resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _resolve_config_path() -> Path:
    config_path_env = os.getenv("ROLEMATRIX_CONFIG", "config/config.yaml")
    config_path = Path(config_path_env)
    if not config_path.is_absolute():
        root = os.getenv("ROLEMATRIX_ROOT")
        base = Path(root).resolve() if root else Path.cwd().resolve()
        config_path = base / config_path
    return config_path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """加载配置（带缓存）。"""
    data = _load_yaml(_resolve_config_path())
    return Settings.model_validate(data)


def reload_settings() -> Settings:
    """清除缓存，重新加载配置（用于热加载）。"""
    get_settings.cache_clear()
    return get_settings()
