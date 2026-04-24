from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    chat_model: str = "qwen2.5:7b-instruct"
    embed_model: str = "nomic-embed-text"
    request_timeout_s: float = 120.0


class RagConfig(BaseModel):
    persist_dir: str = ".aiblog/chroma"
    collection: str = "posts"
    top_k: int = 6
    chunk_chars: int = 2800
    chunk_overlap_chars: int = 400


class StyleConfig(BaseModel):
    language: Literal["ru", "en"] = "ru"
    voice_rules: list[str] = Field(
        default_factory=lambda: [
            "Пиши естественно и конкретно, без канцелярита.",
            "Если в контексте нет фактов/цифр — не выдумывай. Лучше попроси уточнение в тексте как TODO.",
            "Используй короткие абзацы, где уместно — списки.",
        ]
    )


class AppConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    style: StyleConfig = Field(default_factory=StyleConfig)

    posts_dir: str = "data/posts"
    out_dir: str = "out"


def default_config_path() -> Path:
    return Path("config.yaml")


def load_config(path: Optional[Path] = None) -> AppConfig:
    # Load `.env` if present. Values are then available via `os.getenv`.
    load_dotenv(override=False)

    path = path or default_config_path()
    data: dict = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    cfg = AppConfig.model_validate(data)

    posts_dir = os.getenv("AIBLOG_POSTS_DIR")
    if posts_dir and posts_dir.strip():
        cfg.posts_dir = posts_dir.strip()

    out_dir = os.getenv("AIBLOG_OUT_DIR")
    if out_dir and out_dir.strip():
        cfg.out_dir = out_dir.strip()

    # Expand "~" and normalize paths for consistent behavior.
    cfg.posts_dir = str(Path(cfg.posts_dir).expanduser())
    cfg.out_dir = str(Path(cfg.out_dir).expanduser())

    return cfg


def save_default_config(path: Optional[Path] = None) -> Path:
    path = path or default_config_path()
    cfg = AppConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="python"), sort_keys=False), encoding="utf-8")
    return path

