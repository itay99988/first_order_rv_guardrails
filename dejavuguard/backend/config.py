"""
Application configuration loaded from environment variables and .env file.

Settings can be overridden via the Settings UI (stored in SQLite).
These env-based defaults are used on first launch before the user configures anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    """Load .env file from project root if it exists."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


_load_env()


@dataclass(frozen=True)
class Config:
    """Application configuration with environment variable defaults.

    Attributes:
        openrouter_api_key: OpenRouter API key for the chat model.
        openrouter_model: Default chat model identifier.
        grounding_provider: Local LLM provider (ollama, lmstudio, vllm, custom).
        grounding_base_url: Local LLM server base URL.
        grounding_model: Model name on the local LLM server.
        database_path: Path to the SQLite database file.
        host: Server bind host.
        port: Server bind port.
        dejavu_url: Base URL of the DejaVu runtime verification server.
    """

    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "")
    )
    openrouter_model_custom: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL_CUSTOM", "")
    )
    grounding_provider: str = field(
        default_factory=lambda: os.getenv("GROUNDING_PROVIDER", "ollama")
    )
    grounding_base_url: str = field(
        default_factory=lambda: os.getenv("GROUNDING_BASE_URL", "http://localhost:11434")
    )
    grounding_model: str = field(default_factory=lambda: os.getenv("GROUNDING_MODEL", "mistral"))
    grounding_api_key: str = field(default_factory=lambda: os.getenv("GROUNDING_API_KEY", ""))
    database_path: str = field(
        default_factory=lambda: os.getenv("DATABASE_PATH", "./dejavuguard.db")
    )
    host: str = field(
        default_factory=lambda: os.getenv("HOST", "0.0.0.0")  # noqa: S104
    )
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    dejavu_url: str = field(
        default_factory=lambda: os.getenv("DEJAVU_URL", "http://localhost:8080")
    )


def get_config() -> Config:
    """Create a Config instance from current environment."""
    return Config()
