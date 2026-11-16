"""
Central configuration for CompeteIQ.

Every tunable value (model names, thresholds, caps, paths) is read here and
nowhere else, so the code default, .env.example and README can't drift apart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# Single source of truth for model defaults. .env.example and README must match.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    openai_model: str = DEFAULT_OPENAI_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    default_competitors: list[str] = field(
        default_factory=lambda: ["Anthropic", "Google", "Meta", "Mistral", "Perplexity"]
    )
    max_snippets_per_competitor: int = 10
    confidence_threshold: float = 0.6
    min_signals_per_competitor: int = 2
    max_research_retries: int = 2
    dedup_similarity_threshold: float = 0.92
    memory_top_k: int = 5
    data_dir: str = "./data"
    sqlite_db_path: str = "./data/competeiq.db"
    chroma_persist_dir: str = "./data/chroma"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables, falling back to defaults."""
        d = cls()
        env = os.getenv
        competitors = env("DEFAULT_COMPETITORS")
        return cls(
            openai_model=env("OPENAI_MODEL", d.openai_model),
            embedding_model=env("EMBEDDING_MODEL", d.embedding_model),
            default_competitors=_split_csv(competitors) if competitors else d.default_competitors,
            max_snippets_per_competitor=int(
                env("MAX_SNIPPETS_PER_COMPETITOR", d.max_snippets_per_competitor)
            ),
            confidence_threshold=float(env("CONFIDENCE_THRESHOLD", d.confidence_threshold)),
            min_signals_per_competitor=int(
                env("MIN_SIGNALS_PER_COMPETITOR", d.min_signals_per_competitor)
            ),
            max_research_retries=int(env("MAX_RESEARCH_RETRIES", d.max_research_retries)),
            dedup_similarity_threshold=float(
                env("DEDUP_SIMILARITY_THRESHOLD", d.dedup_similarity_threshold)
            ),
            memory_top_k=int(env("MEMORY_TOP_K", d.memory_top_k)),
            data_dir=env("DATA_DIR", d.data_dir),
            sqlite_db_path=env("SQLITE_DB_PATH", d.sqlite_db_path),
            chroma_persist_dir=env("CHROMA_PERSIST_DIR", d.chroma_persist_dir),
            log_level=env("LOG_LEVEL", d.log_level),
        )


def get_settings() -> Settings:
    """Return fresh settings (re-read each call so tests can monkeypatch env)."""
    return Settings.from_env()
