"""
Pydantic v2 domain models and the LangGraph shared state for CompeteIQ.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

SignalType = Literal[
    "product_launch",
    "pricing_change",
    "partnership",
    "research_release",
    "executive_move",
    "market_expansion",
    "other",
]

ImpactLevel = Literal["low", "medium", "high"]


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp (replaces deprecated datetime.utcnow)."""
    return datetime.now(timezone.utc)


class Snippet(BaseModel):
    """One search result. Keeping the URL attached to the text is what makes
    source grounding possible downstream."""

    competitor: str
    url: str
    title: str = ""
    content: str
    query: str = ""


class Signal(BaseModel):
    """A structured competitive-intelligence signal extracted from snippets."""

    competitor: str = Field(..., min_length=1)
    signal_type: SignalType
    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    impact_assessment: ImpactLevel
    source_url: str
    date_detected: datetime = Field(default_factory=utc_now)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("source_url")
    @classmethod
    def require_http_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("source_url must be an http(s) URL")
        return v

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 2)


class Briefing(BaseModel):
    """A synthesized markdown briefing for one pipeline run."""

    run_id: str
    created_at: datetime = Field(default_factory=utc_now)
    competitors_monitored: list[str]
    signal_count: int = Field(..., ge=0)
    content: str


class RunResult(BaseModel):
    """Summary record persisted after each pipeline run."""

    run_id: str
    started_at: datetime
    completed_at: datetime = Field(default_factory=utc_now)
    competitors: list[str]
    signal_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    research_retries: dict[str, int] = Field(
        default_factory=dict,
        description="Re-search attempts per competitor triggered by the supervisor.",
    )
    briefing_id: str | None = None
    success: bool = True


class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes. Nodes return partial updates."""

    # Input / run metadata
    run_id: str
    started_at: str
    competitors: list[str]

    # Supervisor
    search_queries: dict[str, list[str]]
    retry_counts: dict[str, int]
    """Re-search attempts per competitor; supervisor stops at MAX_RESEARCH_RETRIES."""
    low_signal_competitors: list[str]
    """Competitors flagged for re-collection after extraction (< MIN_SIGNALS)."""

    # Collector
    snippets: dict[str, list[dict[str, Any]]]
    """Serialized Snippet dicts keyed by competitor."""

    # Episodic memory (retrieved per competitor)
    memory_context: dict[str, list[str]]

    # Extractor
    extracted_signals: list[dict[str, Any]]

    # Synthesizer
    briefing: str

    # Observability: reducer appends, so nodes return only *new* errors.
    errors: Annotated[list[str], operator.add]
