"""
Pydantic v2 data models and LangGraph shared state for CompeteIQ.

All data flowing through the multi-agent pipeline is typed here.
AgentState is the central TypedDict that every agent reads from and writes to.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------


class Signal(BaseModel):
    """A single structured competitive intelligence signal extracted from raw web data."""

    competitor: str = Field(
        ..., description="Name of the competitor this signal pertains to."
    )
    signal_type: SignalType = Field(
        ..., description="Category of the intelligence signal."
    )
    title: str = Field(..., description="Short, descriptive title for the signal.")
    summary: str = Field(
        ..., description="Concise summary of what happened and why it matters."
    )
    impact_assessment: ImpactLevel = Field(
        ...,
        description="Estimated competitive impact level: low, medium, or high.",
    )
    source_url: str = Field(
        ..., description="URL of the primary source for this signal."
    )
    date_detected: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the signal was first detected.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence that this is a valid, novel signal (0–1).",
    )

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        """Round confidence to two decimal places for consistent storage."""
        return round(v, 2)

    model_config = {"json_encoders": {datetime: lambda dt: dt.isoformat()}}


class Briefing(BaseModel):
    """A fully synthesized competitive intelligence briefing for a single pipeline run."""

    run_id: str = Field(..., description="Unique identifier for the pipeline run.")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the briefing was generated.",
    )
    competitors_monitored: list[str] = Field(
        ..., description="Competitors included in this run."
    )
    signal_count: int = Field(
        ..., description="Total number of unique signals extracted."
    )
    content: str = Field(
        ..., description="Full markdown text of the formatted briefing."
    )


class RunResult(BaseModel):
    """Summary record persisted to SQLite after each pipeline run."""

    run_id: str = Field(..., description="Unique run identifier (UUID).")
    started_at: datetime = Field(
        ..., description="UTC timestamp when the run began."
    )
    completed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the run completed.",
    )
    competitors: list[str] = Field(
        ..., description="Competitors monitored in this run."
    )
    signal_count: int = Field(
        default=0, description="Number of signals extracted and persisted."
    )
    error_count: int = Field(
        default=0, description="Number of non-fatal errors encountered."
    )
    briefing_id: str | None = Field(
        default=None, description="run_id of the resulting Briefing, if generated."
    )
    success: bool = Field(
        default=True, description="Whether the run completed without fatal errors."
    )


# ---------------------------------------------------------------------------
# LangGraph shared state
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """
    Central shared state object passed between every node in the LangGraph pipeline.

    All agents read from and write to this dict. LangGraph merges partial
    updates via the graph's reducer logic — agents only set the fields they touch.
    """

    # --- Input ---
    competitors: list[str]
    """Which competitors to monitor during this run."""

    # --- Supervisor outputs ---
    search_queries: dict[str, list[str]]
    """Generated search queries keyed by competitor name."""

    # --- Collector outputs ---
    raw_signals: dict[str, list[str]]
    """Raw text snippets harvested from web search, keyed by competitor."""

    # --- Memory agent outputs ---
    memory_context: list[str]
    """Top-K relevant past signal summaries retrieved from episodic memory."""

    memory_mode: Literal["read", "write"]
    """Controls whether the memory agent reads or writes on this invocation."""

    # --- Extractor outputs ---
    extracted_signals: list[dict[str, Any]]
    """Validated Signal dicts produced by the extractor (JSON-serialisable)."""

    # --- Synthesizer outputs ---
    briefing: str
    """Final markdown briefing text."""

    # --- Run metadata ---
    run_id: str
    """UUID for this pipeline run — used as the LangGraph thread_id too."""

    started_at: str
    """ISO-format UTC timestamp when the run was initiated."""

    # --- Control flow ---
    next_agent: str
    """Routing signal consumed by conditional edges in the supervisor."""

    # --- Observability ---
    errors: list[str]
    """Non-fatal errors collected across agents (logged but execution continues)."""
