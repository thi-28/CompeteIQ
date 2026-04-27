"""
CompeteIQ FastAPI REST API.

Endpoints:
  POST /run                     — trigger the full intelligence pipeline
  GET  /briefings               — list recent briefings
  GET  /briefings/{run_id}      — fetch a specific briefing by run_id
  GET  /signals                 — list signals with optional filters
  GET  /health                  — system health check

Start the server:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

# Lazy import pipeline to avoid heavy init at module load (tests, health checks)
_pipeline_executor = ThreadPoolExecutor(max_workers=1)

app = FastAPI(
    title="CompeteIQ API",
    description=(
        "REST interface for the CompeteIQ multi-agent competitive intelligence pipeline. "
        "Trigger pipeline runs, retrieve briefings, and explore extracted signals."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """Request body for POST /run."""

    competitors: list[str] = Field(
        default_factory=list,
        description=(
            "Competitors to monitor. Leave empty to use DEFAULT_COMPETITORS env var. "
            "Example: ['Anthropic', 'Google', 'Meta']"
        ),
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Optional UUID to assign to this run. Auto-generated if omitted.",
    )


class RunResponse(BaseModel):
    """Response for POST /run."""

    run_id: str = Field(..., description="Unique identifier for this pipeline run.")
    status: str = Field(..., description="'started' — pipeline runs asynchronously.")
    message: str = Field(..., description="Human-readable status message.")


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    db_accessible: bool
    chroma_accessible: bool
    timestamp: str


# ---------------------------------------------------------------------------
# Background task runner
# ---------------------------------------------------------------------------


def _run_pipeline_task(run_id: str, competitors: list[str]) -> None:
    """
    Execute the pipeline in a background thread.

    Errors are swallowed here — the run result is persisted to SQLite by
    run_pipeline() itself, so the API caller can poll GET /briefings/{run_id}.

    Args:
        run_id: Run identifier.
        competitors: Competitor list for this run.
    """
    try:
        from main import run_pipeline  # imported here to avoid circular deps

        run_pipeline(competitors=competitors if competitors else None, run_id=run_id)
    except Exception as exc:
        logger.error("Background pipeline run %s failed: %s", run_id, exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/run",
    response_model=RunResponse,
    status_code=202,
    summary="Trigger a pipeline run",
    tags=["pipeline"],
)
def trigger_run(
    request: RunRequest,
    background_tasks: BackgroundTasks,
) -> RunResponse:
    """
    Enqueue a new competitive intelligence pipeline run.

    The pipeline runs asynchronously in a background thread. Poll
    ``GET /briefings/{run_id}`` to check when results are available.

    - **competitors**: Optional list of competitors to monitor.
    - **run_id**: Optional UUID. Auto-generated if not provided.

    Returns the ``run_id`` immediately so the caller can track progress.
    """
    run_id = request.run_id or str(uuid.uuid4())
    competitors = request.competitors or []

    background_tasks.add_task(_run_pipeline_task, run_id, competitors)

    logger.info("Pipeline run enqueued — run_id=%s", run_id)
    return RunResponse(
        run_id=run_id,
        status="started",
        message=f"Pipeline run {run_id} enqueued. Poll GET /briefings/{run_id} for results.",
    )


@app.get(
    "/briefings",
    summary="List recent briefings",
    tags=["briefings"],
)
def list_briefings(
    limit: int = Query(default=10, ge=1, le=100, description="Max briefings to return"),
) -> list[dict]:
    """
    Return the most recent intelligence briefings, ordered newest-first.

    - **limit**: Maximum number of briefings to return (1–100).
    """
    from memory.semantic import SemanticMemory

    db = SemanticMemory()
    return db.get_briefings(limit=limit)


@app.get(
    "/briefings/{run_id}",
    summary="Get a specific briefing",
    tags=["briefings"],
)
def get_briefing(run_id: str) -> dict:
    """
    Retrieve the full briefing for a specific pipeline run.

    - **run_id**: The UUID of the pipeline run.

    Returns 404 if the run_id does not exist or if the run has not yet
    completed.
    """
    from memory.semantic import SemanticMemory

    db = SemanticMemory()
    briefing = db.get_briefing(run_id)
    if briefing is None:
        raise HTTPException(
            status_code=404,
            detail=f"Briefing for run_id '{run_id}' not found.",
        )
    return briefing


@app.get(
    "/signals",
    summary="List extracted signals",
    tags=["signals"],
)
def list_signals(
    competitor: Optional[str] = Query(
        default=None, description="Filter by competitor name (partial match)"
    ),
    signal_type: Optional[str] = Query(
        default=None,
        description=(
            "Filter by signal type. One of: product_launch, pricing_change, "
            "partnership, research_release, executive_move, market_expansion, other"
        ),
    ),
    limit: int = Query(default=100, ge=1, le=500, description="Max signals to return"),
) -> list[dict]:
    """
    Return extracted intelligence signals with optional filtering.

    - **competitor**: Case-insensitive partial match on competitor name.
    - **signal_type**: Exact match on signal_type enum value.
    - **limit**: Maximum number of signals to return (1–500).

    Results are ordered by date_detected descending (newest first).
    """
    from memory.semantic import SemanticMemory

    db = SemanticMemory()
    return db.get_signals(competitor=competitor, signal_type=signal_type, limit=limit)


@app.get(
    "/runs",
    summary="List pipeline run history",
    tags=["pipeline"],
)
def list_runs(
    limit: int = Query(default=50, ge=1, le=200, description="Max runs to return"),
) -> list[dict]:
    """
    Return pipeline run history ordered by start time descending.

    - **limit**: Maximum number of run records to return (1–200).
    """
    from memory.semantic import SemanticMemory

    db = SemanticMemory()
    return db.get_runs(limit=limit)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    tags=["system"],
)
def health_check() -> HealthResponse:
    """
    Return the current health status of the CompeteIQ system.

    Checks:
    - SQLite database accessibility
    - ChromaDB collection accessibility
    """
    db_ok = False
    chroma_ok = False

    try:
        from memory.semantic import SemanticMemory

        SemanticMemory().get_runs(limit=1)
        db_ok = True
    except Exception as exc:
        logger.warning("Health: SQLite unavailable — %s", exc)

    try:
        from memory.episodic import EpisodicMemory

        EpisodicMemory().count()
        chroma_ok = True
    except Exception as exc:
        logger.warning("Health: ChromaDB unavailable — %s", exc)

    overall = "healthy" if (db_ok and chroma_ok) else "degraded"
    return HealthResponse(
        status=overall,
        db_accessible=db_ok,
        chroma_accessible=chroma_ok,
        timestamp=datetime.utcnow().isoformat(),
    )
