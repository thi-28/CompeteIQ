"""
FastAPI layer over CompeteIQ's SQLite store, plus a trigger for pipeline runs.

POST /run starts the pipeline in the background and returns the run_id that the
pipeline will actually use, so clients can poll GET /runs/{run_id}. Only one
run at a time is allowed (409 otherwise): concurrent runs would double API
spend and race on the vector store.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from typing import Any, Callable, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text

from config import get_settings
from memory.semantic import SemanticMemory
from models.schemas import RunResult, SignalType

logger = logging.getLogger(__name__)

Runner = Callable[[list[str] | None, str], RunResult]


# --- response models -----------------------------------------------------

class RunRequest(BaseModel):
    competitors: list[str] | None = Field(default=None, description="Defaults to DEFAULT_COMPETITORS.")


class RunAccepted(BaseModel):
    run_id: str
    status: Literal["started"] = "started"
    competitors: list[str]


class RunOut(BaseModel):
    run_id: str
    status: Literal["running", "succeeded", "failed"]
    started_at: datetime | None = None
    completed_at: datetime | None = None
    competitors: list[str] = []
    signal_count: int = 0
    error_count: int = 0
    research_retries: dict[str, int] = {}
    briefing_id: str | None = None


class SignalOut(BaseModel):
    id: int
    run_id: str
    competitor: str
    signal_type: str
    title: str
    summary: str
    impact_assessment: str
    source_url: str
    date_detected: datetime
    confidence: float


class BriefingSummary(BaseModel):
    run_id: str
    created_at: datetime
    competitors_monitored: list[str]
    signal_count: int


class BriefingOut(BriefingSummary):
    content: str


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    openai_model: str
    embedding_model: str
    openai_key_configured: bool
    tavily_key_configured: bool
    active_run: str | None


# --- run tracking ---------------------------------------------------------

class RunTracker:
    """Single-flight guard for pipeline runs (one API process)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active: tuple[str, list[str]] | None = None

    def try_start(self, run_id: str, competitors: list[str]) -> bool:
        with self._lock:
            if self.active is not None:
                return False
            self.active = (run_id, competitors)
            return True

    def finish(self) -> None:
        with self._lock:
            self.active = None


def _default_runner(competitors: list[str] | None, run_id: str) -> RunResult:
    from main import run_pipeline  # deferred: heavy imports only when a run happens

    return run_pipeline(competitors, run_id=run_id)


def _run_out(row: dict[str, Any]) -> RunOut:
    return RunOut(**{k: v for k, v in row.items() if k != "success"},
                  status="succeeded" if row["success"] else "failed")


# --- app factory ------------------------------------------------------------

def create_app(semantic: SemanticMemory | None = None, runner: Runner | None = None) -> FastAPI:
    settings = get_settings()
    store = semantic or SemanticMemory(settings.sqlite_db_path)
    run = runner or _default_runner
    tracker = RunTracker()

    app = FastAPI(title="CompeteIQ API", version="1.0.0",
                  description="Competitive-intelligence signals, briefings, and pipeline runs.")
    app.state.tracker = tracker

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    def execute(run_id: str, competitors: list[str]) -> None:
        try:
            run(competitors, run_id)
        except Exception:  # run_pipeline records its own failures; this is a last resort
            logger.exception("Background run %s crashed", run_id)
        finally:
            tracker.finish()

    @app.post("/run", status_code=202, response_model=RunAccepted, dependencies=[Depends(require_api_key)])
    def start_run(background: BackgroundTasks, body: RunRequest | None = None) -> RunAccepted:
        from main import new_run_id

        competitors = [c.strip() for c in (body.competitors if body and body.competitors else settings.default_competitors) if c.strip()]
        if not competitors:
            raise HTTPException(status_code=422, detail="competitors must not be empty")
        run_id = new_run_id()
        if not tracker.try_start(run_id, competitors):
            raise HTTPException(status_code=409, detail=f"Run {tracker.active[0]} is already in progress")
        background.add_task(execute, run_id, competitors)
        return RunAccepted(run_id=run_id, competitors=competitors)

    @app.get("/runs", response_model=list[RunOut])
    def list_runs(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)) -> list[RunOut]:
        return [_run_out(r) for r in store.get_runs(limit=limit, offset=offset)]

    @app.get("/runs/{run_id}", response_model=RunOut)
    def get_run(run_id: str) -> RunOut:
        row = store.get_run(run_id)
        if row:
            return _run_out(row)
        if tracker.active and tracker.active[0] == run_id:
            return RunOut(run_id=run_id, status="running", competitors=tracker.active[1])
        raise HTTPException(status_code=404, detail="Run not found")

    @app.get("/briefings", response_model=list[BriefingSummary])
    def list_briefings(limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0)) -> list[dict]:
        return store.get_briefings(limit=limit, offset=offset)

    @app.get("/briefings/latest", response_model=BriefingOut)
    def latest_briefing() -> dict:
        row = store.get_latest_briefing()
        if not row:
            raise HTTPException(status_code=404, detail="No briefings yet")
        return row

    @app.get("/briefings/{run_id}", response_model=BriefingOut)
    def get_briefing(run_id: str) -> dict:
        row = store.get_briefing(run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Briefing not found")
        return row

    @app.get("/signals", response_model=list[SignalOut])
    def list_signals(
        competitor: str | None = None,
        signal_type: SignalType | None = None,
        run_id: str | None = None,
        since: datetime | None = Query(None, description="ISO 8601 with timezone, e.g. 2026-11-01T00:00:00Z"),
        limit: int = Query(200, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> list[dict]:
        if since is not None and since.tzinfo is None:
            raise HTTPException(status_code=422, detail="since must include a timezone (e.g. trailing Z)")
        return store.get_signals(competitor=competitor, signal_type=signal_type, run_id=run_id,
                                 since=since, limit=limit, offset=offset)

    @app.get("/health", response_model=Health)
    def health() -> Health:
        try:
            with store.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_ok = True
        except Exception:
            db_ok = False
        return Health(
            status="ok" if db_ok else "degraded",
            database=db_ok,
            openai_model=settings.openai_model,
            embedding_model=settings.embedding_model,
            openai_key_configured=bool(os.getenv("OPENAI_API_KEY")),
            tavily_key_configured=bool(os.getenv("TAVILY_API_KEY")),
            active_run=tracker.active[0] if tracker.active else None,
        )

    return app


_app: FastAPI | None = None


def __getattr__(name: str) -> Any:
    # `uvicorn api.main:app` builds the real app lazily (once), so importing this
    # module in tests doesn't touch the production database.
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
