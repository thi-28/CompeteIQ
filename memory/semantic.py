"""
Semantic memory: structured history (signals, briefings, runs) in SQLite via
SQLAlchemy 2.0. This is the source of truth for the API and dashboard;
ChromaDB (episodic memory) is only used for similarity retrieval.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from config import get_settings
from models.schemas import Briefing, RunResult, Signal

logger = logging.getLogger(__name__)


class UTCDateTime(TypeDecorator):
    """SQLite drops tzinfo. Store naive UTC, always return aware UTC."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Naive datetimes are not allowed; use timezone-aware UTC")
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        return value.replace(tzinfo=timezone.utc) if value is not None else None


class Base(DeclarativeBase):
    pass


class RunRow(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    completed_at: Mapped[datetime] = mapped_column(UTCDateTime)
    competitors: Mapped[list[str]] = mapped_column(JSON)
    signal_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    research_retries: Mapped[dict[str, int]] = mapped_column(JSON, default=dict)
    briefing_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    success: Mapped[bool] = mapped_column(default=True)


class SignalRow(Base):
    __tablename__ = "signals"
    __table_args__ = (
        # Makes save_signals idempotent if a run's write step is retried.
        UniqueConstraint("run_id", "competitor", "source_url", "title", name="uq_signal_per_run"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    competitor: Mapped[str] = mapped_column(String(128), index=True)
    signal_type: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(512))
    summary: Mapped[str] = mapped_column(Text)
    impact_assessment: Mapped[str] = mapped_column(String(16))
    source_url: Mapped[str] = mapped_column(Text)
    date_detected: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    confidence: Mapped[float] = mapped_column(Float)


class BriefingRow(Base):
    __tablename__ = "briefings"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    competitors_monitored: Mapped[list[str]] = mapped_column(JSON)
    signal_count: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)


def _row_to_dict(row: Base) -> dict[str, Any]:
    out = {}
    for col in row.__table__.columns:
        value = getattr(row, col.key)
        out[col.key] = value.isoformat() if isinstance(value, datetime) else value
    return out


class SemanticMemory:
    def __init__(self, db_path: str | None = None) -> None:
        path = Path(db_path or get_settings().sqlite_db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self._session = sessionmaker(bind=self.engine, expire_on_commit=False)

    # --- writes ----------------------------------------------------------

    def save_signals(self, signals: list[Signal], run_id: str) -> int:
        """Insert signals; duplicates within the same run are ignored. Returns rows inserted."""
        if not signals:
            return 0
        values = [{**s.model_dump(), "run_id": run_id} for s in signals]
        # Bind through the ORM column types so UTCDateTime conversion applies.
        stmt = sqlite_insert(SignalRow).values(values).on_conflict_do_nothing()
        with self._session.begin() as session:
            inserted = session.execute(stmt).rowcount
        logger.info("Saved %d/%d signals for run %s", inserted, len(signals), run_id)
        return inserted

    def save_briefing(self, briefing: Briefing) -> None:
        with self._session.begin() as session:
            session.merge(BriefingRow(**briefing.model_dump()))

    def save_run(self, result: RunResult) -> None:
        with self._session.begin() as session:
            session.merge(RunRow(**result.model_dump()))

    # --- reads -----------------------------------------------------------

    def get_signals(
        self,
        competitor: str | None = None,
        signal_type: str | None = None,
        run_id: str | None = None,
        since: datetime | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        stmt = select(SignalRow).order_by(SignalRow.date_detected.desc(), SignalRow.id.desc())
        if competitor:
            stmt = stmt.where(func.lower(SignalRow.competitor) == competitor.lower())
        if signal_type:
            stmt = stmt.where(SignalRow.signal_type == signal_type)
        if run_id:
            stmt = stmt.where(SignalRow.run_id == run_id)
        if since:
            stmt = stmt.where(SignalRow.date_detected >= since)
        with self._session() as session:
            rows = session.scalars(stmt.limit(limit).offset(offset)).all()
        return [_row_to_dict(r) for r in rows]

    def get_briefings(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        """Briefing metadata, newest first (content omitted for list views)."""
        stmt = select(BriefingRow).order_by(BriefingRow.created_at.desc()).limit(limit).offset(offset)
        with self._session() as session:
            rows = session.scalars(stmt).all()
        return [{k: v for k, v in _row_to_dict(r).items() if k != "content"} for r in rows]

    def get_briefing(self, run_id: str) -> dict[str, Any] | None:
        with self._session() as session:
            row = session.get(BriefingRow, run_id)
        return _row_to_dict(row) if row else None

    def get_latest_briefing(self) -> dict[str, Any] | None:
        stmt = select(BriefingRow).order_by(BriefingRow.created_at.desc()).limit(1)
        with self._session() as session:
            row = session.scalars(stmt).first()
        return _row_to_dict(row) if row else None

    def get_runs(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        stmt = select(RunRow).order_by(RunRow.started_at.desc()).limit(limit).offset(offset)
        with self._session() as session:
            rows = session.scalars(stmt).all()
        return [_row_to_dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._session() as session:
            row = session.get(RunRow, run_id)
        return _row_to_dict(row) if row else None
