"""
Semantic memory store backed by SQLite via SQLAlchemy.

Persists structured Signal records and RunResult summaries to a relational
database so the API and dashboard can query, filter, and paginate intelligence
history without touching ChromaDB.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from models.schemas import Briefing, RunResult, Signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all CompeteIQ ORM models."""


class SignalRow(Base):
    """Relational representation of a single intelligence signal."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False, index=True)
    competitor = Column(String(128), nullable=False, index=True)
    signal_type = Column(String(64), nullable=False, index=True)
    title = Column(String(512), nullable=False)
    summary = Column(Text, nullable=False)
    impact_assessment = Column(String(16), nullable=False)
    source_url = Column(Text, nullable=False)
    date_detected = Column(DateTime, nullable=False)
    confidence = Column(Float, nullable=False)


class RunRow(Base):
    """Relational record of a single pipeline run."""

    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), unique=True, nullable=False, index=True)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    competitors = Column(Text, nullable=False)  # JSON array
    signal_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    briefing_id = Column(String(64), nullable=True)
    success = Column(Boolean, default=True)


class BriefingRow(Base):
    """Relational record of a generated intelligence briefing."""

    __tablename__ = "briefings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False)
    competitors_monitored = Column(Text, nullable=False)  # JSON array
    signal_count = Column(Integer, default=0)
    content = Column(Text, nullable=False)


# ---------------------------------------------------------------------------
# SemanticMemory facade
# ---------------------------------------------------------------------------


class SemanticMemory:
    """
    SQLite-backed knowledge base for CompeteIQ run history.

    Wraps all SQLAlchemy I/O behind a clean domain-level interface so the
    rest of the codebase never touches ORM internals directly.
    """

    def __init__(self, db_path: str | None = None) -> None:
        """
        Initialise the database engine and ensure all tables exist.

        Args:
            db_path: Filesystem path for the SQLite file.
                     Defaults to SQLITE_DB_PATH env var or ./data/competeiq.db.
        """
        resolved = db_path or os.getenv("SQLITE_DB_PATH", "./data/competeiq.db")
        os.makedirs(os.path.dirname(os.path.abspath(resolved)), exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{resolved}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self._engine)
        self._SessionLocal = sessionmaker(bind=self._engine, expire_on_commit=False)
        logger.info("SemanticMemory initialised — db=%s", resolved)

    def _session(self) -> Session:
        """Return a new SQLAlchemy session."""
        return self._SessionLocal()

    # ------------------------------------------------------------------
    # Signal persistence
    # ------------------------------------------------------------------

    def save_signals(self, signals: list[Signal], run_id: str) -> int:
        """
        Persist a batch of Signal instances to the signals table.

        Args:
            signals: List of validated Signal objects.
            run_id: The originating pipeline run identifier.

        Returns:
            The number of rows inserted.
        """
        rows = [
            SignalRow(
                run_id=run_id,
                competitor=s.competitor,
                signal_type=s.signal_type,
                title=s.title,
                summary=s.summary,
                impact_assessment=s.impact_assessment,
                source_url=s.source_url,
                date_detected=s.date_detected,
                confidence=s.confidence,
            )
            for s in signals
        ]
        with self._session() as session:
            session.add_all(rows)
            session.commit()
        logger.info("Saved %d signals for run_id=%s", len(rows), run_id)
        return len(rows)

    def get_signals(
        self,
        competitor: str | None = None,
        signal_type: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """
        Query stored signals with optional filters.

        Args:
            competitor: Filter by competitor name (case-insensitive partial match).
            signal_type: Filter by exact signal_type string.
            limit: Maximum number of rows to return.

        Returns:
            List of signal dicts ordered by date_detected descending.
        """
        with self._session() as session:
            stmt = select(SignalRow).order_by(SignalRow.date_detected.desc())
            if competitor:
                stmt = stmt.where(
                    SignalRow.competitor.ilike(f"%{competitor}%")
                )
            if signal_type:
                stmt = stmt.where(SignalRow.signal_type == signal_type)
            stmt = stmt.limit(limit)
            rows = session.execute(stmt).scalars().all()

        return [
            {
                "id": r.id,
                "run_id": r.run_id,
                "competitor": r.competitor,
                "signal_type": r.signal_type,
                "title": r.title,
                "summary": r.summary,
                "impact_assessment": r.impact_assessment,
                "source_url": r.source_url,
                "date_detected": r.date_detected.isoformat() if r.date_detected else None,
                "confidence": r.confidence,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Run persistence
    # ------------------------------------------------------------------

    def save_run(self, result: RunResult) -> None:
        """
        Insert or update a RunResult record.

        Args:
            result: The RunResult domain object to persist.
        """
        with self._session() as session:
            # Upsert by run_id
            existing = session.execute(
                select(RunRow).where(RunRow.run_id == result.run_id)
            ).scalar_one_or_none()

            if existing:
                existing.completed_at = result.completed_at
                existing.signal_count = result.signal_count
                existing.error_count = result.error_count
                existing.briefing_id = result.briefing_id
                existing.success = result.success
            else:
                row = RunRow(
                    run_id=result.run_id,
                    started_at=result.started_at,
                    completed_at=result.completed_at,
                    competitors=json.dumps(result.competitors),
                    signal_count=result.signal_count,
                    error_count=result.error_count,
                    briefing_id=result.briefing_id,
                    success=result.success,
                )
                session.add(row)
            session.commit()
        logger.info("Saved run record run_id=%s", result.run_id)

    def get_runs(self, limit: int = 50) -> list[dict]:
        """
        Retrieve recent run summaries ordered by started_at descending.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            List of run summary dicts.
        """
        with self._session() as session:
            rows = (
                session.execute(
                    select(RunRow).order_by(RunRow.started_at.desc()).limit(limit)
                )
                .scalars()
                .all()
            )
        return [
            {
                "run_id": r.run_id,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "competitors": json.loads(r.competitors) if r.competitors else [],
                "signal_count": r.signal_count,
                "error_count": r.error_count,
                "briefing_id": r.briefing_id,
                "success": r.success,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Briefing persistence
    # ------------------------------------------------------------------

    def save_briefing(self, briefing: Briefing) -> None:
        """
        Persist a generated Briefing to the briefings table.

        Args:
            briefing: The Briefing domain object to store.
        """
        row = BriefingRow(
            run_id=briefing.run_id,
            created_at=briefing.created_at,
            competitors_monitored=json.dumps(briefing.competitors_monitored),
            signal_count=briefing.signal_count,
            content=briefing.content,
        )
        with self._session() as session:
            session.add(row)
            session.commit()
        logger.info("Saved briefing for run_id=%s", briefing.run_id)

    def get_briefings(self, limit: int = 10) -> list[dict]:
        """
        Retrieve recent briefings ordered by created_at descending.

        Args:
            limit: Maximum number of briefings to return.

        Returns:
            List of briefing dicts (without full content, use get_briefing for that).
        """
        with self._session() as session:
            rows = (
                session.execute(
                    select(BriefingRow)
                    .order_by(BriefingRow.created_at.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
        return [
            {
                "run_id": r.run_id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "competitors_monitored": json.loads(r.competitors_monitored)
                if r.competitors_monitored
                else [],
                "signal_count": r.signal_count,
                "content": r.content,
            }
            for r in rows
        ]

    def get_briefing(self, run_id: str) -> dict | None:
        """
        Retrieve a specific briefing by run_id.

        Args:
            run_id: The run identifier for the desired briefing.

        Returns:
            A briefing dict, or None if not found.
        """
        with self._session() as session:
            row = session.execute(
                select(BriefingRow).where(BriefingRow.run_id == run_id)
            ).scalar_one_or_none()

        if row is None:
            return None

        return {
            "run_id": row.run_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "competitors_monitored": json.loads(row.competitors_monitored)
            if row.competitors_monitored
            else [],
            "signal_count": row.signal_count,
            "content": row.content,
        }
