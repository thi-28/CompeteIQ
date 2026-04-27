"""
Tests for CompeteIQ memory stores.

EpisodicMemory (ChromaDB) tests use a temporary directory and mock the
OpenAI embedding calls. SemanticMemory (SQLite) tests use an in-memory
database so no files are created on disk.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from models.schemas import Briefing, RunResult, Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(**kwargs) -> Signal:
    """Return a minimal valid Signal."""
    defaults = {
        "competitor": "Anthropic",
        "signal_type": "product_launch",
        "title": "Claude 4 Released",
        "summary": "Major model release with multimodal capabilities.",
        "impact_assessment": "high",
        "source_url": "https://example.com/news/claude4",
        "confidence": 0.9,
        "date_detected": datetime.utcnow(),
    }
    defaults.update(kwargs)
    return Signal(**defaults)


def _fake_embedding(text: str) -> list[float]:
    """Return a deterministic fake embedding (avoids real API calls in tests).

    Uses a seeded random number generator so different input strings produce
    genuinely distinct vectors — avoids false positives in cosine similarity.
    """
    import random

    rng = random.Random(hash(text))
    return [rng.random() for _ in range(1536)]


# ---------------------------------------------------------------------------
# EpisodicMemory tests
# ---------------------------------------------------------------------------


class TestEpisodicMemory:
    """Tests for memory/episodic.py."""

    def _make_memory(self, tmp_path: str) -> "EpisodicMemory":
        """Construct an EpisodicMemory with a mocked OpenAI client."""
        from memory.episodic import EpisodicMemory

        mock_openai = MagicMock()
        mock_openai.embeddings.create.side_effect = lambda model, input: MagicMock(
            data=[MagicMock(embedding=_fake_embedding(input))]
        )
        return EpisodicMemory(persist_dir=tmp_path, openai_client=mock_openai)

    def test_count_starts_at_zero(self, tmp_path) -> None:
        """Fresh store should have zero signals."""
        mem = self._make_memory(str(tmp_path))
        assert mem.count() == 0

    def test_store_and_count(self, tmp_path) -> None:
        """Storing signals should increase count."""
        mem = self._make_memory(str(tmp_path))
        signals = [_make_signal(), _make_signal(title="Pricing Update", signal_type="pricing_change")]
        stored = mem.store_signals(signals, run_id="run-001")
        assert stored == 2
        assert mem.count() == 2

    def test_retrieve_context_empty(self, tmp_path) -> None:
        """Retrieve from empty store should return empty list."""
        mem = self._make_memory(str(tmp_path))
        context = mem.retrieve_context(["Anthropic"])
        assert context == []

    def test_retrieve_context_after_store(self, tmp_path) -> None:
        """After storing signals, retrieve should return non-empty context."""
        mem = self._make_memory(str(tmp_path))
        mem.store_signals([_make_signal()], run_id="run-001")
        context = mem.retrieve_context(["Anthropic"])
        assert len(context) >= 1

    def test_duplicate_detection(self, tmp_path) -> None:
        """Storing the same signal twice should not create a duplicate."""
        mem = self._make_memory(str(tmp_path))
        signal = _make_signal()
        mem.store_signals([signal], run_id="run-001")
        # Store the same signal again — should be detected as duplicate
        stored = mem.store_signals([signal], run_id="run-002")
        assert stored == 0
        assert mem.count() == 1

    def test_is_duplicate_on_empty_store(self, tmp_path) -> None:
        """is_duplicate should return False on an empty store."""
        mem = self._make_memory(str(tmp_path))
        assert mem.is_duplicate(_make_signal()) is False


# ---------------------------------------------------------------------------
# SemanticMemory tests
# ---------------------------------------------------------------------------


class TestSemanticMemory:
    """Tests for memory/semantic.py — uses in-memory SQLite."""

    def _make_memory(self) -> "SemanticMemory":
        """Construct a SemanticMemory backed by an in-memory SQLite database."""
        from memory.semantic import SemanticMemory

        return SemanticMemory(db_path=":memory:")

    def test_save_and_retrieve_signals(self) -> None:
        """Saved signals should be retrievable."""
        mem = self._make_memory()
        signals = [_make_signal(), _make_signal(competitor="Google", title="Gemini Ultra 2")]
        mem.save_signals(signals, run_id="run-001")
        results = mem.get_signals()
        assert len(results) == 2

    def test_filter_by_competitor(self) -> None:
        """get_signals should filter by competitor name."""
        mem = self._make_memory()
        mem.save_signals(
            [_make_signal(competitor="Anthropic"), _make_signal(competitor="Google")],
            run_id="run-001",
        )
        results = mem.get_signals(competitor="Anthropic")
        assert all(r["competitor"] == "Anthropic" for r in results)
        assert len(results) == 1

    def test_filter_by_signal_type(self) -> None:
        """get_signals should filter by signal_type."""
        mem = self._make_memory()
        mem.save_signals(
            [
                _make_signal(signal_type="product_launch"),
                _make_signal(signal_type="pricing_change"),
            ],
            run_id="run-001",
        )
        results = mem.get_signals(signal_type="product_launch")
        assert len(results) == 1
        assert results[0]["signal_type"] == "product_launch"

    def test_save_and_retrieve_run(self) -> None:
        """Saved run records should be retrievable."""
        mem = self._make_memory()
        run = RunResult(
            run_id="test-run-001",
            started_at=datetime.utcnow(),
            competitors=["Anthropic"],
            signal_count=5,
            error_count=0,
        )
        mem.save_run(run)
        runs = mem.get_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "test-run-001"
        assert runs[0]["signal_count"] == 5

    def test_upsert_run_updates_existing(self) -> None:
        """Saving the same run_id twice should update rather than insert."""
        mem = self._make_memory()
        run = RunResult(
            run_id="test-run-001",
            started_at=datetime.utcnow(),
            competitors=["Anthropic"],
            signal_count=0,
        )
        mem.save_run(run)
        run.signal_count = 10
        mem.save_run(run)
        runs = mem.get_runs()
        assert len(runs) == 1
        assert runs[0]["signal_count"] == 10

    def test_save_and_retrieve_briefing(self) -> None:
        """Saved briefings should be retrievable by run_id."""
        mem = self._make_memory()
        briefing = Briefing(
            run_id="test-run-001",
            created_at=datetime.utcnow(),
            competitors_monitored=["Anthropic", "Google"],
            signal_count=3,
            content="## CompeteIQ Briefing\nTest content.",
        )
        mem.save_briefing(briefing)

        retrieved = mem.get_briefing("test-run-001")
        assert retrieved is not None
        assert retrieved["run_id"] == "test-run-001"
        assert "CompeteIQ" in retrieved["content"]

    def test_get_briefing_returns_none_for_missing(self) -> None:
        """get_briefing should return None for an unknown run_id."""
        mem = self._make_memory()
        assert mem.get_briefing("nonexistent-id") is None

    def test_get_briefings_limit(self) -> None:
        """get_briefings should respect the limit parameter."""
        mem = self._make_memory()
        for i in range(5):
            mem.save_briefing(
                Briefing(
                    run_id=f"run-{i:03d}",
                    created_at=datetime.utcnow(),
                    competitors_monitored=["Anthropic"],
                    signal_count=i,
                    content=f"Briefing {i}",
                )
            )
        results = mem.get_briefings(limit=3)
        assert len(results) == 3

    def test_get_signals_empty_db(self) -> None:
        """get_signals on a fresh database should return an empty list."""
        mem = self._make_memory()
        assert mem.get_signals() == []
