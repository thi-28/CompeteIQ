"""
Tests for the CompeteIQ FastAPI layer.

Uses FastAPI's TestClient so no running server is needed.
All database I/O is directed to an in-memory SQLite instance so tests
are hermetic and leave no files on disk.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from models.schemas import Briefing, RunResult, Signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path) -> TestClient:
    """
    Return a FastAPI TestClient.

    The API endpoints lazily import SemanticMemory inside function bodies,
    so we patch at the source module (memory.semantic.SemanticMemory) with
    a factory that returns a shared in-memory-backed instance.
    """
    from memory.semantic import SemanticMemory

    db_file = str(tmp_path / "test.db")
    mem_instance = SemanticMemory(db_path=db_file)

    # Patch the class in its home module; the API's `from memory.semantic import
    # SemanticMemory` inside function bodies will pick up the patched version.
    with patch("memory.semantic.SemanticMemory", return_value=mem_instance):
        from api.main import app
        yield TestClient(app)


def _seed_briefing(mem, run_id: str = "run-001") -> None:
    """Insert a test briefing into the given SemanticMemory instance."""
    mem.save_briefing(
        Briefing(
            run_id=run_id,
            created_at=datetime.utcnow(),
            competitors_monitored=["Anthropic", "Google"],
            signal_count=3,
            content="## CompeteIQ Briefing\nTest content for run " + run_id,
        )
    )


def _seed_signal(mem, competitor: str = "Anthropic", run_id: str = "run-001") -> None:
    """Insert a test signal into the given SemanticMemory instance."""
    mem.save_signals(
        [
            Signal(
                competitor=competitor,
                signal_type="product_launch",
                title="Test Launch",
                summary="A test product launch.",
                impact_assessment="high",
                source_url="https://example.com",
                confidence=0.9,
                date_detected=datetime.utcnow(),
            )
        ],
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, tmp_path) -> None:
        """Health endpoint should always return 200."""
        with patch("memory.semantic.SemanticMemory"), \
             patch("memory.episodic.EpisodicMemory"):
            from api.main import app
            with TestClient(app) as c:
                r = c.get("/health")
        assert r.status_code == 200

    def test_health_response_shape(self, tmp_path) -> None:
        """Health response should include status and timestamp fields."""
        with patch("memory.semantic.SemanticMemory"), \
             patch("memory.episodic.EpisodicMemory"):
            from api.main import app
            with TestClient(app) as c:
                r = c.get("/health")
        data = r.json()
        assert "status" in data
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# Run endpoint
# ---------------------------------------------------------------------------


class TestRunEndpoint:
    """Tests for POST /run."""

    def test_post_run_returns_202(self, client) -> None:
        """POST /run should return 202 Accepted."""
        with patch("api.main._run_pipeline_task"):
            r = client.post("/run", json={"competitors": ["Anthropic"]})
        assert r.status_code == 202

    def test_post_run_returns_run_id(self, client) -> None:
        """POST /run response should contain a run_id."""
        with patch("api.main._run_pipeline_task"):
            r = client.post("/run", json={})
        data = r.json()
        assert "run_id" in data
        assert len(data["run_id"]) > 0

    def test_post_run_uses_provided_run_id(self, client) -> None:
        """POST /run should honour a caller-supplied run_id."""
        custom_id = "my-custom-run-id"
        with patch("api.main._run_pipeline_task"):
            r = client.post("/run", json={"run_id": custom_id})
        assert r.json()["run_id"] == custom_id


# ---------------------------------------------------------------------------
# Briefings endpoints
# ---------------------------------------------------------------------------


class TestBriefingsEndpoints:
    """Tests for GET /briefings and GET /briefings/{run_id}."""

    def test_list_briefings_empty(self, client) -> None:
        """GET /briefings on empty DB should return empty list."""
        with patch("memory.semantic.SemanticMemory") as mock_cls:
            mock_cls.return_value.get_briefings.return_value = []
            r = client.get("/briefings")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_briefings_returns_data(self, client) -> None:
        """GET /briefings should return stored briefings."""
        with patch("memory.semantic.SemanticMemory") as mock_cls:
            mock_cls.return_value.get_briefings.return_value = [
                {
                    "run_id": "run-001",
                    "created_at": datetime.utcnow().isoformat(),
                    "competitors_monitored": ["Anthropic"],
                    "signal_count": 1,
                    "content": "## Briefing",
                }
            ]
            r = client.get("/briefings")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["run_id"] == "run-001"

    def test_get_briefing_by_run_id(self, client) -> None:
        """GET /briefings/{run_id} should return the correct briefing."""
        briefing_data = {
            "run_id": "run-001",
            "created_at": datetime.utcnow().isoformat(),
            "competitors_monitored": ["Anthropic"],
            "signal_count": 2,
            "content": "## Weekly Brief",
        }
        with patch("memory.semantic.SemanticMemory") as mock_cls:
            mock_cls.return_value.get_briefing.return_value = briefing_data
            r = client.get("/briefings/run-001")
        assert r.status_code == 200
        assert r.json()["run_id"] == "run-001"

    def test_get_briefing_not_found(self, client) -> None:
        """GET /briefings/{run_id} should return 404 for unknown run_id."""
        with patch("memory.semantic.SemanticMemory") as mock_cls:
            mock_cls.return_value.get_briefing.return_value = None
            r = client.get("/briefings/nonexistent-id")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Signals endpoint
# ---------------------------------------------------------------------------


class TestSignalsEndpoint:
    """Tests for GET /signals."""

    def test_list_signals_empty(self, client) -> None:
        """GET /signals on empty DB should return empty list."""
        with patch("memory.semantic.SemanticMemory") as mock_cls:
            mock_cls.return_value.get_signals.return_value = []
            r = client.get("/signals")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_signals_with_competitor_filter(self, client) -> None:
        """GET /signals?competitor=Anthropic should pass filter to SemanticMemory."""
        with patch("memory.semantic.SemanticMemory") as mock_cls:
            mock_cls.return_value.get_signals.return_value = []
            r = client.get("/signals?competitor=Anthropic")
        assert r.status_code == 200
        mock_cls.return_value.get_signals.assert_called_once_with(
            competitor="Anthropic", signal_type=None, limit=100
        )

    def test_list_signals_with_type_filter(self, client) -> None:
        """GET /signals?signal_type=product_launch should pass filter."""
        with patch("memory.semantic.SemanticMemory") as mock_cls:
            mock_cls.return_value.get_signals.return_value = []
            r = client.get("/signals?signal_type=product_launch")
        assert r.status_code == 200
        mock_cls.return_value.get_signals.assert_called_once_with(
            competitor=None, signal_type="product_launch", limit=100
        )

    def test_list_signals_limit_parameter(self, client) -> None:
        """GET /signals?limit=5 should pass limit to SemanticMemory."""
        with patch("memory.semantic.SemanticMemory") as mock_cls:
            mock_cls.return_value.get_signals.return_value = []
            r = client.get("/signals?limit=5")
        assert r.status_code == 200
        mock_cls.return_value.get_signals.assert_called_once_with(
            competitor=None, signal_type=None, limit=5
        )
