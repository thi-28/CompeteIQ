"""
Tests for the CompeteIQ agent nodes.

Uses unittest.mock to isolate OpenAI and Tavily calls so tests run
without live API keys. Each test covers the happy path and at least one
error/edge case per agent.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from models.schemas import AgentState, Signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_state(**kwargs) -> AgentState:
    """Build a minimal AgentState for use in tests."""
    defaults: AgentState = {
        "competitors": ["Anthropic", "Google"],
        "search_queries": {},
        "raw_signals": {},
        "extracted_signals": [],
        "memory_context": [],
        "memory_mode": "read",
        "briefing": "",
        "run_id": "test-run-001",
        "started_at": datetime.utcnow().isoformat(),
        "next_agent": "",
        "errors": [],
    }
    defaults.update(kwargs)
    return defaults


def _make_signal(**kwargs) -> Signal:
    """Create a minimal valid Signal for testing."""
    defaults = {
        "competitor": "Anthropic",
        "signal_type": "product_launch",
        "title": "Claude 4 Released",
        "summary": "Anthropic launched Claude 4 with improved reasoning capabilities.",
        "impact_assessment": "high",
        "source_url": "https://example.com/claude4",
        "confidence": 0.9,
        "date_detected": datetime.utcnow(),
    }
    defaults.update(kwargs)
    return Signal(**defaults)


# ---------------------------------------------------------------------------
# Supervisor tests
# ---------------------------------------------------------------------------


class TestSupervisorAgent:
    """Tests for agents/supervisor.py."""

    def _make_openai_mock(self, json_content: str) -> MagicMock:
        """Return a mock OpenAI client whose chat.completions.create returns json_content."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json_content
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_generates_queries_on_first_call(self) -> None:
        """Supervisor should generate queries and route to collector on first call."""
        from agents.supervisor import SupervisorAgent

        queries_json = json.dumps(
            {
                "Anthropic": ["q1", "q2", "q3"],
                "Google": ["g1", "g2", "g3"],
            }
        )
        agent = SupervisorAgent(openai_client=self._make_openai_mock(queries_json))
        state = _make_state(next_agent="")
        result = agent.run(state)

        assert result["next_agent"] == "collector"
        assert "Anthropic" in result["search_queries"]
        assert len(result["search_queries"]["Anthropic"]) == 3

    def test_routes_to_memory_read_after_collection(self) -> None:
        """Supervisor should route to memory_read when next_agent is memory_read."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(openai_client=MagicMock())
        state = _make_state(next_agent="memory_read")
        result = agent.run(state)
        assert result["next_agent"] == "memory_read"
        assert result.get("memory_mode") == "read"

    def test_routes_to_extractor(self) -> None:
        """Supervisor should route to extractor when instructed."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(openai_client=MagicMock())
        state = _make_state(next_agent="extractor")
        result = agent.run(state)
        assert result["next_agent"] == "extractor"

    def test_routes_to_synthesizer(self) -> None:
        """Supervisor should route to synthesizer when instructed."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(openai_client=MagicMock())
        state = _make_state(next_agent="synthesizer")
        result = agent.run(state)
        assert result["next_agent"] == "synthesizer"

    def test_routes_to_memory_write(self) -> None:
        """Supervisor should route to memory_write when instructed."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(openai_client=MagicMock())
        state = _make_state(next_agent="memory_write")
        result = agent.run(state)
        assert result["next_agent"] == "memory_write"
        assert result.get("memory_mode") == "write"

    def test_falls_back_on_llm_error(self) -> None:
        """Supervisor should use fallback queries if LLM call raises an exception."""
        from agents.supervisor import SupervisorAgent

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("LLM down")
        agent = SupervisorAgent(openai_client=mock_client)
        state = _make_state(next_agent="")
        result = agent.run(state)

        assert result["next_agent"] == "collector"
        assert len(result["errors"]) > 0
        # Fallback queries should still be generated
        assert "Anthropic" in result["search_queries"]

    def test_falls_back_on_invalid_json(self) -> None:
        """Supervisor should use fallback queries if LLM returns invalid JSON."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(
            openai_client=self._make_openai_mock("not valid json at all")
        )
        state = _make_state(next_agent="")
        result = agent.run(state)
        assert result["next_agent"] == "collector"
        assert len(result["errors"]) > 0


# ---------------------------------------------------------------------------
# Collector tests
# ---------------------------------------------------------------------------


class TestCollectorAgent:
    """Tests for agents/collector.py."""

    def _make_search_mock(self, results: list[dict] | None = None) -> MagicMock:
        """Return a mock TavilySearchTool."""
        mock = MagicMock()
        mock.search.return_value = results or [
            {
                "title": "Test Title",
                "url": "https://example.com",
                "content": "Test content snippet.",
                "score": 0.9,
            }
        ]
        return mock

    def test_collects_snippets_for_all_competitors(self) -> None:
        """Collector should populate raw_signals for every competitor."""
        from agents.collector import CollectorAgent

        search_mock = self._make_search_mock()
        agent = CollectorAgent(search_tool=search_mock)
        state = _make_state(
            search_queries={
                "Anthropic": ["q1", "q2"],
                "Google": ["g1"],
            }
        )
        result = agent.run(state)

        assert "Anthropic" in result["raw_signals"]
        assert "Google" in result["raw_signals"]
        # 2 queries × 1 result each = 2 snippets for Anthropic
        assert len(result["raw_signals"]["Anthropic"]) == 2

    def test_routes_to_memory_read(self) -> None:
        """Collector should always set next_agent to memory_read."""
        from agents.collector import CollectorAgent

        agent = CollectorAgent(search_tool=self._make_search_mock())
        result = agent.run(_make_state(search_queries={"Anthropic": ["q1"]}))
        assert result["next_agent"] == "memory_read"

    def test_logs_failed_searches_to_errors(self) -> None:
        """Collector should log failed queries to errors and continue."""
        from agents.collector import CollectorAgent

        mock = MagicMock()
        mock.search.side_effect = RuntimeError("Search API down")
        agent = CollectorAgent(search_tool=mock)
        result = agent.run(_make_state(search_queries={"Anthropic": ["q1", "q2"]}))

        assert result["next_agent"] == "memory_read"
        assert len(result["errors"]) == 2  # both queries failed
        assert result["raw_signals"]["Anthropic"] == []

    def test_empty_queries_produces_empty_snippets(self) -> None:
        """Collector should handle empty search_queries gracefully."""
        from agents.collector import CollectorAgent

        agent = CollectorAgent(search_tool=self._make_search_mock())
        result = agent.run(_make_state(search_queries={}))
        assert result["raw_signals"] == {}
        assert result["next_agent"] == "memory_read"


# ---------------------------------------------------------------------------
# Extractor tests
# ---------------------------------------------------------------------------


class TestExtractorAgent:
    """Tests for agents/extractor.py."""

    def _make_openai_mock(self, signals_data: list[dict]) -> MagicMock:
        """Return a mock OpenAI client returning the given signals JSON."""
        payload = json.dumps({"signals": signals_data})
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = payload
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_extracts_signals_above_threshold(self) -> None:
        """Extractor should return signals with confidence >= 0.6."""
        from agents.extractor import ExtractorAgent

        raw_signal = {
            "competitor": "Anthropic",
            "signal_type": "product_launch",
            "title": "Claude 4 Released",
            "summary": "New model launched.",
            "impact_assessment": "high",
            "source_url": "https://example.com",
            "confidence": 0.85,
            "date_detected": datetime.utcnow().isoformat(),
        }
        agent = ExtractorAgent(openai_client=self._make_openai_mock([raw_signal]))
        state = _make_state(
            raw_signals={"Anthropic": ["some raw text"]},
            memory_context=[],
        )
        result = agent.run(state)
        assert len(result["extracted_signals"]) == 1
        assert result["extracted_signals"][0]["confidence"] == 0.85

    def test_filters_low_confidence_signals(self) -> None:
        """Extractor should discard signals with confidence < 0.6."""
        from agents.extractor import ExtractorAgent

        low_conf = {
            "competitor": "Anthropic",
            "signal_type": "other",
            "title": "Rumour",
            "summary": "Unverified rumour.",
            "impact_assessment": "low",
            "source_url": "https://example.com",
            "confidence": 0.4,
            "date_detected": datetime.utcnow().isoformat(),
        }
        agent = ExtractorAgent(openai_client=self._make_openai_mock([low_conf]))
        result = agent.run(_make_state(raw_signals={"Anthropic": ["raw"]}))
        assert result["extracted_signals"] == []

    def test_handles_empty_snippets(self) -> None:
        """Extractor should skip competitors with no snippets."""
        from agents.extractor import ExtractorAgent

        agent = ExtractorAgent(openai_client=MagicMock())
        result = agent.run(_make_state(raw_signals={"Anthropic": []}))
        assert result["extracted_signals"] == []
        # OpenAI should not have been called for an empty competitor
        agent._client.chat.completions.create.assert_not_called()

    def test_routes_to_synthesizer(self) -> None:
        """Extractor should always route to synthesizer."""
        from agents.extractor import ExtractorAgent

        agent = ExtractorAgent(openai_client=self._make_openai_mock([]))
        result = agent.run(_make_state(raw_signals={"Anthropic": ["raw"]}))
        assert result["next_agent"] == "synthesizer"

    def test_handles_llm_error_gracefully(self) -> None:
        """Extractor should log error and continue if LLM call fails."""
        from agents.extractor import ExtractorAgent

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("LLM error")
        agent = ExtractorAgent(openai_client=mock_client)
        result = agent.run(_make_state(raw_signals={"Anthropic": ["raw"]}))
        assert result["next_agent"] == "synthesizer"
        assert len(result["errors"]) > 0


# ---------------------------------------------------------------------------
# Synthesizer tests
# ---------------------------------------------------------------------------


class TestSynthesizerAgent:
    """Tests for agents/synthesizer.py."""

    def _make_openai_mock(self) -> MagicMock:
        """Return a mock OpenAI client with a plausible synthesis response."""
        narrative = json.dumps(
            {
                "executive_summary": "Strong competitive activity this week.",
                "strategic_implications": ["OpenAI must accelerate.", "Monitor pricing."],
                "signals_to_watch": ["Claude 4 reception.", "Google Gemini adoption."],
            }
        )
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = narrative
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_generates_briefing_markdown(self) -> None:
        """Synthesizer should produce non-empty markdown briefing."""
        from agents.synthesizer import SynthesizerAgent

        signal = _make_signal()
        agent = SynthesizerAgent(openai_client=self._make_openai_mock())
        state = _make_state(
            extracted_signals=[signal.model_dump(mode="json")],
            competitors=["Anthropic"],
            run_id="test-001",
        )
        result = agent.run(state)
        assert "briefing" in result
        assert "CompeteIQ" in result["briefing"]
        assert "Anthropic" in result["briefing"]

    def test_routes_to_memory_write(self) -> None:
        """Synthesizer should always route to memory_write."""
        from agents.synthesizer import SynthesizerAgent

        agent = SynthesizerAgent(openai_client=self._make_openai_mock())
        result = agent.run(_make_state(extracted_signals=[], competitors=["Anthropic"]))
        assert result["next_agent"] == "memory_write"

    def test_handles_empty_signals(self) -> None:
        """Synthesizer should produce a valid briefing even with no signals."""
        from agents.synthesizer import SynthesizerAgent

        agent = SynthesizerAgent(openai_client=self._make_openai_mock())
        result = agent.run(_make_state(extracted_signals=[], competitors=["Anthropic"]))
        assert isinstance(result["briefing"], str)
        assert len(result["briefing"]) > 0

    def test_handles_llm_error_gracefully(self) -> None:
        """Synthesizer should produce a fallback briefing on LLM failure."""
        from agents.synthesizer import SynthesizerAgent

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("LLM down")
        agent = SynthesizerAgent(openai_client=mock_client)
        result = agent.run(_make_state(extracted_signals=[], competitors=["Anthropic"]))
        assert result["next_agent"] == "memory_write"
        # Should still produce some briefing content
        assert isinstance(result["briefing"], str)
