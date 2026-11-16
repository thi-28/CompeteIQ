from datetime import timezone

import pytest
from pydantic import ValidationError

import config
from models import Briefing, RunResult, Signal, Snippet


def _signal(**overrides):
    data = dict(
        competitor="Anthropic",
        signal_type="product_launch",
        title="New model",
        summary="Released a new model.",
        impact_assessment="high",
        source_url="https://example.com/a",
        confidence=0.876,
    )
    data.update(overrides)
    return Signal(**data)


def test_signal_defaults_are_timezone_aware_utc():
    s = _signal()
    assert s.date_detected.tzinfo is not None
    assert s.date_detected.utcoffset() == timezone.utc.utcoffset(None)


def test_signal_rounds_confidence():
    assert _signal().confidence == 0.88


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_signal_rejects_out_of_range_confidence(bad):
    with pytest.raises(ValidationError):
        _signal(confidence=bad)


def test_signal_rejects_non_http_url():
    with pytest.raises(ValidationError):
        _signal(source_url="not-a-url")


def test_signal_rejects_unknown_type():
    with pytest.raises(ValidationError):
        _signal(signal_type="rumor")


def test_snippet_round_trip():
    s = Snippet(competitor="Meta", url="https://x.com", content="text")
    assert Snippet(**s.model_dump()) == s


def test_briefing_and_runresult_utc_defaults():
    b = Briefing(run_id="r1", competitors_monitored=["Meta"], signal_count=0, content="# x")
    r = RunResult(run_id="r1", started_at=b.created_at, competitors=["Meta"])
    assert b.created_at.tzinfo is not None
    assert r.completed_at.tzinfo is not None
    assert r.research_retries == {}


def test_settings_defaults(monkeypatch):
    for var in ("OPENAI_MODEL", "EMBEDDING_MODEL", "MAX_SNIPPETS_PER_COMPETITOR"):
        monkeypatch.delenv(var, raising=False)
    s = config.get_settings()
    assert s.openai_model == "gpt-4o-mini"
    assert s.embedding_model == "text-embedding-3-small"
    assert s.max_snippets_per_competitor == 10
    assert s.max_research_retries == 2


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("MAX_SNIPPETS_PER_COMPETITOR", "5")
    monkeypatch.setenv("DEFAULT_COMPETITORS", "A, B ,")
    s = config.get_settings()
    assert s.max_snippets_per_competitor == 5
    assert s.default_competitors == ["A", "B"]


def test_env_example_matches_code_defaults():
    """Guards against model-name drift between code and .env.example."""
    from pathlib import Path

    text = (Path(__file__).parent.parent / ".env.example").read_text()
    assert f"OPENAI_MODEL={config.DEFAULT_OPENAI_MODEL}" in text
    assert f"EMBEDDING_MODEL={config.DEFAULT_EMBEDDING_MODEL}" in text
