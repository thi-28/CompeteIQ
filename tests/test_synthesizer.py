import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from agents.synthesizer import SynthesizerAgent, filter_cited
from models import Signal
from tools.formatter import format_briefing, order_signals, signals_to_context_block

RUN_DATE = datetime(2026, 11, 16, 9, 0, tzinfo=timezone.utc)


def sig(competitor="Anthropic", title="Launch", impact="high", confidence=0.9, url="https://a.com/1"):
    return Signal(competitor=competitor, signal_type="product_launch", title=title, summary=f"{title} summary.",
                  impact_assessment=impact, source_url=url, confidence=confidence)


def llm(payload):
    client = MagicMock()
    if isinstance(payload, Exception):
        client.chat.completions.create.side_effect = payload
    else:
        content = payload if isinstance(payload, str) else json.dumps(payload)
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return client


def state(signals, **kw):
    s = {"run_id": "r1", "started_at": RUN_DATE.isoformat(), "competitors": ["Anthropic", "Meta"],
         "extracted_signals": [x.model_dump(mode="json") for x in signals]}
    s.update(kw)
    return s


GOOD = {"executive_summary": "Big week.", "strategic_implications": ["Pressure rising [1][2]", "Unsupported claim", "Bad ref [9]"],
        "signals_to_watch": ["- Watch pricing"]}


# --- formatter ------------------------------------------------------------

def test_order_signals_by_competitor_then_impact_then_confidence():
    ordered = order_signals([
        sig("Meta", "m-high"), sig("Anthropic", "a-low", "low"),
        sig("Anthropic", "a-high-lo", confidence=0.7), sig("Anthropic", "a-high-hi", confidence=0.95),
        sig("Unknown", "u"),
    ], ["Anthropic", "Meta"])
    assert [s.title for s in ordered] == ["a-high-hi", "a-high-lo", "a-low", "m-high", "u"]


def test_context_block_is_numbered():
    block = signals_to_context_block([sig(), sig("Meta", "M")])
    assert block.splitlines()[1].startswith("[2] Meta | product_launch | impact=high | M:")
    assert signals_to_context_block([]) == "No signals."


def test_format_briefing_links_sources_and_marks_empty_competitors():
    md = format_briefing(run_id="r1", run_date=RUN_DATE, competitors=["Anthropic", "Meta"], signals=[sig()],
                         executive_summary="Summary.", strategic_implications=["Thing [1]"],
                         signals_to_watch=["• Watch"], retry_counts={"Meta": 2})
    assert md.startswith("# CompeteIQ Briefing: 2026-11-16")
    assert "**[1] Product Launch: Launch** (HIGH impact, confidence 0.90)" in md
    assert "[source](https://a.com/1)" in md
    assert "_No grounded signals found after 2 re-searches._" in md
    assert "- Thing [1]" in md and "- Watch" in md


def test_format_briefing_without_narrative_does_not_invent_content():
    md = format_briefing(run_id="r1", run_date=RUN_DATE, competitors=["Anthropic"], signals=[sig()],
                         executive_summary=None, strategic_implications=[], signals_to_watch=[])
    assert "Narrative unavailable" in md
    assert "## Strategic Implications" not in md and "## Signals to Watch" not in md
    assert "[source](https://a.com/1)" in md


def test_format_briefing_no_signals():
    md = format_briefing(run_id="r1", run_date=RUN_DATE, competitors=["Anthropic"], signals=[],
                         executive_summary=None, strategic_implications=[], signals_to_watch=[])
    assert "No signals met the grounding and confidence thresholds" in md
    assert "_No grounded signals found._" in md


def test_format_is_deterministic():
    kw = dict(run_id="r", run_date=RUN_DATE, competitors=["Anthropic"], signals=[sig()],
              executive_summary="S", strategic_implications=["x [1]"], signals_to_watch=[])
    assert format_briefing(**kw) == format_briefing(**kw)


# --- citation filter ------------------------------------------------------

def test_filter_cited():
    assert filter_cited(["ok [1]", "ok [1][3]", "none", "out [4]", "zero [0]", "mixed [1][7]"], 3) == ["ok [1]", "ok [1][3]"]


# --- agent ----------------------------------------------------------------

def test_agent_builds_briefing_and_filters_uncited_implications():
    client = llm(GOOD)
    out = SynthesizerAgent(client=client)(state([sig(), sig("Meta", "M", url="https://m.com")]))

    md = out["briefing"]
    assert "Big week." in md
    assert "- Pressure rising [1][2]" in md
    assert "Unsupported claim" not in md and "Bad ref" not in md
    assert "- Watch pricing" in md
    assert out["errors"] == []
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert "[1] Anthropic" in kwargs["messages"][1]["content"]


def test_agent_skips_llm_when_no_signals():
    client = llm(GOOD)
    out = SynthesizerAgent(client=client)(state([]))
    client.chat.completions.create.assert_not_called()
    assert "No signals met" in out["briefing"]


def test_agent_llm_failure_records_error_and_renders_signals():
    out = SynthesizerAgent(client=llm(RuntimeError("down")))(state([sig()]))
    assert "Narrative unavailable" in out["briefing"]
    assert "[source](https://a.com/1)" in out["briefing"]
    assert out["errors"] == ["synthesizer:narrative: down"]


def test_agent_bad_json_or_missing_summary_is_a_failure():
    for payload in ("not json", {"strategic_implications": ["x [1]"]}):
        out = SynthesizerAgent(client=llm(payload))(state([sig()]))
        assert "Narrative unavailable" in out["briefing"]
        assert len(out["errors"]) == 1


def test_agent_skips_invalid_signal_dicts():
    st = state([sig()])
    st["extracted_signals"].append({"competitor": "Meta", "title": "broken"})
    out = SynthesizerAgent(client=llm(GOOD))(st)
    assert "Signals:** 1" in out["briefing"]
    assert len(out["errors"]) == 1 and "invalid signal" in out["errors"][0]


def test_agent_uses_started_at_for_date():
    out = SynthesizerAgent(client=llm(GOOD))(state([sig()], started_at="2026-01-05T00:00:00+00:00"))
    assert "# CompeteIQ Briefing: 2026-01-05" in out["briefing"]
