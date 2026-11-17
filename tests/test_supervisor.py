import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

from agents.supervisor import SupervisorAgent, _failure_reason, signal_counts
from config import Settings

SETTINGS = replace(Settings(), min_signals_per_competitor=2, max_research_retries=2)


def llm(*payloads):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        p if isinstance(p, Exception) else SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=p if isinstance(p, str) else json.dumps(p)))])
        for p in payloads
    ]
    return client


def sigs(**counts):
    return [{"competitor": c} for c, n in counts.items() for _ in range(n)]


# --- plan --------------------------------------------------------------------

def test_plan_uses_llm_queries_and_initializes_tracking():
    client = llm({"A": ["a1", "a2", "a2", "a3", "a4"], "B": ["b1"]})
    out = SupervisorAgent(client, SETTINGS).plan({"competitors": ["A", "B"]})
    assert out["search_queries"] == {"A": ["a1", "a2", "a3"], "B": ["b1"]}
    assert out["query_history"] == out["search_queries"]
    assert out["retry_counts"] == {"A": 0, "B": 0}
    assert out["low_signal_competitors"] == [] and out["errors"] == []


def test_plan_falls_back_per_competitor_and_on_failure():
    out = SupervisorAgent(llm({"A": ["a1"]}), SETTINGS).plan({"competitors": ["A", "B"]})
    assert out["search_queries"]["A"] == ["a1"]
    assert len(out["search_queries"]["B"]) == 3 and all("B" in q for q in out["search_queries"]["B"])

    out = SupervisorAgent(llm(RuntimeError("down")), SETTINGS).plan({"competitors": ["A"]})
    assert len(out["search_queries"]["A"]) == 3
    assert "fallback" in out["errors"][0]


# --- review / route (the decision) ------------------------------------------

def test_review_flags_low_signal_competitors_with_retries_left():
    sup = SupervisorAgent(MagicMock(), SETTINGS)
    state = {"competitors": ["A", "B", "C", "D"], "extracted_signals": sigs(A=2, B=1, D=0),
             "retry_counts": {"A": 0, "B": 0, "C": 1, "D": 2}}
    out = sup.review(state)
    assert out["low_signal_competitors"] == ["B", "C"]  # D exhausted retries, A has enough
    assert SupervisorAgent.route({**state, **out}) == "rewrite"


def test_review_proceeds_when_all_satisfied_or_exhausted():
    sup = SupervisorAgent(MagicMock(), SETTINGS)
    state = {"competitors": ["A", "B"], "extracted_signals": sigs(A=3), "retry_counts": {"A": 0, "B": 2}}
    out = sup.review(state)
    assert out["low_signal_competitors"] == []
    assert SupervisorAgent.route({**state, **out}) == "synthesize"


def test_review_respects_configured_thresholds():
    sup = SupervisorAgent(MagicMock(), replace(SETTINGS, min_signals_per_competitor=4, max_research_retries=0))
    out = sup.review({"competitors": ["A"], "extracted_signals": sigs(A=1), "retry_counts": {"A": 0}})
    assert out["low_signal_competitors"] == []  # max_research_retries=0 disables the loop


def test_signal_counts_includes_zero():
    assert signal_counts({"competitors": ["A", "B"], "extracted_signals": sigs(A=2)}) == {"A": 2, "B": 0}


# --- rewrite -----------------------------------------------------------------

def base_state():
    return {
        "competitors": ["A", "B"], "low_signal_competitors": ["B"],
        "search_queries": {"A": ["a1"], "B": ["b1", "b2"]},
        "query_history": {"A": ["a1"], "B": ["b1", "b2"]},
        "retry_counts": {"A": 0, "B": 0},
        "snippets": {"B": [{"url": "u"}]},
        "extraction_stats": {"B": {"candidates": 3, "ungrounded": 2, "low_confidence": 1, "invalid": 0}},
    }


def test_rewrite_only_targets_low_signal_and_increments_retry():
    client = llm({"queries": ["B1", "new b", "another b", "extra"]})  # "B1" repeats b1 (case-insensitive)
    out = SupervisorAgent(client, SETTINGS).rewrite(base_state())

    assert out["search_queries"] == {"A": ["a1"], "B": ["new b", "another b", "extra"]}
    assert out["query_history"]["B"] == ["b1", "b2", "new b", "another b", "extra"]
    assert out["retry_counts"] == {"A": 0, "B": 1}
    user = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "2 ungrounded" in user and "- b1" in user


def test_rewrite_falls_back_to_unused_queries_on_failure():
    state = base_state()
    out = SupervisorAgent(llm(RuntimeError("x")), SETTINGS).rewrite(state)
    assert len(out["search_queries"]["B"]) == 3
    assert not set(out["search_queries"]["B"]) & set(state["query_history"]["B"])
    assert out["retry_counts"]["B"] == 1 and len(out["errors"]) == 1


def test_failure_reason():
    assert "no usable results" in _failure_reason(None, 0)
    assert "none contained" in _failure_reason({"candidates": 0}, 4)
    assert "1 ungrounded" in _failure_reason({"candidates": 2, "ungrounded": 1}, 4)
