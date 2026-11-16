import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agents.extractor import ExtractorAgent, normalize_url
from config import Settings


def snip(comp, url):
    return {"competitor": comp, "url": url, "title": "t", "content": "body", "query": "q"}


def cand(url, confidence=0.9, **kw):
    data = {
        "signal_type": "product_launch", "title": "Launch", "summary": "Launched X.",
        "impact_assessment": "high", "source_url": url, "confidence": confidence,
    }
    data.update(kw)
    return data


def llm_returning(*payloads):
    """Mock OpenAI client whose successive calls return the given payloads."""
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        p if isinstance(p, Exception)
        else SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=p if isinstance(p, str) else json.dumps(p)))])
        for p in payloads
    ]
    return client


def make_agent(client=None, threshold=0.6):
    return ExtractorAgent(client=client or MagicMock(), settings=replace(Settings(), confidence_threshold=threshold))


# --- grounding guard -------------------------------------------------------

def test_discards_signal_with_url_not_in_snippets():
    agent = make_agent()
    accepted, stats = agent.filter_candidates(
        "A", [cand("https://a.com/real"), cand("https://hallucinated.com/x")], [snip("A", "https://a.com/real")]
    )
    assert [s.source_url for s in accepted] == ["https://a.com/real"]
    assert stats["ungrounded"] == 1 and stats["accepted"] == 1


def test_rejects_url_from_another_competitors_snippets():
    agent = make_agent(llm_returning({"signals": [cand("https://b.com/1")]}, {"signals": []}))
    out = agent({"snippets": {"A": [snip("A", "https://a.com/1")], "B": [snip("B", "https://b.com/1")]}})
    assert out["extracted_signals"] == []
    assert out["extraction_stats"]["A"]["ungrounded"] == 1


@pytest.mark.parametrize("variant", ["https://A.com/post/", "https://a.com/post#section", "HTTPS://a.com/post"])
def test_cosmetic_url_differences_still_ground_and_are_canonicalized(variant):
    accepted, _ = make_agent().filter_candidates("A", [cand(variant)], [snip("A", "https://a.com/post")])
    assert len(accepted) == 1
    assert accepted[0].source_url == "https://a.com/post"  # stored as the collected URL


def test_query_string_is_significant():
    accepted, stats = make_agent().filter_candidates(
        "A", [cand("https://a.com/p?id=2")], [snip("A", "https://a.com/p?id=1")]
    )
    assert accepted == [] and stats["ungrounded"] == 1


def test_normalize_url():
    assert normalize_url(" HTTPS://Ex.COM/a/b/#frag ") == "https://ex.com/a/b"


# --- confidence + schema ---------------------------------------------------

def test_confidence_threshold_is_configurable_and_inclusive():
    snippets = [snip("A", "https://a.com")]
    cands = [cand("https://a.com", 0.59), cand("https://a.com", 0.6, title="Edge")]
    accepted, stats = make_agent(threshold=0.6).filter_candidates("A", cands, snippets)
    assert [s.title for s in accepted] == ["Edge"]
    assert stats["low_confidence"] == 1

    accepted, _ = make_agent(threshold=0.9).filter_candidates("A", cands, snippets)
    assert accepted == []


def test_invalid_candidates_counted_and_competitor_forced():
    snippets = [snip("A", "https://a.com")]
    cands = ["not a dict", cand("https://a.com", signal_type="rumor"), cand("https://a.com", competitor="Evil")]
    accepted, stats = make_agent().filter_candidates("A", cands, snippets)
    assert stats == {"candidates": 3, "invalid": 2, "ungrounded": 0, "low_confidence": 0, "accepted": 1}
    assert accepted[0].competitor == "A"
    assert accepted[0].date_detected.tzinfo is not None


# --- run() -----------------------------------------------------------------

def test_run_calls_llm_per_competitor_with_numbered_snippets_and_memory():
    client = llm_returning({"signals": [cand("https://a.com/1")]})
    agent = make_agent(client)
    out = agent({
        "snippets": {"A": [snip("A", "https://a.com/1")]},
        "memory_context": {"A": ["Old launch"]},
    })

    kwargs = client.chat.completions.create.call_args.kwargs
    user_msg = kwargs["messages"][1]["content"]
    assert kwargs["model"] == "gpt-4o-mini"
    assert "[S1] URL: https://a.com/1" in user_msg and "- Old launch" in user_msg
    assert len(out["extracted_signals"]) == 1
    assert out["extracted_signals"][0]["competitor"] == "A"
    assert out["errors"] == []


def test_llm_failure_and_bad_json_are_non_fatal():
    client = llm_returning(RuntimeError("api down"), "not json")
    out = make_agent(client)({"snippets": {"A": [snip("A", "https://a.com")], "B": [snip("B", "https://b.com")]}})
    assert out["extracted_signals"] == []
    assert len(out["errors"]) == 2


def test_competitor_without_snippets_skips_llm():
    client = MagicMock()
    out = make_agent(client)({"snippets": {"A": []}})
    client.chat.completions.create.assert_not_called()
    assert out["extraction_stats"]["A"]["accepted"] == 0


def test_reextraction_replaces_only_targeted_competitors():
    client = llm_returning({"signals": [cand("https://a.com/new"), cand("https://a.com/old", title="Again")]})
    prior_b = {"competitor": "B", "source_url": "https://b.com", "title": "B sig"}
    prior_a = {"competitor": "A", "source_url": "https://a.com/old", "title": "A sig"}
    state = {
        "snippets": {"A": [snip("A", "https://a.com/old"), snip("A", "https://a.com/new")], "B": [snip("B", "https://b.com")]},
        "low_signal_competitors": ["A"],
        "extracted_signals": [prior_a, prior_b],
        "extraction_stats": {"B": {"accepted": 1}},
    }
    out = make_agent(client)(state)

    assert client.chat.completions.create.call_count == 1
    comps = [s["competitor"] for s in out["extracted_signals"]]
    assert comps.count("B") == 1 and comps.count("A") == 2
    assert prior_b in out["extracted_signals"] and prior_a not in out["extracted_signals"]
    assert out["extraction_stats"]["B"] == {"accepted": 1}
