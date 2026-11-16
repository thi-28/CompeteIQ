from dataclasses import replace
from unittest.mock import MagicMock

from agents.collector import CollectorAgent
from config import Settings
from tools.search import SearchError


def result(url, score=0.5, content="body", title="t"):
    return {"title": title, "url": url, "content": content, "score": score}


def make_agent(responses, cap=10, **kw):
    """responses: dict query -> list of results, or an Exception to raise."""
    tool = MagicMock()

    def fake_search(query):
        r = responses[query]
        if isinstance(r, Exception):
            raise r
        return r

    tool.search.side_effect = fake_search
    settings = replace(Settings(), max_snippets_per_competitor=cap)
    return CollectorAgent(search_tool=tool, settings=settings, **kw), tool


def test_collects_structured_snippets_per_competitor():
    agent, _ = make_agent({"a1": [result("https://a.com/1")], "m1": [result("https://m.com/1")]})
    out = agent({"search_queries": {"Anthropic": ["a1"], "Meta": ["m1"]}})

    assert set(out["snippets"]) == {"Anthropic", "Meta"}
    snip = out["snippets"]["Anthropic"][0]
    assert snip == {
        "competitor": "Anthropic", "url": "https://a.com/1",
        "title": "t", "content": "body", "query": "a1",
    }
    assert out["errors"] == []


def test_dedups_urls_across_queries_keeping_highest_score():
    agent, _ = make_agent({
        "q1": [result("https://x.com", score=0.3)],
        "q2": [result("https://x.com", score=0.9), result("https://y.com", score=0.5)],
    })
    snips = agent({"search_queries": {"A": ["q1", "q2"]}})["snippets"]["A"]

    assert [s["url"] for s in snips] == ["https://x.com", "https://y.com"]
    assert snips[0]["query"] == "q2"


def test_snippet_cap_keeps_top_scored():
    results = [result(f"https://s.com/{i}", score=i / 10) for i in range(6)]
    agent, _ = make_agent({"q": results}, cap=3)
    snips = agent({"search_queries": {"A": ["q"]}})["snippets"]["A"]

    assert [s["url"] for s in snips] == ["https://s.com/5", "https://s.com/4", "https://s.com/3"]


def test_truncates_content():
    agent, _ = make_agent({"q": [result("https://a.com", content="x" * 50)]}, max_content_chars=10)
    assert agent({"search_queries": {"A": ["q"]}})["snippets"]["A"][0]["content"] == "x" * 10


def test_failed_query_is_recorded_and_others_continue():
    agent, _ = make_agent({"bad": SearchError("boom"), "good": [result("https://ok.com")]})
    out = agent({"search_queries": {"A": ["bad", "good"]}})

    assert [s["url"] for s in out["snippets"]["A"]] == ["https://ok.com"]
    assert len(out["errors"]) == 1 and "boom" in out["errors"][0]


def test_competitor_with_no_snippets_is_flagged():
    agent, _ = make_agent({"q": []})
    out = agent({"search_queries": {"A": ["q"]}})
    assert out["snippets"]["A"] == []
    assert out["errors"] == ["collector:A: no snippets collected"]


def test_recollection_targets_only_low_signal_and_appends_new_urls():
    agent, tool = make_agent({
        "a-retry": [result("https://a.com/old"), result("https://a.com/new")],
    })
    state = {
        "search_queries": {"A": ["a-retry"], "B": ["b-orig"]},
        "low_signal_competitors": ["A"],
        "snippets": {
            "A": [{"competitor": "A", "url": "https://a.com/old", "title": "", "content": "c", "query": "a-orig"}],
            "B": [{"competitor": "B", "url": "https://b.com", "title": "", "content": "c", "query": "b-orig"}],
        },
    }
    out = agent(state)

    tool.search.assert_called_once_with("a-retry")  # B is not re-searched
    assert [s["url"] for s in out["snippets"]["A"]] == ["https://a.com/old", "https://a.com/new"]
    assert out["snippets"]["B"] == state["snippets"]["B"]
    assert state["snippets"]["A"][0]["url"] == "https://a.com/old" and len(state["snippets"]["A"]) == 1  # input not mutated
