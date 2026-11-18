"""
Graph-level tests: the real compiled LangGraph with a fake LLM (dispatching on
each agent's system prompt), a fake search tool, a deterministic embedder, and
real SQLite + ChromaDB in a temp dir.
"""

import hashlib
import json
import random
import re
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agents import extractor as extractor_mod
from agents import supervisor as supervisor_mod
from agents import synthesizer as synthesizer_mod
from agents.collector import CollectorAgent
from agents.extractor import ExtractorAgent
from agents.supervisor import SupervisorAgent
from agents.synthesizer import SynthesizerAgent
from config import Settings
from main import Dependencies, build_graph, run_pipeline
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory


def hash_embed(texts):
    """Pseudo-random unit vectors seeded by text: unrelated texts are ~orthogonal."""
    out = []
    for t in texts:
        rng = random.Random(hashlib.sha256(t.encode()).hexdigest())
        v = [rng.gauss(0, 1) for _ in range(64)]
        n = sum(x * x for x in v) ** 0.5
        out.append([x / n for x in v])
    return out


class FakeLLM:
    """Routes chat.completions.create by system prompt to per-agent handlers."""

    def __init__(self, *, hallucinate_for=()):
        self.calls: list[tuple[str, str]] = []
        self.hallucinate_for = set(hallucinate_for)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.rewrite_n = 0

    def create(self, *, model, messages, **_):
        system, user = messages[0]["content"], messages[1]["content"]
        if system == supervisor_mod.PLAN_PROMPT:
            agent, payload = "plan", self._plan(user)
        elif system == supervisor_mod.REWRITE_PROMPT:
            agent, payload = "rewrite", self._rewrite(user)
        elif system == extractor_mod.SYSTEM_PROMPT:
            agent, payload = "extract", self._extract(user)
        elif system == synthesizer_mod.SYSTEM_PROMPT:
            agent, payload = "synthesize", {"executive_summary": "Test summary.",
                                            "strategic_implications": ["Theme [1]"], "signals_to_watch": ["Watch"]}
        else:
            raise AssertionError("unexpected prompt")
        comp = re.search(r"Competitors?: (.+)", user).group(1) if "Competitor" in user else ""
        self.calls.append((agent, comp.strip()))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])

    def _plan(self, user):
        comps = re.search(r"Competitors: (.+)", user).group(1).split(", ")
        return {c: [f"{c} initial"] for c in comps}

    def _rewrite(self, user):
        comp = re.search(r"Competitor: (.+)", user).group(1)
        self.rewrite_n += 1
        return {"queries": [f"{comp} retry {self.rewrite_n}"]}

    def _extract(self, user):
        comp = re.search(r"COMPETITOR: (.+)", user).group(1)
        urls = re.findall(r"\[S\d+\] URL: (\S+)", user)
        if comp in self.hallucinate_for:
            urls = [u.replace("https://", "https://fake-") for u in urls]
        return {"signals": [{
            "signal_type": "product_launch", "title": f"{comp} development {i}",
            "summary": f"Details about {u}.", "impact_assessment": "medium",
            "source_url": u, "confidence": 0.9,
        } for i, u in enumerate(urls)]}


def search_tool(results_by_query):
    tool = MagicMock()

    def search(query):
        return [{"title": f"title {u}", "url": u, "content": f"content {u}", "score": 0.5}
                for u in results_by_query.get(query, [])]

    tool.search.side_effect = search
    return tool


@pytest.fixture
def make_deps(tmp_path):
    def _make(results_by_query, llm, **settings_kw):
        settings = replace(Settings(), data_dir=str(tmp_path), min_signals_per_competitor=2,
                           max_research_retries=2, **settings_kw)
        return Dependencies(
            supervisor=SupervisorAgent(llm, settings),
            collector=CollectorAgent(search_tool(results_by_query), settings),
            extractor=ExtractorAgent(llm, settings),
            synthesizer=SynthesizerAgent(llm, settings),
            semantic=SemanticMemory(str(tmp_path / "test.db")),
            episodic=EpisodicMemory(str(tmp_path / "chroma"), embedder=hash_embed, settings=settings),
            settings=settings,
        )
    return _make


def test_happy_path_no_retries(make_deps, tmp_path):
    llm = FakeLLM()
    deps = make_deps({"A initial": ["https://a.com/1", "https://a.com/2"]}, llm)

    result = run_pipeline(["A"], deps)

    assert result.success and result.signal_count == 2 and result.research_retries == {}
    assert [a for a, _ in llm.calls] == ["plan", "extract", "synthesize"]
    assert len(deps.semantic.get_signals()) == 2
    assert deps.episodic.count("A") == 2
    briefing = deps.semantic.get_briefing(result.run_id)
    assert "Test summary." in briefing["content"] and "[source](https://a.com/1)" in briefing["content"]
    assert (tmp_path / "briefings" / f"{result.run_id}.md").read_text() == briefing["content"]
    assert deps.semantic.get_run(result.run_id)["signal_count"] == 2


def test_low_signal_competitor_is_re_searched_until_satisfied(make_deps):
    llm = FakeLLM()
    deps = make_deps({
        "A initial": ["https://a.com/1", "https://a.com/2"],
        "B initial": ["https://b.com/1"],          # only 1 signal -> retry
        "B retry 1": ["https://b.com/2"],          # now 2 -> proceed
    }, llm)

    result = run_pipeline(["A", "B"], deps)

    assert result.research_retries == {"B": 1}
    assert result.signal_count == 4
    assert [c for a, c in llm.calls if a == "rewrite"] == ["B"]
    extract_calls = [a for a, _ in llm.calls].count("extract")
    assert extract_calls == 3  # A, B, then B again only
    b_urls = {s["source_url"] for s in deps.semantic.get_signals(competitor="B")}
    assert b_urls == {"https://b.com/1", "https://b.com/2"}


def test_retries_are_capped_at_max(make_deps):
    llm = FakeLLM()
    deps = make_deps({"A initial": ["https://a.com/1", "https://a.com/2"]}, llm)  # B never finds anything

    result = run_pipeline(["A", "B"], deps)

    assert result.success
    assert result.research_retries == {"B": 2}
    assert [a for a, _ in llm.calls].count("rewrite") == 2
    content = deps.semantic.get_briefing(result.run_id)["content"]
    assert "_No grounded signals found after 2 re-searches._" in content


def test_hallucinated_urls_trigger_retry_and_never_persist(make_deps):
    llm = FakeLLM(hallucinate_for={"A"})
    deps = make_deps({"A initial": ["https://a.com/1", "https://a.com/2"]}, llm)

    result = run_pipeline(["A"], deps)

    assert result.signal_count == 0
    assert result.research_retries == {"A": 2}
    assert deps.semantic.get_signals() == []
    assert deps.episodic.count() == 0


def test_second_run_sees_first_runs_memory(make_deps):
    llm = FakeLLM()
    deps = make_deps({"A initial": ["https://a.com/1", "https://a.com/2"]}, llm)
    run_pipeline(["A"], deps)
    seen = []
    original = deps.extractor._call_llm
    deps.extractor._call_llm = lambda comp, snips, memory: seen.append(memory) or original(comp, snips, memory)

    run_pipeline(["A"], deps)

    assert len(seen[0]) == 2 and "A development" in seen[0][0]
    assert deps.episodic.count("A") == 2  # second run's identical signals deduped


def test_fatal_error_still_records_failed_run(make_deps):
    deps = make_deps({}, FakeLLM())
    deps.collector.run = MagicMock(side_effect=RuntimeError("boom"))
    result = run_pipeline(["A"], deps)
    assert not result.success
    row = deps.semantic.get_run(result.run_id)
    assert row["success"] is False and row["error_count"] == 1


def test_run_with_no_search_results_at_all_is_failed(make_deps):
    deps = make_deps({}, FakeLLM())  # every query returns nothing (e.g. bad Tavily key)
    result = run_pipeline(["A", "B"], deps)
    assert not result.success and result.signal_count == 0
    assert deps.semantic.get_run(result.run_id)["success"] is False
    assert result.briefing_id  # the briefing still records what happened


def test_graph_structure(make_deps):
    graph = build_graph(make_deps({}, FakeLLM())).get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("review", "rewrite") in edges and ("rewrite", "collect") in edges
    assert ("review", "synthesize") in edges and ("memory_write", "__end__") in edges


def test_vector_store_failures_are_non_fatal(make_deps):
    deps = make_deps({"A initial": ["https://a.com/1", "https://a.com/2"]}, FakeLLM())
    deps.episodic.retrieve = MagicMock(side_effect=RuntimeError("chroma read down"))
    deps.episodic.store = MagicMock(side_effect=RuntimeError("chroma write down"))

    result = run_pipeline(["A"], deps)

    assert result.success and result.signal_count == 2
    assert result.error_count == 2
    assert len(deps.semantic.get_signals()) == 2  # SQLite (source of truth) still written
