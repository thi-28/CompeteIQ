"""
Supervisor agent: plans search queries and makes the one real control-flow
decision in the graph.

After extraction, `review` counts signals per competitor. Any competitor below
MIN_SIGNALS_PER_COMPETITOR that still has retries left (MAX_RESEARCH_RETRIES,
tracked in state.retry_counts) is sent to `rewrite`, which generates new
queries informed by why the previous ones failed. Only those competitors are
re-collected and re-extracted.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Literal

from openai import OpenAI

from config import Settings, get_settings
from models.schemas import AgentState, utc_now

logger = logging.getLogger(__name__)

QUERIES_PER_COMPETITOR = 3

PLAN_PROMPT = """You plan web searches for competitive intelligence on AI companies.
For each competitor, write exactly 3 search queries likely to surface news from the
last 2 weeks: product launches, pricing changes, partnerships, research releases,
executive moves, market expansion. Include the month and year where it helps recency.
Return JSON: {"<competitor>": ["q1", "q2", "q3"], ...} using the exact competitor names given."""

REWRITE_PROMPT = """Previous web searches for a competitor did not yield enough usable signals.
Write exactly 3 NEW search queries that take a different angle (different wording, sources
like official blogs or press releases, specific product names, or broader news terms).
Do not repeat or lightly rephrase any previous query.
Return JSON: {"queries": ["q1", "q2", "q3"]}."""

FALLBACK_INITIAL = [
    "{c} product launch announcement {month}",
    "{c} partnership OR acquisition news {month}",
    "{c} pricing OR funding OR executive news {month}",
]
FALLBACK_RETRY = [
    "{c} AI news this week",
    "{c} official blog announcement {month}",
    "{c} press release {month}",
    "{c} latest model release",
    "{c} hires OR appoints executive",
    "{c} expands new market {month}",
]


class SupervisorAgent:
    def __init__(self, client: Any | None = None, settings: Settings | None = None) -> None:
        self._client = client
        self.settings = settings or get_settings()

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    # --- node: plan -------------------------------------------------------

    def plan(self, state: AgentState) -> dict[str, Any]:
        competitors = state["competitors"]
        errors: list[str] = []
        try:
            data = self._chat(PLAN_PROMPT, f"Competitors: {', '.join(competitors)}\nToday: {_today()}")
        except Exception as exc:
            errors.append(f"supervisor:plan: {exc}; using fallback queries")
            data = {}

        queries = {}
        for comp in competitors:
            q = _clean_queries(data.get(comp) if isinstance(data, dict) else None)
            queries[comp] = q or _fallback(FALLBACK_INITIAL, comp, used=set())
        return {
            "search_queries": queries,
            "query_history": {c: list(q) for c, q in queries.items()},
            "retry_counts": {c: 0 for c in competitors},
            "low_signal_competitors": [],
            "errors": errors,
        }

    # --- node: review (the decision) ---------------------------------------

    def review(self, state: AgentState) -> dict[str, Any]:
        counts = signal_counts(state)
        retries = state.get("retry_counts", {})
        low = [
            c for c in state["competitors"]
            if counts[c] < self.settings.min_signals_per_competitor
            and retries.get(c, 0) < self.settings.max_research_retries
        ]
        exhausted = [
            c for c in state["competitors"]
            if counts[c] < self.settings.min_signals_per_competitor and c not in low
        ]
        if low:
            logger.info("Supervisor: re-searching %s (counts=%s)", low, dict(counts))
        if exhausted:
            logger.info("Supervisor: retries exhausted for %s; proceeding", exhausted)
        return {"low_signal_competitors": low}

    @staticmethod
    def route(state: AgentState) -> Literal["rewrite", "synthesize"]:
        return "rewrite" if state.get("low_signal_competitors") else "synthesize"

    # --- node: rewrite -----------------------------------------------------

    def rewrite(self, state: AgentState) -> dict[str, Any]:
        targets = state["low_signal_competitors"]
        history = {c: list(q) for c, q in state.get("query_history", {}).items()}
        queries = dict(state.get("search_queries", {}))
        retries = dict(state.get("retry_counts", {}))
        stats = state.get("extraction_stats", {})
        errors: list[str] = []

        for comp in targets:
            tried = history.get(comp, [])
            reason = _failure_reason(stats.get(comp), len(state.get("snippets", {}).get(comp, [])))
            user = (
                f"Competitor: {comp}\nToday: {_today()}\nWhy it failed: {reason}\n"
                "Previous queries:\n" + "\n".join(f"- {q}" for q in tried)
            )
            new: list[str] = []
            try:
                data = self._chat(REWRITE_PROMPT, user)
                new = _clean_queries(data.get("queries"), exclude=tried)
            except Exception as exc:
                errors.append(f"supervisor:rewrite:{comp}: {exc}; using fallback queries")
            if not new:
                new = _fallback(FALLBACK_RETRY, comp, used={q.lower() for q in tried})
            queries[comp] = new
            history[comp] = tried + new
            retries[comp] = retries.get(comp, 0) + 1
            logger.info("Supervisor: retry %d for %s (%s): %s", retries[comp], comp, reason, new)

        return {"search_queries": queries, "query_history": history, "retry_counts": retries, "errors": errors}

    # --- helpers -----------------------------------------------------------

    def _chat(self, system: str, user: str) -> Any:
        response = self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content or "{}")


def signal_counts(state: AgentState) -> Counter:
    counts: Counter = Counter({c: 0 for c in state.get("competitors", [])})
    counts.update(s["competitor"] for s in state.get("extracted_signals", []))
    return counts


def _failure_reason(stats: dict[str, int] | None, n_snippets: int) -> str:
    if n_snippets == 0:
        return "searches returned no usable results"
    if not stats or stats.get("candidates", 0) == 0:
        return f"{n_snippets} results but none contained a concrete development"
    return (
        f"{stats.get('candidates', 0)} candidate signals: {stats.get('ungrounded', 0)} ungrounded, "
        f"{stats.get('low_confidence', 0)} low confidence, {stats.get('invalid', 0)} invalid"
    )


def _clean_queries(value: Any, exclude: list[str] = ()) -> list[str]:
    """Strip, dedup (case-insensitive), drop excluded, THEN cap, so a repeated
    query doesn't consume one of the slots."""
    if not isinstance(value, list):
        return []
    seen = {q.lower() for q in exclude}
    out: list[str] = []
    for q in value:
        if isinstance(q, str) and q.strip() and q.strip().lower() not in seen:
            seen.add(q.strip().lower())
            out.append(q.strip())
    return out[:QUERIES_PER_COMPETITOR]


def _fallback(templates: list[str], comp: str, used: set[str]) -> list[str]:
    month = utc_now().strftime("%B %Y")
    candidates = [t.format(c=comp, month=month) for t in templates]
    fresh = [q for q in candidates if q.lower() not in used]
    return (fresh or candidates)[:QUERIES_PER_COMPETITOR]


def _today() -> str:
    return utc_now().strftime("%Y-%m-%d")
