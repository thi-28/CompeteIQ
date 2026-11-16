"""
Extractor agent: turns collected snippets into validated Signal objects.

Every candidate signal passes three gates, in order:
1. Schema: must validate as a Signal.
2. Grounding: source_url must match a URL in *this competitor's* collected
   snippets. LLMs will happily invent plausible URLs; this guard drops them.
3. Confidence: must be >= CONFIDENCE_THRESHOLD.

Per-competitor drop counts are returned in `extraction_stats` for observability
and evals.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from config import Settings, get_settings
from models.schemas import AgentState, Signal
from tools.urls import normalize_url

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You extract competitive-intelligence signals from web search snippets.

Each snippet is labelled [S<n>] with its URL. For each distinct, concrete development
about the competitor that is explicitly supported by a snippet, return:
- signal_type: one of product_launch, pricing_change, partnership, research_release,
  executive_move, market_expansion, other
- title: short title (max 15 words)
- summary: 2-3 sentences on what happened and why it matters
- impact_assessment: low, medium, or high
- source_url: the URL of the snippet that supports it, copied EXACTLY
- confidence: 0.0-1.0, how well the snippet supports the claim

Rules:
- Only use facts stated in the snippets. Never invent URLs or details.
- Skip anything already listed under PREVIOUSLY KNOWN.
- Rumours or speculation get confidence below 0.5.
Respond with JSON: {"signals": [...]}. Return {"signals": []} if nothing qualifies."""


class ExtractorAgent:
    def __init__(self, client: Any | None = None, settings: Settings | None = None) -> None:
        self._client = client
        self.settings = settings or get_settings()

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def __call__(self, state: AgentState) -> dict[str, Any]:
        return self.run(state)

    def run(self, state: AgentState) -> dict[str, Any]:
        snippets_by_comp = state.get("snippets", {})
        memory = state.get("memory_context", {})
        retry_targets = state.get("low_signal_competitors") or []
        targets = retry_targets or list(snippets_by_comp)

        # In re-extraction mode keep signals for competitors we aren't redoing.
        kept = [s for s in state.get("extracted_signals", []) if s["competitor"] not in targets]
        new_signals: list[dict[str, Any]] = []
        stats: dict[str, dict[str, int]] = dict(state.get("extraction_stats", {}))
        errors: list[str] = []

        for comp in targets:
            snippets = snippets_by_comp.get(comp, [])
            if not snippets:
                stats[comp] = _empty_stats()
                continue
            try:
                candidates = self._call_llm(comp, snippets, memory.get(comp, []))
            except Exception as exc:  # LLM/network/JSON failure is non-fatal per competitor
                errors.append(f"extractor:{comp}: {exc}")
                logger.error(errors[-1])
                stats[comp] = _empty_stats()
                continue
            accepted, stats[comp] = self.filter_candidates(comp, candidates, snippets)
            new_signals.extend(s.model_dump(mode="json") for s in accepted)
            logger.info("Extractor: %s %s", comp, stats[comp])

        return {
            "extracted_signals": kept + new_signals,
            "extraction_stats": stats,
            "errors": errors,
        }

    def filter_candidates(
        self, competitor: str, candidates: list[Any], snippets: list[dict[str, Any]]
    ) -> tuple[list[Signal], dict[str, int]]:
        """Apply schema, grounding, and confidence gates. Pure: no I/O."""
        allowed = {normalize_url(s["url"]): s["url"] for s in snippets}
        stats = _empty_stats()
        stats["candidates"] = len(candidates)
        accepted: list[Signal] = []

        for raw in candidates:
            if not isinstance(raw, dict):
                stats["invalid"] += 1
                continue
            raw = {**raw, "competitor": competitor}  # never trust the model's attribution
            raw.pop("date_detected", None)  # we stamp detection time ourselves
            try:
                signal = Signal.model_validate(raw)
            except ValidationError:
                stats["invalid"] += 1
                continue

            canonical = allowed.get(normalize_url(signal.source_url))
            if canonical is None:
                stats["ungrounded"] += 1
                logger.warning("Dropping ungrounded signal %r (url=%s)", signal.title, signal.source_url)
                continue
            if signal.confidence < self.settings.confidence_threshold:
                stats["low_confidence"] += 1
                continue

            accepted.append(signal.model_copy(update={"source_url": canonical}))

        stats["accepted"] = len(accepted)
        return accepted, stats

    def _call_llm(
        self, competitor: str, snippets: list[dict[str, Any]], memory: list[str]
    ) -> list[Any]:
        snippet_block = "\n\n".join(
            f"[S{i}] URL: {s['url']}\nTitle: {s.get('title', '')}\n{s['content']}"
            for i, s in enumerate(snippets, 1)
        )
        known = "\n".join(f"- {m}" for m in memory) or "None."
        user = (
            f"COMPETITOR: {competitor}\n\nPREVIOUSLY KNOWN:\n{known}\n\n"
            f"SNIPPETS:\n{snippet_block}"
        )
        response = self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content or "{}")
        signals = data.get("signals", []) if isinstance(data, dict) else []
        return signals if isinstance(signals, list) else []


def _empty_stats() -> dict[str, int]:
    return {"candidates": 0, "invalid": 0, "ungrounded": 0, "low_confidence": 0, "accepted": 0}
