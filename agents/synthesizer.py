"""
Synthesizer agent: writes the narrative sections of the briefing from the
extracted signals and renders the final markdown via tools.formatter.

Strategic implications must cite signals by number ([n]); bullets with no valid
citation are dropped, so every claim in that section traces back to a signal
(and from there to a source URL).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from config import Settings, get_settings
from models.schemas import AgentState, Signal, utc_now
from tools.formatter import format_briefing, order_signals, signals_to_context_block

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a competitive-intelligence analyst writing a weekly briefing
about the AI market from a numbered list of verified signals.

Return JSON with exactly these keys:
- "executive_summary": 2-3 sentences on the most important themes. No list of every signal.
- "strategic_implications": 3-5 strings. Each MUST cite the supporting signals by number,
  e.g. "Price pressure on frontier APIs is rising [2][5]". Only cite numbers that exist.
- "signals_to_watch": 2-3 strings on developments worth tracking next week.

Use only information in the signals. Do not add facts, numbers, or companies not present."""

CITATION = re.compile(r"\[(\d+)\]")


class SynthesizerAgent:
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
        competitors = state.get("competitors", [])
        errors: list[str] = []
        signals: list[Signal] = []
        for raw in state.get("extracted_signals", []):
            try:
                signals.append(Signal.model_validate(raw))
            except ValidationError as exc:
                errors.append(f"synthesizer: invalid signal skipped: {exc.errors()[0]['msg']}")
        signals = order_signals(signals, competitors)

        narrative: dict[str, Any] = {"executive_summary": None, "strategic_implications": [], "signals_to_watch": []}
        if signals:
            try:
                narrative = self._narrative(signals, competitors)
            except Exception as exc:  # LLM/JSON failure: render without narrative, don't invent one
                errors.append(f"synthesizer:narrative: {exc}")
                logger.error(errors[-1])

        briefing = format_briefing(
            run_id=state.get("run_id", "unknown"),
            run_date=_parse_started_at(state.get("started_at")),
            competitors=competitors,
            signals=signals,
            executive_summary=narrative["executive_summary"],
            strategic_implications=narrative["strategic_implications"],
            signals_to_watch=narrative["signals_to_watch"],
            retry_counts=state.get("retry_counts", {}),
        )
        return {"briefing": briefing, "errors": errors}

    def _narrative(self, signals: list[Signal], competitors: list[str]) -> dict[str, Any]:
        user = f"Competitors: {', '.join(competitors)}\n\nSignals:\n{signals_to_context_block(signals)}"
        response = self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content or "")
        summary = data.get("executive_summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("missing executive_summary")
        return {
            "executive_summary": summary,
            "strategic_implications": filter_cited(_str_list(data.get("strategic_implications")), len(signals)),
            "signals_to_watch": _str_list(data.get("signals_to_watch")),
        }


def filter_cited(bullets: list[str], n_signals: int) -> list[str]:
    """Keep bullets whose citations are all valid signal numbers (and that cite at least one)."""
    kept = []
    for b in bullets:
        refs = [int(m) for m in CITATION.findall(b)]
        if refs and all(1 <= r <= n_signals for r in refs):
            kept.append(b)
        else:
            logger.warning("Dropping uncited/invalid implication: %r", b)
    return kept


def _str_list(value: Any) -> list[str]:
    return [v for v in value if isinstance(v, str) and v.strip()] if isinstance(value, list) else []


def _parse_started_at(value: str | None) -> datetime:
    if value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return utc_now()
