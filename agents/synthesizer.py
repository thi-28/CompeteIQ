"""
Synthesizer agent — generates the structured weekly intelligence briefing.

Takes all extracted signals, calls GPT-4o for the narrative sections
(executive summary, strategic implications, signals to watch), then
assembles the final markdown document via BriefingFormatter.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from openai import OpenAI

from models.schemas import AgentState, Signal
from tools.formatter import BriefingFormatter

logger = logging.getLogger(__name__)

_SYNTHESIS_SYSTEM_PROMPT = """You are a senior competitive intelligence analyst at an AI company.
You have been given a list of structured competitive intelligence signals collected this week.
Your task is to produce three narrative sections for a weekly briefing:

1. executive_summary: 2-3 sentences summarising the overall competitive landscape this week.
   Focus on the most impactful themes — do not list every signal.

2. strategic_implications: A list of 3-5 bullet points describing what these signals mean
   for OpenAI and the broader AI market. Be specific and forward-looking.

3. signals_to_watch: A list of 2-3 developing situations that deserve close monitoring
   next week. These can be unconfirmed signals or trends that are just beginning to emerge.

Return ONLY a JSON object with these three keys. Values for strategic_implications and
signals_to_watch must be arrays of strings. Example:
{
  "executive_summary": "...",
  "strategic_implications": ["...", "..."],
  "signals_to_watch": ["...", "..."]
}"""


class SynthesizerAgent:
    """
    Generates narrative briefing sections from structured signals using GPT-4o.

    Delegates final document assembly to BriefingFormatter to ensure
    consistent markdown structure regardless of LLM output variation.
    """

    def __init__(self, openai_client: OpenAI | None = None) -> None:
        """
        Initialise the synthesizer with an OpenAI client.

        Args:
            openai_client: Optional pre-constructed client (useful in tests).
        """
        self._client = openai_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.info("SynthesizerAgent initialised — model=%s", self._model)

    def _generate_narrative(
        self, signals: list[Signal], competitors: list[str]
    ) -> dict[str, Any]:
        """
        Call GPT-4o to produce narrative briefing sections.

        Args:
            signals: All extracted Signal instances for this run.
            competitors: Ordered list of monitored competitor names.

        Returns:
            Dict with keys: executive_summary, strategic_implications, signals_to_watch.
        """
        from tools.formatter import BriefingFormatter

        signals_block = BriefingFormatter.signals_to_context_block(signals)
        user_message = (
            f"Competitors monitored: {', '.join(competitors)}\n\n"
            f"Extracted signals this week:\n{signals_block}"
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.5,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            narrative = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("Synthesizer JSON parse error: %s", exc)
            narrative = {}
        except Exception as exc:
            logger.error("Synthesizer LLM call failed: %s", exc)
            narrative = {}

        # Provide safe fallbacks
        narrative.setdefault(
            "executive_summary",
            "Competitive monitoring completed this week. "
            "See the key developments section for detailed signal breakdown.",
        )
        narrative.setdefault("strategic_implications", ["Review extracted signals for strategic context."])
        narrative.setdefault("signals_to_watch", ["Continue monitoring all tracked competitors."])

        return narrative

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Synthesize extracted signals into a structured markdown briefing.

        Args:
            state: Current pipeline state.

        Returns:
            Partial state update with ``briefing``, ``next_agent``, and ``errors``.
        """
        raw_signals_list: list[dict] = state.get("extracted_signals", [])
        competitors: list[str] = state.get("competitors", [])
        run_id: str = state.get("run_id", "unknown")
        errors: list[str] = list(state.get("errors", []))

        # Deserialise dicts back to Signal objects
        signals: list[Signal] = []
        for raw in raw_signals_list:
            try:
                signals.append(Signal.model_validate(raw))
            except Exception as exc:
                logger.warning("Synthesizer: skipping invalid signal — %s", exc)

        logger.info(
            "Synthesizer: generating briefing from %d signals.", len(signals)
        )

        try:
            narrative = self._generate_narrative(signals, competitors)
        except Exception as exc:
            msg = f"synthesizer:narrative: {exc}"
            logger.error(msg)
            errors.append(msg)
            narrative = {
                "executive_summary": "Briefing generation encountered an error. See errors list.",
                "strategic_implications": [],
                "signals_to_watch": [],
            }

        try:
            briefing_md = BriefingFormatter.format_briefing(
                signals=signals,
                competitors=competitors,
                run_id=run_id,
                executive_summary=narrative["executive_summary"],
                strategic_implications=narrative.get("strategic_implications", []),
                signals_to_watch=narrative.get("signals_to_watch", []),
                run_date=datetime.utcnow(),
            )
        except Exception as exc:
            msg = f"synthesizer:format: {exc}"
            logger.error(msg)
            errors.append(msg)
            briefing_md = f"# CompeteIQ Briefing\n\nBriefing formatting failed: {exc}"

        logger.info("Synthesizer: briefing generated (%d chars).", len(briefing_md))

        return {
            "briefing": briefing_md,
            "next_agent": "memory_write",
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

_synthesizer_agent: SynthesizerAgent | None = None


def synthesizer_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node wrapper for the SynthesizerAgent.

    The agent singleton is lazily initialised on first call so that importing
    this module does not require OPENAI_API_KEY to be set.

    Args:
        state: Shared pipeline state passed by LangGraph.

    Returns:
        Partial state update dict.
    """
    global _synthesizer_agent
    if _synthesizer_agent is None:
        _synthesizer_agent = SynthesizerAgent()
    return _synthesizer_agent.run(state)
