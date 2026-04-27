"""
Extractor agent — converts raw web text into structured Signal objects.

Uses GPT-4o with JSON structured output to parse competitive intelligence
from the raw snippets collected by the collector agent. Memory context
from previous runs is injected so the model can distinguish genuinely new
signals from already-known facts.

Signals with confidence < 0.6 are discarded. Potential duplicates of
memory-context entries are also filtered out before returning.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from openai import OpenAI

from models.schemas import AgentState, Signal

logger = logging.getLogger(__name__)

# Minimum confidence threshold — signals below this are discarded
_MIN_CONFIDENCE = 0.6

_EXTRACTOR_SYSTEM_PROMPT = """You are a competitive intelligence extraction specialist.
Your task is to read raw web search snippets about AI companies and extract structured
intelligence signals.

For each distinct, verifiable piece of intelligence you find, extract a signal with:
- competitor: exact company name from the provided list
- signal_type: one of [product_launch, pricing_change, partnership, research_release, executive_move, market_expansion, other]
- title: short, descriptive title (max 15 words)
- summary: clear explanation of what happened and why it matters (2-4 sentences)
- impact_assessment: "low", "medium", or "high" — based on competitive significance
- source_url: the URL this signal came from (use the URL from the snippet)
- confidence: float 0.0-1.0 — how confident you are this is real and accurate

IMPORTANT RULES:
1. Only extract signals explicitly supported by the provided text — do not hallucinate.
2. Skip signals that match the PREVIOUSLY KNOWN context (avoid duplicates).
3. Assign confidence < 0.6 for unverified rumours; these will be filtered out.
4. Return a JSON object with a single key "signals" containing a list of signal objects.
5. If no new signals are found, return {"signals": []}.
6. The date_detected field should be today's UTC date in ISO format."""


class ExtractorAgent:
    """
    Extracts structured Signal instances from raw text using GPT-4o.

    Processes each competitor's raw snippets in a single batched LLM call
    (with memory context as additional system context) to minimise API round
    trips while keeping prompts focused.
    """

    def __init__(self, openai_client: OpenAI | None = None) -> None:
        """
        Initialise the extractor with an OpenAI client.

        Args:
            openai_client: Optional pre-constructed client (useful in tests).
        """
        self._client = openai_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.info("ExtractorAgent initialised — model=%s", self._model)

    def _extract_for_competitor(
        self,
        competitor: str,
        snippets: list[str],
        memory_context: list[str],
    ) -> list[Signal]:
        """
        Call GPT-4o to extract signals for a single competitor.

        Args:
            competitor: The competitor name being processed.
            snippets: Raw text snippets from the collector.
            memory_context: Previously known signal summaries for context.

        Returns:
            List of validated Signal objects above the confidence threshold.
        """
        if not snippets:
            logger.debug("No snippets for %s — skipping extraction.", competitor)
            return []

        memory_block = (
            "\n".join(f"- {c}" for c in memory_context)
            if memory_context
            else "None available."
        )
        snippets_block = "\n\n---\n\n".join(snippets[:10])  # cap at 10 snippets per call

        user_message = (
            f"COMPETITOR: {competitor}\n\n"
            f"PREVIOUSLY KNOWN SIGNALS (do not re-extract these):\n{memory_block}\n\n"
            f"RAW SEARCH SNIPPETS:\n{snippets_block}"
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _EXTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or '{"signals": []}'
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            # Malformed JSON is logged and treated as zero signals; not fatal.
            logger.error("Extractor JSON parse error for %s: %s", competitor, exc)
            return []
        except Exception:
            # LLM / network errors propagate to run() so they appear in state.errors.
            raise

        signals: list[Signal] = []
        today_iso = datetime.utcnow().isoformat()

        for raw_sig in data.get("signals", []):
            try:
                # Ensure date_detected is set
                raw_sig.setdefault("date_detected", today_iso)
                # Enforce competitor field matches expected value
                raw_sig["competitor"] = competitor
                signal = Signal.model_validate(raw_sig)
                if signal.confidence >= _MIN_CONFIDENCE:
                    signals.append(signal)
                else:
                    logger.debug(
                        "Discarding low-confidence signal (%.2f): %s",
                        signal.confidence,
                        signal.title,
                    )
            except Exception as exc:
                logger.warning("Signal validation failed: %s — %s", raw_sig, exc)

        return signals

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Extract signals for all competitors and populate ``extracted_signals``.

        Args:
            state: Current pipeline state.

        Returns:
            Partial state update with ``extracted_signals``, ``next_agent``,
            and ``errors``.
        """
        raw_signals: dict[str, list[str]] = state.get("raw_signals", {})
        memory_context: list[str] = state.get("memory_context", [])
        errors: list[str] = list(state.get("errors", []))
        all_signals: list[Signal] = []

        for competitor, snippets in raw_signals.items():
            logger.info(
                "Extractor: processing %d snippets for '%s'.",
                len(snippets),
                competitor,
            )
            try:
                extracted = self._extract_for_competitor(
                    competitor, snippets, memory_context
                )
                all_signals.extend(extracted)
                logger.info(
                    "Extractor: extracted %d signals for '%s'.",
                    len(extracted),
                    competitor,
                )
            except Exception as exc:
                msg = f"extractor:{competitor}: {exc}"
                logger.error(msg)
                errors.append(msg)

        logger.info("Extractor: total signals extracted = %d.", len(all_signals))

        # Serialise to dicts so LangGraph state remains JSON-serialisable
        serialised = [s.model_dump(mode="json") for s in all_signals]

        return {
            "extracted_signals": serialised,
            "next_agent": "synthesizer",
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

_extractor_agent: ExtractorAgent | None = None


def extractor_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node wrapper for the ExtractorAgent.

    The agent singleton is lazily initialised on first call so that importing
    this module does not require OPENAI_API_KEY to be set.

    Args:
        state: Shared pipeline state passed by LangGraph.

    Returns:
        Partial state update dict.
    """
    global _extractor_agent
    if _extractor_agent is None:
        _extractor_agent = ExtractorAgent()
    return _extractor_agent.run(state)
