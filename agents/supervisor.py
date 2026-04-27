"""
Supervisor agent — orchestrates the pipeline and generates search queries.

Responsibilities:
1. On first invocation: generates 3 targeted search queries per competitor
   using GPT-4o and routes to the collector.
2. Controls sequential routing between collector → memory_read → extractor
   → synthesizer → memory_write via the ``next_agent`` state field.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI

from models.schemas import AgentState

logger = logging.getLogger(__name__)

_QUERY_SYSTEM_PROMPT = """You are a competitive intelligence analyst.
Given a list of AI/tech competitors, generate exactly 3 highly targeted web search queries
for each competitor that would surface recent product launches, pricing changes, partnerships,
research releases, executive moves, or market expansion news from the past 2 weeks.

Return ONLY a JSON object with competitor names as keys and lists of 3 query strings as values.
Example:
{
  "Anthropic": [
    "Anthropic new model release 2024",
    "Anthropic pricing update API",
    "Anthropic partnership announcement"
  ]
}
Do not include any explanation or markdown — raw JSON only."""


class SupervisorAgent:
    """
    Routes the pipeline and generates search queries using GPT-4o.

    The supervisor is called multiple times during a single pipeline run.
    It uses the ``next_agent`` field to determine which phase it is in and
    emits the correct routing decision for LangGraph's conditional edges.
    """

    def __init__(self, openai_client: OpenAI | None = None) -> None:
        """
        Initialise the supervisor with an OpenAI client.

        Args:
            openai_client: Optional pre-constructed client (useful in tests).
        """
        self._client = openai_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.info("SupervisorAgent initialised — model=%s", self._model)

    def _generate_queries(self, competitors: list[str]) -> dict[str, list[str]]:
        """
        Call GPT-4o to generate 3 search queries per competitor.

        Args:
            competitors: List of competitor name strings.

        Returns:
            Dict mapping competitor name → list of query strings.

        Raises:
            ValueError: If the LLM response cannot be parsed as JSON.
        """
        user_msg = f"Competitors: {json.dumps(competitors)}"
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            queries: dict[str, list[str]] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON for queries: {raw}") from exc

        # Ensure every competitor has entries even if the model missed some
        for comp in competitors:
            if comp not in queries:
                queries[comp] = [
                    f"{comp} latest AI news",
                    f"{comp} product announcement",
                    f"{comp} funding or partnership",
                ]
        return queries

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Execute one supervisor step and return state updates.

        The supervisor is a pure router — on the first call it also
        generates search queries. Routing decisions:

        - No ``next_agent`` set → generate queries, route to ``collector``
        - ``next_agent == "extractor"`` → route forward (memory read done)
        - ``next_agent == "synthesizer"`` → route forward (extraction done)
        - ``next_agent == "memory_write"`` → route forward (synthesis done)

        Args:
            state: Current pipeline state dict.

        Returns:
            Partial state update dict with ``search_queries`` (if first call)
            and updated ``next_agent``.
        """
        errors: list[str] = list(state.get("errors", []))
        current_next = state.get("next_agent", "")

        # ----------------------------------------------------------------
        # First call — generate queries and route to collector
        # ----------------------------------------------------------------
        if not current_next or current_next == "supervisor":
            competitors = state.get("competitors", [])
            logger.info(
                "Supervisor: generating queries for %d competitors.", len(competitors)
            )
            try:
                queries = self._generate_queries(competitors)
                logger.info(
                    "Generated %d query sets.", len(queries)
                )
            except Exception as exc:
                logger.error("Query generation failed: %s", exc)
                errors.append(f"supervisor:query_generation: {exc}")
                # Fallback — one generic query per competitor
                queries = {
                    c: [f"{c} AI news", f"{c} product update", f"{c} announcement"]
                    for c in competitors
                }
            return {
                "search_queries": queries,
                "next_agent": "collector",
                "errors": errors,
            }

        # ----------------------------------------------------------------
        # Post-collection — route to memory_read
        # ----------------------------------------------------------------
        if current_next == "memory_read":
            logger.info("Supervisor: routing to memory_read.")
            return {"next_agent": "memory_read", "memory_mode": "read", "errors": errors}

        # ----------------------------------------------------------------
        # Post-memory-read — route to extractor
        # ----------------------------------------------------------------
        if current_next == "extractor":
            logger.info("Supervisor: routing to extractor.")
            return {"next_agent": "extractor", "errors": errors}

        # ----------------------------------------------------------------
        # Post-extraction — route to synthesizer
        # ----------------------------------------------------------------
        if current_next == "synthesizer":
            logger.info("Supervisor: routing to synthesizer.")
            return {"next_agent": "synthesizer", "errors": errors}

        # ----------------------------------------------------------------
        # Post-synthesis — route to memory_write
        # ----------------------------------------------------------------
        if current_next == "memory_write":
            logger.info("Supervisor: routing to memory_write.")
            return {"next_agent": "memory_write", "memory_mode": "write", "errors": errors}

        # Fallback — should not be reached in normal operation
        logger.warning("Supervisor: unrecognised next_agent='%s', ending.", current_next)
        return {"next_agent": "end", "errors": errors}


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

_supervisor_agent: SupervisorAgent | None = None


def supervisor_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node wrapper for the SupervisorAgent.

    The agent singleton is lazily initialised on first call so that importing
    this module does not require OPENAI_API_KEY to be set.

    Args:
        state: Shared pipeline state passed by LangGraph.

    Returns:
        Partial state update dict.
    """
    global _supervisor_agent
    if _supervisor_agent is None:
        _supervisor_agent = SupervisorAgent()
    return _supervisor_agent.run(state)
