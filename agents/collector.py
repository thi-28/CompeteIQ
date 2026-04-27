"""
Collector agent — executes web searches and harvests raw signal text.

For each competitor, the collector runs the pre-generated search queries
through Tavily and aggregates the result snippets into ``raw_signals``.
Rate-limit errors and individual query failures are logged to the
``errors`` list so the pipeline can continue gracefully.
"""

from __future__ import annotations

import logging
from typing import Any

from tools.search import TavilySearchTool
from models.schemas import AgentState

logger = logging.getLogger(__name__)

# Maximum content characters to keep per search result (avoids token bloat)
_MAX_CONTENT_CHARS = 1_200


class CollectorAgent:
    """
    Executes Tavily searches for all competitor queries and stores raw text.

    Results are truncated to ``_MAX_CONTENT_CHARS`` to keep total token
    consumption manageable for downstream LLM calls.
    """

    def __init__(self, search_tool: TavilySearchTool | None = None) -> None:
        """
        Initialise the collector with a search tool.

        Args:
            search_tool: Optional pre-constructed TavilySearchTool.
                         A new one will be created from environment variables if omitted.
        """
        self._search = search_tool or TavilySearchTool()
        logger.info("CollectorAgent initialised.")

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Execute searches for all competitors and populate ``raw_signals``.

        For each competitor:
        1. Retrieves its list of search queries from ``state["search_queries"]``.
        2. Calls Tavily for each query.
        3. Concatenates result titles + content snippets into strings.
        4. Stores the list under ``raw_signals[competitor]``.

        Failed individual queries are recorded in ``errors`` and skipped;
        the pipeline continues with whatever data was collected.

        Args:
            state: Current pipeline state.

        Returns:
            Partial state update with ``raw_signals``, ``next_agent``, and ``errors``.
        """
        search_queries: dict[str, list[str]] = state.get("search_queries", {})
        errors: list[str] = list(state.get("errors", []))
        raw_signals: dict[str, list[str]] = {}

        for competitor, queries in search_queries.items():
            competitor_snippets: list[str] = []
            logger.info(
                "Collector: searching %d queries for '%s'.", len(queries), competitor
            )
            for query in queries:
                try:
                    results = self._search.search(query)
                    for r in results:
                        title = r.get("title", "")
                        content = r.get("content", "")[: _MAX_CONTENT_CHARS]
                        url = r.get("url", "")
                        snippet = f"Title: {title}\nURL: {url}\nContent: {content}"
                        competitor_snippets.append(snippet)
                except RuntimeError as exc:
                    msg = f"collector:{competitor}:{query}: {exc}"
                    logger.error(msg)
                    errors.append(msg)

            raw_signals[competitor] = competitor_snippets
            logger.info(
                "Collector: collected %d snippets for '%s'.",
                len(competitor_snippets),
                competitor,
            )

        total = sum(len(v) for v in raw_signals.values())
        logger.info("Collector: total snippets collected = %d.", total)

        return {
            "raw_signals": raw_signals,
            "next_agent": "memory_read",
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

_collector_agent: CollectorAgent | None = None


def collector_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node wrapper for the CollectorAgent.

    The agent singleton is lazily initialised on first call so that importing
    this module does not require TAVILY_API_KEY to be set.

    Args:
        state: Shared pipeline state passed by LangGraph.

    Returns:
        Partial state update dict.
    """
    global _collector_agent
    if _collector_agent is None:
        _collector_agent = CollectorAgent()
    return _collector_agent.run(state)
