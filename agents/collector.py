"""
Collector agent: runs each competitor's search queries and stores structured,
URL-attributed snippets in state.

Two modes:
- Initial pass: collect for every competitor in `search_queries`.
- Re-collection: when the supervisor flags `low_signal_competitors`, collect only
  for those, skip URLs already collected, and append the new snippets.
"""

from __future__ import annotations

import logging
from typing import Any

from config import Settings, get_settings
from models.schemas import AgentState, Snippet
from tools.search import SearchError, TavilySearchTool

logger = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 1_200


class CollectorAgent:
    def __init__(
        self,
        search_tool: TavilySearchTool | None = None,
        settings: Settings | None = None,
        max_content_chars: int = MAX_CONTENT_CHARS,
    ) -> None:
        self._search_tool = search_tool
        self.settings = settings or get_settings()
        self.max_content_chars = max_content_chars

    @property
    def search_tool(self) -> TavilySearchTool:
        # Lazy so importing/constructing the graph doesn't require TAVILY_API_KEY.
        if self._search_tool is None:
            self._search_tool = TavilySearchTool()
        return self._search_tool

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """LangGraph node entry point."""
        return self.run(state)

    def run(self, state: AgentState) -> dict[str, Any]:
        queries_by_comp = state.get("search_queries", {})
        existing = state.get("snippets", {})
        targets = state.get("low_signal_competitors") or list(queries_by_comp)
        cap = self.settings.max_snippets_per_competitor

        snippets = {comp: list(items) for comp, items in existing.items()}
        errors: list[str] = []

        for comp in targets:
            seen = {s["url"] for s in snippets.get(comp, [])}
            fresh, comp_errors = self._collect(comp, queries_by_comp.get(comp, []), seen)
            errors.extend(comp_errors)
            snippets[comp] = snippets.get(comp, []) + [s.model_dump() for s in fresh[:cap]]
            logger.info(
                "Collector: %s +%d snippets (%d total)", comp, min(len(fresh), cap), len(snippets[comp])
            )
            if not snippets[comp]:
                errors.append(f"collector:{comp}: no snippets collected")

        return {"snippets": snippets, "errors": errors}

    def _collect(
        self, competitor: str, queries: list[str], seen_urls: set[str]
    ) -> tuple[list[Snippet], list[str]]:
        """Run all queries for one competitor. Returns new snippets ranked by score."""
        best: dict[str, tuple[float, Snippet]] = {}
        errors: list[str] = []

        for query in queries:
            try:
                results = self.search_tool.search(query)
            except SearchError as exc:
                errors.append(f"collector:{competitor}:{query!r}: {exc}")
                logger.error(errors[-1])
                continue
            for r in results:
                url = r["url"]
                if url in seen_urls:
                    continue
                if url not in best or r["score"] > best[url][0]:
                    best[url] = (
                        r["score"],
                        Snippet(
                            competitor=competitor,
                            url=url,
                            title=r["title"],
                            content=r["content"][: self.max_content_chars],
                            query=query,
                        ),
                    )

        ranked = sorted(best.values(), key=lambda pair: pair[0], reverse=True)
        return [snippet for _, snippet in ranked], errors
