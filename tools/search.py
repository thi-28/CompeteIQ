"""
Tavily web search tool wrapper with retry logic and rate-limit handling.

Each call returns up to max_results structured results containing title,
URL, and a content snippet — suitable for feeding directly into LLM prompts.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from tavily import TavilyClient

logger = logging.getLogger(__name__)

# Default number of search results to request per query
DEFAULT_MAX_RESULTS = 5
# Exponential backoff parameters
_INITIAL_BACKOFF_SECS = 2.0
_MAX_BACKOFF_SECS = 32.0
_MAX_RETRIES = 4


class TavilySearchTool:
    """
    Thin wrapper around the Tavily search client.

    Provides search with automatic exponential-backoff retries on rate-limit
    or transient network errors, and normalises results into a consistent dict
    structure consumed by the collector agent.
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> None:
        """
        Initialise the Tavily client.

        Args:
            api_key: Tavily API key. Defaults to TAVILY_API_KEY env var.
            max_results: Maximum number of results to request per search query.
        """
        resolved_key = api_key or os.getenv("TAVILY_API_KEY")
        if not resolved_key:
            raise ValueError(
                "TAVILY_API_KEY is not set. "
                "Export the environment variable or pass api_key= explicitly."
            )
        self._client = TavilyClient(api_key=resolved_key)
        self._max_results = max_results
        logger.info(
            "TavilySearchTool initialised — max_results=%d", self._max_results
        )

    def search(
        self,
        query: str,
        search_depth: str = "basic",
    ) -> list[dict[str, Any]]:
        """
        Execute a single search query with retry logic.

        Args:
            query: The search query string.
            search_depth: Tavily search depth — "basic" (faster/cheaper)
                          or "advanced" (deeper, costs more credits).

        Returns:
            A list of result dicts, each with keys:
            - ``title`` (str): Page title.
            - ``url`` (str): Source URL.
            - ``content`` (str): Relevant text snippet.
            - ``score`` (float): Tavily relevance score.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        backoff = _INITIAL_BACKOFF_SECS

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.search(
                    query=query,
                    max_results=self._max_results,
                    search_depth=search_depth,
                )
                results = response.get("results", [])
                normalised = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                        "score": r.get("score", 0.0),
                    }
                    for r in results
                ]
                logger.debug(
                    "Search '%s' returned %d results (attempt %d).",
                    query,
                    len(normalised),
                    attempt,
                )
                return normalised

            except Exception as exc:
                error_str = str(exc).lower()
                is_rate_limit = any(
                    tok in error_str
                    for tok in ("rate limit", "429", "too many requests", "quota")
                )
                if attempt == _MAX_RETRIES:
                    logger.error(
                        "Search '%s' failed after %d attempts: %s",
                        query,
                        _MAX_RETRIES,
                        exc,
                    )
                    raise RuntimeError(
                        f"Tavily search failed for query '{query}': {exc}"
                    ) from exc

                wait = min(backoff, _MAX_BACKOFF_SECS)
                logger.warning(
                    "Search '%s' %s (attempt %d/%d). Retrying in %.1fs.",
                    query,
                    "rate-limited" if is_rate_limit else "errored",
                    attempt,
                    _MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                backoff *= 2

        # Unreachable — kept for type checker
        return []

    def batch_search(
        self,
        queries: list[str],
        delay_between: float = 0.5,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Run multiple search queries sequentially with a small inter-query delay.

        Failed individual queries are logged and skipped rather than halting
        the entire batch.

        Args:
            queries: List of query strings to execute.
            delay_between: Seconds to wait between queries (avoids burst rate limits).

        Returns:
            Dict mapping each query string to its list of result dicts.
            Queries that fail after all retries map to an empty list.
        """
        results: dict[str, list[dict[str, Any]]] = {}
        for i, query in enumerate(queries):
            try:
                results[query] = self.search(query)
            except RuntimeError as exc:
                logger.error("Batch search skipping query '%s': %s", query, exc)
                results[query] = []

            if i < len(queries) - 1:
                time.sleep(delay_between)

        return results
