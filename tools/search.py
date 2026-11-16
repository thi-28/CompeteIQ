"""
Tavily web search wrapper with exponential-backoff retries.

Only transient failures (timeouts, rate limits, network errors) are retried.
Auth, permission, bad-request and exhausted-quota errors fail immediately,
because retrying them just burns time and credits.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Callable, Protocol

from tavily import TavilyClient
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)

logger = logging.getLogger(__name__)

NON_RETRYABLE: tuple[type[Exception], ...] = (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)


class SearchError(RuntimeError):
    """Raised when a search fails permanently or exhausts its retries."""


class SearchClient(Protocol):
    def search(self, query: str, **kwargs: Any) -> dict[str, Any]: ...


class TavilySearchTool:
    def __init__(
        self,
        client: SearchClient | None = None,
        api_key: str | None = None,
        max_results: int = 5,
        max_retries: int = 4,
        base_delay: float = 1.0,
        max_delay: float = 16.0,
        jitter: float = 0.1,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if client is None:
            key = api_key or os.getenv("TAVILY_API_KEY")
            if not key:
                raise SearchError("TAVILY_API_KEY is not set")
            client = TavilyClient(api_key=key)
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self._client = client
        self.max_results = max_results
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self._sleep = sleep

    def backoff_delay(self, attempt: int) -> float:
        """Delay before retry number `attempt` (1-based): base * 2^(n-1), capped, plus jitter."""
        delay = min(self.base_delay * 2 ** (attempt - 1), self.max_delay)
        return delay + random.uniform(0, delay * self.jitter)

    def search(self, query: str, search_depth: str = "basic") -> list[dict[str, Any]]:
        """Run one query. Returns normalized results that have both a URL and content."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.search(
                    query=query, max_results=self.max_results, search_depth=search_depth
                )
                return self._normalize(response)
            except NON_RETRYABLE as exc:
                raise SearchError(f"Non-retryable search error for {query!r}: {exc}") from exc
            except Exception as exc:  # transient: timeout, 429, connection reset, ...
                if attempt == self.max_retries:
                    raise SearchError(
                        f"Search for {query!r} failed after {attempt} attempts: {exc}"
                    ) from exc
                wait = self.backoff_delay(attempt)
                logger.warning(
                    "Search %r failed (attempt %d/%d): %s. Retrying in %.1fs",
                    query, attempt, self.max_retries, exc, wait,
                )
                self._sleep(wait)
        raise AssertionError("unreachable")

    @staticmethod
    def _normalize(response: dict[str, Any]) -> list[dict[str, Any]]:
        results = []
        for r in response.get("results", []) or []:
            url, content = (r.get("url") or "").strip(), (r.get("content") or "").strip()
            if not url or not content:
                continue  # can't ground a signal on a result without both
            results.append(
                {
                    "title": r.get("title") or "",
                    "url": url,
                    "content": content,
                    "score": float(r.get("score") or 0.0),
                }
            )
        return results
