"""URL helpers shared by the extractor (grounding) and episodic memory (dedup ids)."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def normalize_url(url: str) -> str:
    """Canonical form for comparison: lowercase scheme/host, drop fragment and
    trailing slash. Query strings are kept (they can identify distinct pages)."""
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))
