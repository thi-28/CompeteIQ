"""
Data access and pure transforms for the dashboard. The dashboard is an API
client only; it never opens the database, so the API stays the single place
that defines how data is queried.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

# Validated categorical palette (light-mode steps), assigned in fixed order.
# Color follows the competitor, never its rank, so filters never repaint series.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER_COLOR = "#8a8984"
SINGLE_SERIES = CATEGORICAL[0]

SIGNAL_TYPES = [
    "product_launch", "pricing_change", "partnership", "research_release",
    "executive_move", "market_expansion", "other",
]


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 10.0, session: Any | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def _get(self, path: str, **params: Any) -> Any:
        params = {k: v for k, v in params.items() if v not in (None, "", [])}
        try:
            resp = self.session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ApiError(f"Cannot reach API at {self.base_url}: {exc}") from exc
        if resp.status_code == 404:
            return None
        if not resp.ok:
            raise ApiError(f"GET {path} failed: {resp.status_code} {resp.text[:200]}")
        return resp.json()

    def health(self) -> dict:
        return self._get("/health")

    def briefings(self, limit: int = 50) -> list[dict]:
        return self._get("/briefings", limit=limit) or []

    def briefing(self, run_id: str) -> dict | None:
        return self._get(f"/briefings/{run_id}")

    def signals(self, **filters: Any) -> list[dict]:
        return self._get("/signals", **filters) or []

    def runs(self, limit: int = 100) -> list[dict]:
        return self._get("/runs", limit=limit) or []

    def run(self, run_id: str) -> dict | None:
        return self._get(f"/runs/{run_id}")

    def start_run(self, competitors: list[str] | None = None, api_key: str | None = None) -> dict:
        headers = {"X-API-Key": api_key} if api_key else {}
        body = {"competitors": competitors} if competitors else None
        try:
            resp = self.session.post(f"{self.base_url}/run", json=body, headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ApiError(f"Cannot reach API at {self.base_url}: {exc}") from exc
        if resp.status_code != 202:
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            raise ApiError(f"Run not started ({resp.status_code}): {detail}")
        return resp.json()


# --- pure transforms ---------------------------------------------------------

def signals_frame(signals: list[dict]) -> pd.DataFrame:
    cols = ["date_detected", "competitor", "signal_type", "impact_assessment", "title", "summary",
            "confidence", "source_url", "run_id"]
    df = pd.DataFrame(signals, columns=cols)
    df["date_detected"] = pd.to_datetime(df["date_detected"], utc=True)
    return df


def color_map(competitors: list[str]) -> dict[str, str]:
    """Stable color per competitor in first-seen order; past 8 they fold to 'Other' gray."""
    return {c: (CATEGORICAL[i] if i < len(CATEGORICAL) else OTHER_COLOR) for i, c in enumerate(competitors)}


def ordered_competitors(df: pd.DataFrame, configured: list[str] | None = None) -> list[str]:
    """Configured order first (so colors match across sessions), then any others alphabetically."""
    configured = configured or []
    seen = set(df["competitor"]) if not df.empty else set()
    return [c for c in configured if c in seen] + sorted(seen - set(configured))


def weekly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Signals per ISO week (Monday start, UTC) per competitor, zero-filled so lines don't skip weeks."""
    if df.empty:
        return pd.DataFrame(columns=["week", "competitor", "signals"])
    week = df["date_detected"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time
    counts = df.assign(week=week).groupby(["week", "competitor"]).size().rename("signals")
    weeks = pd.date_range(week.min(), week.max(), freq="W-MON")
    full = pd.MultiIndex.from_product([weeks, sorted(df["competitor"].unique())], names=["week", "competitor"])
    return counts.reindex(full, fill_value=0).reset_index()


WEEK_MS = 7 * 24 * 60 * 60 * 1000


def week_tick_step(weekly: pd.DataFrame, max_ticks: int = 10) -> int:
    """Plotly dtick (ms) that lands on week starts: every week, or every Nth week when crowded."""
    n = weekly["week"].nunique() if not weekly.empty else 1
    return WEEK_MS * max(1, -(-n // max_ticks))


def type_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["signal_type"].value_counts() if not df.empty else pd.Series(dtype=int)
    out = pd.DataFrame({"signal_type": SIGNAL_TYPES, "signals": [int(counts.get(t, 0)) for t in SIGNAL_TYPES]})
    return out.sort_values("signals", ascending=False, kind="stable").reset_index(drop=True)


def runs_frame(runs: list[dict]) -> pd.DataFrame:
    cols = ["run_id", "status", "started_at", "completed_at", "competitors", "signal_count",
            "error_count", "research_retries"]
    df = pd.DataFrame(runs, columns=cols)
    if df.empty:
        return df
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["completed_at"] = pd.to_datetime(df["completed_at"], utc=True)
    df["duration_s"] = (df["completed_at"] - df["started_at"]).dt.total_seconds().round(0)
    df["competitors"] = df["competitors"].apply(lambda v: ", ".join(v or []))
    df["research_retries"] = df["research_retries"].apply(
        lambda v: ", ".join(f"{k}×{n}" for k, n in (v or {}).items()) or "none")
    return df
