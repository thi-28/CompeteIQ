from types import SimpleNamespace

import pandas as pd
import pytest
import requests

import dashboard.data as data
from dashboard.data import (
    CATEGORICAL, OTHER_COLOR, ApiClient, ApiError, color_map, ordered_competitors,
    runs_frame, signals_frame, type_counts, weekly_counts,
)

SIGNALS = [
    {"date_detected": "2026-11-02T10:00:00+00:00", "competitor": "Meta", "signal_type": "partnership",
     "impact_assessment": "high", "title": "m1", "summary": "s", "confidence": 0.8, "source_url": "https://m.com", "run_id": "r1"},
    {"date_detected": "2026-11-03T10:00:00+00:00", "competitor": "Anthropic", "signal_type": "product_launch",
     "impact_assessment": "medium", "title": "a1", "summary": "s", "confidence": 0.9, "source_url": "https://a.com/1", "run_id": "r1"},
    {"date_detected": "2026-11-18T10:00:00+00:00", "competitor": "Anthropic", "signal_type": "product_launch",
     "impact_assessment": "high", "title": "a2", "summary": "s", "confidence": 0.7, "source_url": "https://a.com/2", "run_id": "r2"},
]
RUNS = [{"run_id": "r2", "status": "succeeded", "started_at": "2026-11-18T10:00:00+00:00",
         "completed_at": "2026-11-18T10:02:05+00:00", "competitors": ["Anthropic", "Meta"],
         "signal_count": 1, "error_count": 0, "research_retries": {"Meta": 2}, "briefing_id": "r2"}]


# --- transforms ----------------------------------------------------------------

def test_weekly_counts_zero_fills_missing_weeks_and_competitors():
    weekly = weekly_counts(signals_frame(SIGNALS))
    pivot = weekly.pivot(index="week", columns="competitor", values="signals")
    assert [str(w.date()) for w in pivot.index] == ["2026-11-02", "2026-11-09", "2026-11-16"]
    assert pivot["Anthropic"].tolist() == [1, 0, 1]
    assert pivot["Meta"].tolist() == [1, 0, 0]


def test_weekly_counts_empty():
    assert weekly_counts(signals_frame([])).empty


def test_type_counts_includes_all_types_sorted():
    tc = type_counts(signals_frame(SIGNALS))
    assert tc.iloc[0].tolist() == ["product_launch", 2]
    assert len(tc) == 7 and tc["signals"].sum() == 3
    assert type_counts(signals_frame([]))["signals"].sum() == 0


def test_colors_follow_entity_in_configured_order_and_fold_past_eight():
    df = signals_frame(SIGNALS)
    comps = ordered_competitors(df, ["Anthropic", "Google", "Meta"])
    assert comps == ["Anthropic", "Meta"]  # configured order, absent ones skipped
    many = [f"C{i}" for i in range(10)]
    cmap = color_map(many)
    assert cmap["C0"] == CATEGORICAL[0] and cmap["C7"] == CATEGORICAL[7] and cmap["C9"] == OTHER_COLOR
    # Filtering out a competitor never repaints the others.
    assert color_map(["Anthropic", "Meta"])["Meta"] == color_map(["Anthropic", "Meta", "Zeta"])["Meta"]


def test_ordered_competitors_appends_unconfigured_alphabetically():
    df = signals_frame(SIGNALS + [{**SIGNALS[0], "competitor": "Cohere"}])
    assert ordered_competitors(df, ["Meta"]) == ["Meta", "Anthropic", "Cohere"]


def test_runs_frame_formats_fields():
    df = runs_frame(RUNS)
    assert df.loc[0, "duration_s"] == 125
    assert df.loc[0, "competitors"] == "Anthropic, Meta"
    assert df.loc[0, "research_retries"] == "Meta×2"
    assert runs_frame([]).empty


# --- API client --------------------------------------------------------------------

class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = responses, []

    def _resp(self, status, payload):
        return SimpleNamespace(status_code=status, ok=200 <= status < 300, json=lambda: payload,
                               text=str(payload), headers={"content-type": "application/json"})

    def get(self, url, params=None, timeout=None):
        self.calls.append(("GET", url, params))
        r = self.responses[url.split("8000")[1]]
        if isinstance(r, Exception):
            raise r
        return self._resp(*r)

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(("POST", url, json, headers))
        return self._resp(*self.responses["POST /run"])


def test_client_drops_empty_params_and_handles_404():
    s = FakeSession({"/signals": (200, SIGNALS), "/briefings/x": (404, {"detail": "nf"})})
    c = ApiClient("http://api:8000/", session=s)
    assert len(c.signals(competitor="", limit=5, run_id=None)) == 3
    assert s.calls[0] == ("GET", "http://api:8000/signals", {"limit": 5})
    assert c.briefing("x") is None


def test_client_errors():
    s = FakeSession({"/health": requests.ConnectionError("refused"), "/runs": (500, "boom")})
    c = ApiClient("http://api:8000", session=s)
    with pytest.raises(ApiError, match="Cannot reach API"):
        c.health()
    with pytest.raises(ApiError, match="500"):
        c.runs()


def test_client_start_run():
    ok = FakeSession({"POST /run": (202, {"run_id": "abc", "status": "started", "competitors": ["A"]})})
    assert ApiClient("http://api:8000", session=ok).start_run(["A"], "key")["run_id"] == "abc"
    assert ok.calls[0][2:] == ({"competitors": ["A"]}, {"X-API-Key": "key"})

    busy = FakeSession({"POST /run": (409, {"detail": "Run r1 is already in progress"})})
    with pytest.raises(ApiError, match="409.*already in progress"):
        ApiClient("http://api:8000", session=busy).start_run()


# --- app renders end to end (Streamlit AppTest) --------------------------------------

class FakeClient:
    def __init__(self, base_url, *a, **k):
        pass

    def health(self):
        return {"openai_model": "gpt-4o-mini", "embedding_model": "text-embedding-3-small", "active_run": None}

    def briefings(self, limit=50):
        return [{"run_id": "r2", "created_at": "2026-11-18T10:00:00+00:00", "signal_count": 1}]

    def briefing(self, run_id):
        return {"run_id": run_id, "content": "# CompeteIQ Briefing: 2026-11-18\n\nBody"}

    def signals(self, **kw):
        return SIGNALS

    def runs(self, limit=100):
        return RUNS


class DownClient(FakeClient):
    def health(self):
        raise ApiError("Cannot reach API at http://x")


def run_app(monkeypatch, client_cls):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_data.clear()  # cache is process-global; isolate app runs from each other
    monkeypatch.setattr(data, "ApiClient", client_cls)
    at = AppTest.from_file("../dashboard/app.py", default_timeout=30)
    at.run()
    return at


def test_app_renders_all_tabs(monkeypatch):
    at = run_app(monkeypatch, FakeClient)
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Briefings", "Signal explorer", "Trends", "Run history"]
    assert any("CompeteIQ Briefing: 2026-11-18" in m.value for m in at.markdown)
    assert at.metric[0].value == "3"


def test_app_shows_error_when_api_down(monkeypatch):
    at = run_app(monkeypatch, DownClient)
    assert not at.exception
    assert "Cannot reach API" in at.error[0].value
