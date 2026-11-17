import threading
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from memory.semantic import SemanticMemory
from models import Briefing, RunResult, Signal

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)


def sig(comp, title, days=0, stype="product_launch"):
    return Signal(competitor=comp, signal_type=stype, title=title, summary="s", impact_assessment="high",
                  source_url=f"https://{comp.lower()}.com/{title}", confidence=0.9, date_detected=T0 + timedelta(days=days))


@pytest.fixture
def store(tmp_path):
    s = SemanticMemory(str(tmp_path / "api.db"))
    s.save_signals([sig("Anthropic", "a1"), sig("Meta", "m1", 2, "partnership")], "r1")
    s.save_signals([sig("Anthropic", "a2", 7)], "r2")
    s.save_briefing(Briefing(run_id="r1", created_at=T0, competitors_monitored=["Anthropic", "Meta"], signal_count=2, content="# r1"))
    s.save_briefing(Briefing(run_id="r2", created_at=T0 + timedelta(days=7), competitors_monitored=["Anthropic"], signal_count=1, content="# r2"))
    s.save_run(RunResult(run_id="r1", started_at=T0, competitors=["Anthropic", "Meta"], signal_count=2, success=True,
                         research_retries={"Meta": 1}))
    s.save_run(RunResult(run_id="r2", started_at=T0 + timedelta(days=7), competitors=["Anthropic"], success=False))
    return s


class FakeRunner:
    def __init__(self, store, block: threading.Event | None = None):
        self.store, self.block, self.calls = store, block, []

    def __call__(self, competitors, run_id):
        self.calls.append((competitors, run_id))
        if self.block:
            self.block.wait(5)
        result = RunResult(run_id=run_id, started_at=datetime.now(timezone.utc), competitors=competitors)
        self.store.save_run(result)
        return result


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    runner = FakeRunner(store)
    c = TestClient(create_app(store, runner))
    c.runner = runner
    return c


# --- read endpoints ---------------------------------------------------------

def test_health(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["database"] is True
    assert body["openai_model"] == "gpt-4o-mini" and body["embedding_model"] == "text-embedding-3-small"
    assert body["openai_key_configured"] is True and body["tavily_key_configured"] is False
    assert "sk-secret" not in str(body)


def test_list_and_get_briefings(client):
    listed = client.get("/briefings").json()
    assert [b["run_id"] for b in listed] == ["r2", "r1"] and "content" not in listed[0]
    assert client.get("/briefings/r1").json()["content"] == "# r1"
    assert client.get("/briefings/latest").json()["run_id"] == "r2"
    assert client.get("/briefings/nope").status_code == 404
    assert client.get("/briefings?limit=1&offset=1").json()[0]["run_id"] == "r1"


def test_signals_filters(client):
    assert len(client.get("/signals").json()) == 3
    assert {s["title"] for s in client.get("/signals?competitor=anthropic").json()} == {"a1", "a2"}
    assert [s["title"] for s in client.get("/signals?signal_type=partnership").json()] == ["m1"]
    assert [s["title"] for s in client.get("/signals?run_id=r2").json()] == ["a2"]
    assert [s["title"] for s in client.get("/signals", params={"since": "2026-11-02T00:00:00Z"}).json()] == ["a2", "m1"]
    assert client.get("/signals?limit=1").json()[0]["date_detected"].startswith("2026-11-08")


@pytest.mark.parametrize("query", ["signal_type=rumor", "limit=0", "limit=5000", "since=2026-11-02T00:00:00"])
def test_signals_validation(client, query):
    assert client.get(f"/signals?{query}").status_code == 422


def test_runs(client):
    runs = client.get("/runs").json()
    assert [(r["run_id"], r["status"]) for r in runs] == [("r2", "failed"), ("r1", "succeeded")]
    r1 = client.get("/runs/r1").json()
    assert r1["research_retries"] == {"Meta": 1} and "success" not in r1
    assert client.get("/runs/nope").status_code == 404


# --- POST /run --------------------------------------------------------------

def test_post_run_returns_trackable_run_id(client):
    resp = client.post("/run", json={"competitors": ["Mistral", " "]})
    assert resp.status_code == 202
    body = resp.json()
    assert body["competitors"] == ["Mistral"] and body["status"] == "started"
    # TestClient runs background tasks before returning; the id is the one the pipeline used.
    assert client.runner.calls == [(["Mistral"], body["run_id"])]
    assert client.get(f"/runs/{body['run_id']}").json()["status"] == "succeeded"


def test_post_run_defaults_competitors(client):
    resp = client.post("/run")
    assert resp.status_code == 202
    assert resp.json()["competitors"] == ["Anthropic", "Google", "Meta", "Mistral", "Perplexity"]


def test_post_run_rejects_empty_competitors(client):
    assert client.post("/run", json={"competitors": ["", " "]}).status_code == 422


def test_concurrent_run_is_rejected_and_reported_as_running(store, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    gate = threading.Event()
    app = create_app(store, FakeRunner(store, block=gate))
    tracker = app.state.tracker
    tracker.try_start("busy123", ["A"])  # simulate a run already in flight
    with TestClient(app) as c:
        resp = c.post("/run")
        assert resp.status_code == 409 and "busy123" in resp.json()["detail"]
        running = c.get("/runs/busy123").json()
        assert running["status"] == "running" and running["competitors"] == ["A"]
        assert c.get("/health").json()["active_run"] == "busy123"
    tracker.finish()
    gate.set()


def test_tracker_released_after_runner_crash(store, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)

    def boom(competitors, run_id):
        raise RuntimeError("crash")

    app = create_app(store, boom)
    with TestClient(app) as c:
        assert c.post("/run").status_code == 202
        assert app.state.tracker.active is None
        assert c.post("/run").status_code == 202  # not stuck at 409


def test_api_key_required_for_run_when_configured(store, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    c = TestClient(create_app(store, FakeRunner(store)))
    assert c.post("/run").status_code == 401
    assert c.post("/run", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/run", headers={"X-API-Key": "s3cret"}).status_code == 202
    assert c.get("/briefings").status_code == 200  # reads stay open
