from datetime import datetime, timedelta, timezone

import pytest

from memory.semantic import SemanticMemory
from models import Briefing, RunResult, Signal

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def mem(tmp_path):
    return SemanticMemory(db_path=str(tmp_path / "sub" / "test.db"))


def sig(competitor="Anthropic", title="Launch", url="https://a.com", days=0, **kw):
    data = dict(
        competitor=competitor, signal_type="product_launch", title=title, summary="s",
        impact_assessment="high", source_url=url, confidence=0.9, date_detected=T0 + timedelta(days=days),
    )
    data.update(kw)
    return Signal(**data)


def test_creates_parent_dir_and_tables(tmp_path):
    SemanticMemory(db_path=str(tmp_path / "nested" / "x.db"))
    assert (tmp_path / "nested" / "x.db").exists()


def test_save_and_get_signals_roundtrip_preserves_utc(mem):
    assert mem.save_signals([sig()], run_id="r1") == 1
    [row] = mem.get_signals()
    assert row["run_id"] == "r1" and row["competitor"] == "Anthropic"
    assert row["date_detected"] == "2026-11-01T12:00:00+00:00"


def test_non_utc_aware_datetime_is_converted(mem):
    est = timezone(timedelta(hours=-5))
    mem.save_signals([sig(date_detected=datetime(2026, 11, 1, 7, 0, tzinfo=est))], run_id="r1")
    assert mem.get_signals()[0]["date_detected"] == "2026-11-01T12:00:00+00:00"


def test_save_signals_is_idempotent_within_run(mem):
    s = [sig(), sig(title="Other")]
    assert mem.save_signals(s, "r1") == 2
    assert mem.save_signals(s, "r1") == 0  # retried write
    assert mem.save_signals(s, "r2") == 2  # same signal in a later run is a new row
    assert len(mem.get_signals()) == 4


def test_save_empty_signals(mem):
    assert mem.save_signals([], "r1") == 0


def test_signal_filters_and_ordering(mem):
    mem.save_signals([
        sig("Anthropic", "a-old", days=0),
        sig("Anthropic", "a-new", days=2, signal_type="partnership"),
        sig("Meta", "m", days=1),
        sig("Metaculus", "mc", days=1),
    ], "r1")
    mem.save_signals([sig("Meta", "m2", days=3)], "r2")

    assert [s["title"] for s in mem.get_signals()] == ["m2", "a-new", "mc", "m", "a-old"]
    assert {s["title"] for s in mem.get_signals(competitor="meta")} == {"m", "m2"}  # exact, case-insensitive
    assert [s["title"] for s in mem.get_signals(signal_type="partnership")] == ["a-new"]
    assert {s["title"] for s in mem.get_signals(run_id="r2")} == {"m2"}
    assert [s["title"] for s in mem.get_signals(since=T0 + timedelta(days=2))] == ["m2", "a-new"]
    assert [s["title"] for s in mem.get_signals(limit=2, offset=1)] == ["a-new", "mc"]


def test_briefings(mem):
    assert mem.get_latest_briefing() is None
    mem.save_briefing(Briefing(run_id="r1", created_at=T0, competitors_monitored=["A"], signal_count=1, content="# one"))
    mem.save_briefing(Briefing(run_id="r2", created_at=T0 + timedelta(days=7), competitors_monitored=["A"], signal_count=2, content="# two"))

    listed = mem.get_briefings()
    assert [b["run_id"] for b in listed] == ["r2", "r1"]
    assert "content" not in listed[0]
    assert mem.get_briefing("r1")["content"] == "# one"
    assert mem.get_briefing("missing") is None
    assert mem.get_latest_briefing()["run_id"] == "r2"


def test_runs_upsert_and_order(mem):
    mem.save_run(RunResult(run_id="r1", started_at=T0, competitors=["A", "B"], success=False))
    mem.save_run(RunResult(run_id="r1", started_at=T0, competitors=["A", "B"], signal_count=5,
                           research_retries={"B": 2}, briefing_id="r1"))
    mem.save_run(RunResult(run_id="r2", started_at=T0 + timedelta(days=7), competitors=["A"]))

    runs = mem.get_runs()
    assert [r["run_id"] for r in runs] == ["r2", "r1"]
    r1 = mem.get_run("r1")
    assert r1["signal_count"] == 5 and r1["success"] is True
    assert r1["research_retries"] == {"B": 2} and r1["competitors"] == ["A", "B"]
    assert r1["started_at"].endswith("+00:00")
    assert mem.get_run("nope") is None


def test_persists_across_instances(tmp_path):
    path = str(tmp_path / "p.db")
    SemanticMemory(path).save_signals([sig()], "r1")
    assert len(SemanticMemory(path).get_signals()) == 1


def test_naive_datetime_is_rejected(mem):
    from sqlalchemy.exc import StatementError

    with pytest.raises(StatementError, match="Naive datetimes"):
        mem.save_signals([sig(date_detected=datetime(2026, 11, 1, 12, 0))], "r1")
