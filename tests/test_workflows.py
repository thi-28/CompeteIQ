"""Contract checks on the GitHub Actions workflows (actionlint covers syntax)."""

from pathlib import Path

import yaml

WF = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def load(name):
    doc = yaml.safe_load((WF / name).read_text())
    doc["on"] = doc.pop(True, doc.get("on"))  # YAML 1.1 parses the bare key `on` as True
    return doc


def steps_by_name(job):
    return {s.get("name", s.get("uses")): s for s in job["steps"]}


def test_ci_runs_tests_with_coverage_without_secrets():
    ci = load("ci.yml")
    text = (WF / "ci.yml").read_text()
    assert {"push", "pull_request"} <= set(ci["on"])
    run = steps_by_name(ci["jobs"]["test"])["Run tests with coverage"]["run"]
    assert "pytest --cov" in run and "--cov-report=xml" in run
    assert "secrets." not in text
    assert "docker build" in steps_by_name(ci["jobs"]["docker"])["Build image"]["run"]


def test_weekly_restores_data_before_running_and_saves_after():
    wf = load("weekly.yml")
    assert wf["on"]["schedule"][0]["cron"] == "0 8 * * 1" and "workflow_dispatch" in wf["on"]
    assert wf["concurrency"]["cancel-in-progress"] is False
    names = [s.get("name") for s in wf["jobs"]["run"]["steps"]]
    order = ["Restore data/ from cache", "Fall back to latest data artifact", "Run pipeline",
             "Save data/ to cache", "Upload data/ artifact"]
    assert [names.index(n) for n in order] == sorted(names.index(n) for n in order)


def test_weekly_cache_keys_are_unique_per_run_and_restore_newest_by_prefix():
    steps = steps_by_name(load("weekly.yml")["jobs"]["run"])
    restore, save = steps["Restore data/ from cache"], steps["Save data/ to cache"]
    assert restore["uses"].startswith("actions/cache/restore@") and save["uses"].startswith("actions/cache/save@")
    assert restore["with"]["path"] == save["with"]["path"] == "data"
    assert "github.run_id" in save["with"]["key"] and restore["with"]["key"] == save["with"]["key"]
    assert restore["with"]["restore-keys"] == "${{ env.DATA_CACHE_PREFIX }}"


def test_weekly_never_saves_unless_memory_was_restored_first():
    steps = steps_by_name(load("weekly.yml")["jobs"]["run"])
    for name in ("Save data/ to cache", "Upload data/ artifact"):
        cond = steps[name]["if"]
        assert "always()" in cond and "steps.restore.outcome == 'success'" in cond
        assert "steps.pipeline.outcome != 'skipped'" in cond
    assert steps["Upload data/ artifact"]["with"]["retention-days"] == 90


def test_weekly_storage_paths_point_into_the_saved_directory():
    env = load("weekly.yml")["jobs"]["run"]["env"]
    assert env["DATA_DIR"] == "./data"
    assert env["SQLITE_DB_PATH"].startswith("./data/") and env["CHROMA_PERSIST_DIR"].startswith("./data/")
    assert env["OPENAI_API_KEY"] == "${{ secrets.OPENAI_API_KEY }}"
