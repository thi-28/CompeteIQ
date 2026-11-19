import json
import re
from types import SimpleNamespace
from typing import get_args

import pytest

from agents import extractor as extractor_mod
from config import Settings
from evals import run_eval
from evals.run_eval import evaluate, load_dataset, markdown_table, match
from models.schemas import SignalType

CASES = load_dataset()
URL = {c["id"]: c["snippet"]["url"] for c in CASES}


def test_dataset_is_well_formed():
    assert len(CASES) == 20
    assert len({c["id"] for c in CASES}) == len({c["snippet"]["url"] for c in CASES}) == 20
    types = set(get_args(SignalType))
    for c in CASES:
        assert set(c["expected"]) <= types and c["note"]
        assert re.match(r"https://[a-z.]+\.example/", c["snippet"]["url"])  # fictional, reserved TLD
    assert sum(not c["expected"] for c in CASES) >= 5  # enough negatives to measure rejection
    assert any(len(c["expected"]) > 1 for c in CASES)  # multi-signal snippet exercises recall


def sig(case_id, stype, conf=0.9, url=None, title="t"):
    return {"signal_type": stype, "title": title, "summary": "s", "impact_assessment": "medium",
            "source_url": url or URL[case_id], "confidence": conf}


def test_match_strict_vs_lenient():
    cases = [c for c in CASES if c["id"] in ("nimbus-05", "nimbus-04")]
    accepted = [sig("nimbus-05", "product_launch"), sig("nimbus-05", "partnership"), sig("nimbus-04", "other")]
    assert match(cases, accepted, strict=True) == (1, 2, 1)
    assert match(cases, accepted, strict=False) == (2, 1, 0)


# Scripted model output per competitor: mostly right, plus every failure mode.
SCRIPT = {
    "Corvid Labs": [
        sig("corvid-01", "product_launch", title="Corvid 3"),
        sig("corvid-02", "pricing_change"),
        sig("corvid-03", "other", conf=0.3),                              # rumor, dropped by confidence gate
        sig("corvid-04", "executive_move"),
        sig("corvid-05", "market_expansion"),                             # off-topic false positive
    ],
    "Nimbus AI": [
        sig("nimbus-01", "partnership"),
        sig("nimbus-02", "market_expansion"),
        sig("nimbus-03", "product_launch"),                               # wrong type: lenient TP, strict FP
        sig("nimbus-05", "product_launch"),                               # misses the pricing change
        sig("nimbus-05", "pricing_change", url="https://news.example/made-up"),  # hallucinated URL
    ],
    "Halyard": [sig("halyard-01", "product_launch"), sig("halyard-02", "executive_move"),
                sig("halyard-04", "partnership")],
    "Tessellate": [sig("tessellate-01", "market_expansion"), sig("tessellate-02", "pricing_change"),
                   sig("tessellate-03", "research_release"), sig("tessellate-05", "other"),
                   {"signal_type": "nonsense"}],                          # schema-invalid
}


class FakeClient:
    def __init__(self):
        self.judged = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, model, messages, **_):
        system, user = messages[0]["content"], messages[1]["content"]
        if system == extractor_mod.SYSTEM_PROMPT:
            comp = re.search(r"COMPETITOR: (.+)", user).group(1)
            payload = {"signals": SCRIPT[comp]}
        else:
            assert system == run_eval.JUDGE_PROMPT
            self.judged.append(model)
            claim = user.split("CLAIM:\n", 1)[1]
            bad = claim.startswith("Corvid 3\n")
            payload = {"supported": not bad, "unsupported_facts": ["invented date"] if bad else []}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])


@pytest.fixture
def report():
    client = FakeClient()
    rep = evaluate(CASES, client, Settings(), judge=True, judge_model="judge-x")
    rep["_client"] = client
    return rep


def test_metrics_from_scripted_run(report):
    m = report["metrics"]
    # accepted: corvid 4 (01,02,04,05) + nimbus 4 (01,02,03,05) + halyard 3 + tessellate 4 = 15
    assert m["accepted_signals"] == 15 and m["gold_signals"] == 15
    # strict TP: 13 (all but corvid-05 off-topic and nimbus-03 wrong type)
    assert m["precision"] == round(13 / 15, 3) and m["recall"] == round(13 / 15, 3)
    # lenient: nimbus-03 counts; misses are nimbus-05 pricing only
    assert m["precision_lenient"] == round(14 / 15, 3) and m["recall_lenient"] == round(14 / 15, 3)
    assert m["negative_rejection"] == round(5 / 6, 3) and m["false_positive_snippets"] == ["corvid-05"]
    # raw candidates 18; ungrounded: made-up URL + the invalid one without a URL
    assert m["url_grounding_pre_guard"] == round(16 / 18, 3)
    assert m["gates"] == {"candidates": 18, "invalid": 1, "ungrounded": 1, "low_confidence": 1, "accepted": 15}


def test_faithfulness_uses_judge_model(report):
    f = report["faithfulness"]
    assert f["judge_model"] == "judge-x" and set(report["_client"].judged) == {"judge-x"}
    assert f["faithfulness"] == round(14 / 15, 3)
    assert [(v["title"], v["unsupported_facts"]) for v in f["verdicts"] if not v["supported"]] == [
        ("Corvid 3", ["invented date"])]
    assert len(report["candidates"]["Tessellate"]) == 5  # raw output kept, incl. the invalid one


def test_markdown_table_and_cli(report, monkeypatch, tmp_path, capsys):
    table = markdown_table(report)
    assert "| Extraction precision (strict) | 0.87 |" in table and "| Faithfulness (LLM judge) | 0.93 |" in table
    assert "gpt-4o-mini" in table

    import openai
    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    out = tmp_path / "r.json"
    run_eval.main(["--out", str(out), "--model", "m-test"])
    saved = json.loads(out.read_text())
    assert saved["model"] == "m-test" and "faithfulness" not in saved
    assert "| Faithfulness (LLM judge) | n/a |" in capsys.readouterr().out
