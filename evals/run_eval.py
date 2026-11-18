"""
Extraction eval: precision, recall and faithfulness of the extractor on a
labeled snippet set.

    python evals/run_eval.py                 # precision / recall / grounding
    python evals/run_eval.py --judge         # + LLM-judged faithfulness

Snippets are grouped by competitor and sent in one extractor call per
competitor, as the pipeline does. The real ExtractorAgent prompt and gates
(schema, URL grounding, confidence) are used, so the numbers describe what
would actually be stored.

Metrics
- precision / recall / F1 (strict): an accepted signal is a true positive if
  its source_url is a labeled snippet AND its signal_type matches an unmatched
  gold label for that snippet.
- precision / recall (lenient): same, ignoring signal_type.
- negative rejection: share of no-signal snippets (rumor, opinion, off-topic)
  that produced zero accepted signals.
- URL grounding (pre-guard): share of raw model candidates whose URL matches a
  provided snippet. The guard drops the rest, so this measures how often the
  model invents sources, not what gets stored.
- faithfulness (--judge): share of accepted signals whose title and summary
  are fully supported by their source snippet, as judged by an LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python evals/run_eval.py`

from agents.extractor import ExtractorAgent  # noqa: E402
from config import Settings, get_settings  # noqa: E402
from tools.urls import normalize_url  # noqa: E402

DATASET = Path(__file__).resolve().parent / "dataset.json"

JUDGE_PROMPT = """You check whether an extracted claim is supported by its source text.
Given a SOURCE and a CLAIM (title + summary), answer whether every factual statement in
the claim is stated in or directly implied by the source. Commentary on why it matters
is fine; new facts, numbers, names or dates that are not in the source are not.
Respond with JSON: {"supported": true|false, "reason": "<one sentence>"}"""


def load_dataset(path: Path = DATASET) -> list[dict[str, Any]]:
    return json.loads(path.read_text())["cases"]


def run_extraction(cases: list[dict[str, Any]], extractor: ExtractorAgent) -> dict[str, dict[str, Any]]:
    """One extractor call per competitor; returns raw candidates, accepted signals and gate stats."""
    by_comp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        by_comp[c["competitor"]].append({**c["snippet"], "competitor": c["competitor"]})

    out: dict[str, dict[str, Any]] = {}
    for comp, snippets in by_comp.items():
        candidates = extractor._call_llm(comp, snippets, [])
        accepted, stats = extractor.filter_candidates(comp, candidates, snippets)
        out[comp] = {
            "snippets": snippets,
            "candidates": candidates,
            "accepted": [s.model_dump(mode="json") for s in accepted],
            "stats": stats,
        }
    return out


def match(cases: list[dict[str, Any]], accepted: list[dict[str, Any]], strict: bool) -> tuple[int, int, int]:
    """Greedy match accepted signals to gold labels by snippet URL (and type if strict). Returns tp, fp, fn."""
    gold: dict[str, list[str]] = {normalize_url(c["snippet"]["url"]): list(c["expected"]) for c in cases}
    tp = fp = 0
    for sig in accepted:
        labels = gold.get(normalize_url(sig["source_url"]))
        if not labels:
            fp += 1
            continue
        if strict:
            if sig["signal_type"] in labels:
                labels.remove(sig["signal_type"])
                tp += 1
            else:
                fp += 1
        else:
            labels.pop(0)
            tp += 1
    fn = sum(len(v) for v in gold.values())
    return tp, fp, fn


def _ratio(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


def score(cases: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    accepted = [s for r in results.values() for s in r["accepted"]]
    tp, fp, fn = match(cases, accepted, strict=True)
    ltp, lfp, lfn = match(cases, accepted, strict=False)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = round(2 * precision * recall / (precision + recall), 3) if precision and recall else 0.0

    hit_urls = {normalize_url(s["source_url"]) for s in accepted}
    negatives = [c for c in cases if not c["expected"]]
    rejected = [c for c in negatives if normalize_url(c["snippet"]["url"]) not in hit_urls]

    raw = grounded = 0
    for r in results.values():
        allowed = {normalize_url(s["url"]) for s in r["snippets"]}
        for cand in r["candidates"]:
            raw += 1
            url = cand.get("source_url") if isinstance(cand, dict) else None
            grounded += isinstance(url, str) and normalize_url(url) in allowed

    gates = defaultdict(int)
    for r in results.values():
        for k, v in r["stats"].items():
            gates[k] += v

    return {
        "cases": len(cases),
        "gold_signals": sum(len(c["expected"]) for c in cases),
        "accepted_signals": len(accepted),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_lenient": _ratio(ltp, ltp + lfp),
        "recall_lenient": _ratio(ltp, ltp + lfn),
        "negative_rejection": _ratio(len(rejected), len(negatives)),
        "url_grounding_pre_guard": _ratio(grounded, raw),
        "gates": dict(gates),
        "false_positive_snippets": sorted(
            c["id"] for c in negatives if normalize_url(c["snippet"]["url"]) in hit_urls),
    }


def judge_faithfulness(results: dict[str, dict[str, Any]], client: Any, model: str) -> dict[str, Any]:
    verdicts = []
    for r in results.values():
        by_url = {normalize_url(s["url"]): s for s in r["snippets"]}
        for sig in r["accepted"]:
            src = by_url[normalize_url(sig["source_url"])]
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": f"SOURCE:\n{src['title']}\n{src['content']}\n\n"
                                                f"CLAIM:\n{sig['title']}\n{sig['summary']}"},
                ],
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            verdicts.append({"title": sig["title"], "source_url": sig["source_url"],
                             "supported": data.get("supported") is True, "reason": data.get("reason", "")})
    supported = sum(v["supported"] for v in verdicts)
    return {"faithfulness": _ratio(supported, len(verdicts)), "judge_model": model, "verdicts": verdicts}


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)


def markdown_table(report: dict[str, Any]) -> str:
    m = report["metrics"]
    rows = [
        ("Extraction precision (strict)", m["precision"]),
        ("Extraction recall (strict)", m["recall"]),
        ("F1 (strict)", m["f1"]),
        ("Precision / recall (type-agnostic)", f"{_fmt(m['precision_lenient'])} / {_fmt(m['recall_lenient'])}"),
        ("Negative rejection (rumor, opinion, off-topic)", m["negative_rejection"]),
        ("URL grounding before guard", m["url_grounding_pre_guard"]),
        ("Faithfulness (LLM judge)", report.get("faithfulness", {}).get("faithfulness")),
    ]
    lines = [f"Model `{report['model']}`, {m['cases']} snippets, {m['gold_signals']} gold signals, "
             f"run {report['run_at'][:10]}", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {name} | {_fmt(v)} |" for name, v in rows]
    return "\n".join(lines)


def evaluate(cases: list[dict[str, Any]], client: Any, settings: Settings, judge: bool = False,
             judge_model: str | None = None) -> dict[str, Any]:
    results = run_extraction(cases, ExtractorAgent(client, settings))
    report: dict[str, Any] = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.openai_model,
        "confidence_threshold": settings.confidence_threshold,
        "metrics": score(cases, results),
        "accepted": {comp: r["accepted"] for comp, r in results.items()},
    }
    if judge:
        report["faithfulness"] = judge_faithfulness(results, client, judge_model or settings.openai_model)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate extraction precision, recall and faithfulness.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--model", help="Override OPENAI_MODEL for extraction")
    parser.add_argument("--judge", action="store_true", help="Also score faithfulness with an LLM judge")
    parser.add_argument("--judge-model", help="Judge model (default: the extraction model)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "latest.json")
    args = parser.parse_args(argv)

    from openai import OpenAI

    settings = get_settings()
    if args.model:
        settings = replace(settings, openai_model=args.model)
    report = evaluate(load_dataset(args.dataset), OpenAI(), settings, args.judge, args.judge_model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(markdown_table(report))
    print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()
