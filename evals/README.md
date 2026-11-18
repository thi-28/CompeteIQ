# Extraction evals

Measures how well the extractor turns snippets into signals, using the real
`ExtractorAgent` prompt and its schema, grounding and confidence gates.

```bash
python evals/run_eval.py            # precision, recall, negative rejection, URL grounding
python evals/run_eval.py --judge    # + LLM-judged faithfulness (one extra call per accepted signal)
```

Needs `OPENAI_API_KEY` only (no search calls). Writes the full report, including every
accepted signal and judge verdict, to `evals/results/latest.json`, and prints a markdown
table for the main README.

## Dataset

`dataset.json`: 20 hand-labeled snippets across four **fictional** companies (Corvid Labs,
Nimbus AI, Halyard, Tessellate) on the reserved `.example` domain, so no fixture can be
mistaken for real news about a real company. 15 gold signals, including one snippet
carrying two developments; 6 snippets should produce nothing:

| Negative kind | Why it must be rejected |
|---|---|
| Rumor (unnamed sources, "reportedly") | Prompt asks for confidence < 0.5; the 0.6 gate drops it |
| Opinion column | No concrete development |
| Off-topic | The development belongs to another company |
| Listicle mention, job posting | Mentions the company, reports nothing new |

Labeling rule: a gold signal is a concrete, confirmed development stated in the snippet,
typed with the closest `signal_type` (funding goes to `other`).

## Metrics

| Metric | Definition |
|---|---|
| Precision / recall (strict) | Accepted signal matches a gold label on the same snippet URL **and** type |
| Precision / recall (type-agnostic) | Same, ignoring type; the gap shows typing errors vs detection errors |
| Negative rejection | Share of no-signal snippets that produced zero accepted signals |
| URL grounding before guard | Share of raw model candidates citing a provided URL. The guard drops the rest, so stored signals are always grounded; this shows how often the model invents sources |
| Faithfulness | Share of accepted signals whose title and summary an LLM judge finds fully supported by the cited snippet |

Limits: 20 synthetic snippets give coarse numbers (one signal moves precision by about
0.07). The judge defaults to the extraction model, which can be lenient toward its own
output; pass `--judge-model` to use a different one.
