# CompeteIQ

Multi-agent competitive-intelligence pipeline built with LangGraph. It searches the web
for news about a set of AI competitors, extracts source-grounded signals, and writes a
weekly markdown briefing backed by SQLite and ChromaDB memory.

> Work in progress. Being built stage by stage; full documentation lands in the final stage.

## Configuration

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and `TAVILY_API_KEY`.

| Setting | Default |
|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` |
| `MAX_SNIPPETS_PER_COMPETITOR` | `10` |
| `MIN_SIGNALS_PER_COMPETITOR` | `2` |
| `MAX_RESEARCH_RETRIES` | `2` |

## Development

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

## Docker

```bash
cp .env.example .env            # add your keys
docker compose up --build        # API on :8000, dashboard on :8501
docker compose run --rm api python main.py   # one pipeline run into the shared volume
```

## GitHub Actions

- `ci.yml`: pytest with coverage (threshold in `.coveragerc`) and a Docker build, on every push and PR. Needs no secrets.
- `weekly.yml`: runs the pipeline Mondays 08:00 UTC (or on demand). Restores `data/` from the newest cache,
  falls back to the latest `competeiq-data` artifact if the cache was evicted, then saves both after the run.
  Requires repository secrets `OPENAI_API_KEY` and `TAVILY_API_KEY`.
