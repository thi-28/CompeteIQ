"""Keep the README honest: defaults match config.py, one model name everywhere, routes exist."""

import re
from dataclasses import fields
from pathlib import Path

from api.main import create_app
from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_OPENAI_MODEL, Settings

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()


def test_readme_uses_only_the_configured_model_names():
    assert f"`{DEFAULT_OPENAI_MODEL}`" in README and f"`{DEFAULT_EMBEDDING_MODEL}`" in README
    mentioned = set(re.findall(r"gpt-[\w.-]+|text-embedding-[\w-]+", README))
    assert mentioned == {DEFAULT_OPENAI_MODEL, DEFAULT_EMBEDDING_MODEL}


def test_readme_settings_table_matches_config_defaults():
    rows = dict(re.findall(r"^\| `([A-Z_]+)` \| `([^`]+)` \|", README, flags=re.M))
    defaults = {f.name.upper(): getattr(Settings(), f.name) for f in fields(Settings)}
    defaults["DEFAULT_COMPETITORS"] = ",".join(Settings().default_competitors)
    assert len(rows) >= 9
    for key, documented in rows.items():
        if key in defaults:
            assert documented == str(defaults[key]), key


def test_readme_api_table_matches_routes():
    documented = set(re.findall(r"^\| `(GET|POST)` \| `([^`]+)` \|", README, flags=re.M))
    app = create_app(semantic=object(), runner=lambda *a: None)
    actual = {(m, r.path) for r in app.routes if r.path not in ("/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc")
              for m in getattr(r, "methods", ()) if m in ("GET", "POST")}
    assert documented == actual


def test_readme_has_no_fabricated_results():
    assert "_TBD_" in README and "Placeholder" in README
    assert "news.example" not in README  # eval fixtures stay out of the sample-output section
