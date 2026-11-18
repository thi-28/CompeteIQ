"""Static checks on the container setup, so CI catches drift without a Docker daemon."""

from pathlib import Path

import yaml

from config import Settings

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
DOCKERFILE = (ROOT / "Dockerfile").read_text()
DOCKERIGNORE = (ROOT / ".dockerignore").read_text().splitlines()


def test_services_share_one_data_volume_at_the_configured_path():
    api, dash = COMPOSE["services"]["api"], COMPOSE["services"]["dashboard"]
    assert "competeiq-data" in COMPOSE["volumes"]
    assert "competeiq-data:/app/data" in api["volumes"]
    env = api["environment"]
    assert env["DATA_DIR"] == "/app/data"
    assert env["SQLITE_DB_PATH"].startswith("/app/data/") and env["CHROMA_PERSIST_DIR"].startswith("/app/data/")
    # Defaults resolve inside /app/data too (WORKDIR /app + ./data).
    assert "WORKDIR /app" in DOCKERFILE and Settings().data_dir == "./data"
    # The dashboard is an API client: no volume, no secrets.
    assert "volumes" not in dash and "env_file" not in dash


def test_dashboard_reaches_api_over_compose_network_after_it_is_healthy():
    dash = COMPOSE["services"]["dashboard"]
    assert dash["environment"]["API_BASE_URL"] == "http://api:8000"
    assert dash["depends_on"]["api"]["condition"] == "service_healthy"
    assert "/health" in " ".join(COMPOSE["services"]["api"]["healthcheck"]["test"])
    assert "dashboard/app.py" in dash["command"] and "--server.address=0.0.0.0" in dash["command"]


def test_both_services_use_the_same_image():
    api, dash = COMPOSE["services"]["api"], COMPOSE["services"]["dashboard"]
    assert api["image"] == dash["image"] and api["build"] == dash["build"] == "."


def test_image_runs_as_non_root_and_keeps_secrets_and_data_out():
    assert "USER app" in DOCKERFILE
    assert DOCKERFILE.index("COPY requirements.txt") < DOCKERFILE.index("COPY . .")  # cached deps layer
    for pattern in (".env", "data/", ".venv/", ".git/"):
        assert pattern in DOCKERIGNORE
    assert "!.env.example" in DOCKERIGNORE


def test_runtime_requirements_exclude_test_tooling():
    runtime = (ROOT / "requirements.txt").read_text()
    dev = (ROOT / "requirements-dev.txt").read_text()
    assert "pytest" not in runtime
    assert "-r requirements.txt" in dev and "pytest-cov" in dev
