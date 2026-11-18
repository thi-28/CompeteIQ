# One image, two services (api + dashboard); docker-compose picks the command.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so code edits don't invalidate the pip layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run as non-root. /app/data is created and owned here so a fresh named volume
# mounted on it inherits this ownership and stays writable.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data \
    && chown -R app:app /app/data
USER app

EXPOSE 8000 8501

# Default: the API. The dashboard service overrides this command.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
