"""
Episodic memory: past signals as vectors in ChromaDB.

- Retrieval is scoped per competitor with a `where` filter, so one competitor's
  history can't crowd out another's.
- Storage dedups two ways: a deterministic id (exact repeats are no-ops) and a
  cosine-similarity check against the same competitor's existing signals
  (near-duplicates, e.g. the same launch covered by two outlets).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable

import chromadb

from config import Settings, get_settings
from models.schemas import Signal
from tools.urls import normalize_url

logger = logging.getLogger(__name__)

COLLECTION_NAME = "competeiq_signals"

Embedder = Callable[[list[str]], list[list[float]]]


class OpenAIEmbedder:
    """Batched OpenAI embeddings. Model comes from settings (text-embedding-3-small)."""

    def __init__(self, model: str, client: Any | None = None) -> None:
        self.model = model
        self._client = client

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]


def signal_text(signal: Signal) -> str:
    return f"{signal.competitor} | {signal.signal_type} | {signal.title} | {signal.summary}"


def signal_id(signal: Signal) -> str:
    key = f"{signal.competitor.lower()}|{normalize_url(signal.source_url)}|{signal.title.strip().lower()}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


class EpisodicMemory:
    def __init__(
        self,
        persist_dir: str | None = None,
        embedder: Embedder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        path = Path(persist_dir or self.settings.chroma_persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=chromadb.Settings(anonymized_telemetry=False)
        )
        # embedding_function=None: we always pass vectors ourselves, and this
        # stops Chroma from downloading its default ONNX model.
        self._collection = self._client.get_or_create_collection(
            COLLECTION_NAME, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        self._embed = embedder or OpenAIEmbedder(self.settings.embedding_model)

    def count(self, competitor: str | None = None) -> int:
        if competitor is None:
            return self._collection.count()
        return len(self._collection.get(where={"competitor": competitor}, include=[])["ids"])

    def retrieve(
        self, competitor: str, query_texts: list[str] | None = None, top_k: int | None = None
    ) -> list[str]:
        """Top-k past signals for ONE competitor, most similar to the query texts
        (typically this run's snippet titles). Returns formatted strings for prompts."""
        top_k = top_k or self.settings.memory_top_k
        if self.count(competitor) == 0:
            return []
        query = " ".join(query_texts or []) or competitor
        [vector] = self._embed([query])
        res = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where={"competitor": competitor},
            include=["documents", "metadatas"],
        )
        return [
            f"[{meta['date_detected'][:10]}] {doc}"
            for doc, meta in zip(res["documents"][0], res["metadatas"][0])
        ]

    def store(self, signals: list[Signal], run_id: str) -> dict[str, Any]:
        """Embed and store signals, skipping exact and near duplicates.
        Returns {"stored": [...ids], "duplicates": [...ids]}."""
        stored: list[str] = []
        duplicates: list[str] = []
        if not signals:
            return {"stored": stored, "duplicates": duplicates}

        vectors = self._embed([signal_text(s) for s in signals])  # one batched call
        for signal, vector in zip(signals, vectors):
            sid = signal_id(signal)
            if self._is_duplicate(sid, signal.competitor, vector):
                duplicates.append(sid)
                continue
            self._collection.add(
                ids=[sid],
                embeddings=[vector],
                documents=[signal_text(signal)],
                metadatas=[{
                    "competitor": signal.competitor,
                    "signal_type": signal.signal_type,
                    "impact_assessment": signal.impact_assessment,
                    "source_url": signal.source_url,
                    "confidence": signal.confidence,
                    "date_detected": signal.date_detected.isoformat(),
                    "run_id": run_id,
                }],
            )
            stored.append(sid)

        logger.info("Episodic: stored %d, skipped %d duplicates (run %s)", len(stored), len(duplicates), run_id)
        return {"stored": stored, "duplicates": duplicates}

    def _is_duplicate(self, sid: str, competitor: str, vector: list[float]) -> bool:
        if self._collection.get(ids=[sid], include=[])["ids"]:
            return True
        res = self._collection.query(
            query_embeddings=[vector], n_results=1, where={"competitor": competitor}, include=["distances"]
        )
        distances = res["distances"][0]
        return bool(distances) and (1.0 - distances[0]) >= self.settings.dedup_similarity_threshold
