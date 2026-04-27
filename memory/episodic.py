"""
Episodic memory store backed by ChromaDB.

Provides vector-similarity search over past competitive intelligence signals
so that each pipeline run can retrieve relevant historical context before
extraction, enabling deduplication and trend awareness.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from models.schemas import Signal

logger = logging.getLogger(__name__)

# Similarity threshold above which a candidate signal is considered a duplicate
DUPLICATE_THRESHOLD = 0.95
# Number of past signals injected into each run's context
TOP_K_RETRIEVAL = 5
# ChromaDB collection name
COLLECTION_NAME = "competeiq_signals"


class EpisodicMemory:
    """
    Vector store for competitive intelligence signals.

    Uses ChromaDB for persistent storage and OpenAI embeddings
    (text-embedding-ada-002) for semantic similarity search.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        openai_client: OpenAI | None = None,
    ) -> None:
        """
        Initialise the ChromaDB client and embedding model.

        Args:
            persist_dir: Directory for ChromaDB on-disk persistence.
                         Defaults to the CHROMA_PERSIST_DIR env var or ./data/chroma.
            openai_client: Optional pre-constructed OpenAI client (useful for testing).
        """
        self._persist_dir = persist_dir or os.getenv(
            "CHROMA_PERSIST_DIR", "./data/chroma"
        )
        os.makedirs(self._persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._openai = openai_client or OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        logger.info(
            "EpisodicMemory initialised — persist_dir=%s, collection=%s",
            self._persist_dir,
            COLLECTION_NAME,
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """
        Compute an OpenAI embedding for the given text.

        Args:
            text: The input string to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        response = self._openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
        )
        return response.data[0].embedding

    @staticmethod
    def _signal_to_text(signal: Signal) -> str:
        """
        Convert a Signal to a plain-text representation suitable for embedding.

        Args:
            signal: The Signal instance to serialise.

        Returns:
            A compact string capturing the most semantically meaningful fields.
        """
        return (
            f"{signal.competitor} | {signal.signal_type} | "
            f"{signal.title} | {signal.summary}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve_context(
        self,
        competitors: list[str],
        top_k: int = TOP_K_RETRIEVAL,
    ) -> list[str]:
        """
        Retrieve the most relevant past signals for the given competitors.

        Builds a composite query string from the competitor names and fetches
        the top-K nearest neighbours from ChromaDB.

        Args:
            competitors: List of competitor names to query for.
            top_k: Maximum number of past signals to return.

        Returns:
            A list of human-readable signal summaries for LLM context injection.
        """
        if self._collection.count() == 0:
            logger.debug("Episodic memory is empty — no context to retrieve.")
            return []

        query_text = " ".join(competitors)
        try:
            embedding = self._embed(query_text)
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return []

        context: list[str] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            similarity = 1.0 - dist  # cosine distance → similarity
            context.append(
                f"[{meta.get('competitor', '?')} | {meta.get('signal_type', '?')} "
                f"| similarity={similarity:.2f}] {doc}"
            )

        logger.info(
            "Retrieved %d past signals for competitors: %s",
            len(context),
            competitors,
        )
        return context

    def is_duplicate(self, signal: Signal) -> bool:
        """
        Check whether a semantically identical signal is already stored.

        Args:
            signal: The Signal to check.

        Returns:
            True if a near-duplicate exists above DUPLICATE_THRESHOLD.
        """
        if self._collection.count() == 0:
            return False

        try:
            text = self._signal_to_text(signal)
            embedding = self._embed(text)
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["distances"],
            )
            distances = results.get("distances", [[]])[0]
            if distances:
                similarity = 1.0 - distances[0]
                return similarity >= DUPLICATE_THRESHOLD
        except Exception as exc:
            logger.error("Duplicate check failed: %s", exc)

        return False

    def store_signals(self, signals: list[Signal], run_id: str) -> int:
        """
        Embed and persist a list of signals to ChromaDB.

        Skips any signal that is already stored above the duplicate threshold.

        Args:
            signals: List of validated Signal instances to store.
            run_id: The run_id used to tag metadata for provenance.

        Returns:
            The number of signals actually written (after deduplication).
        """
        stored = 0
        for signal in signals:
            if self.is_duplicate(signal):
                logger.debug(
                    "Skipping duplicate signal: %s — %s", signal.competitor, signal.title
                )
                continue

            doc_id = f"{run_id}_{signal.competitor}_{signal.title[:40]}".replace(
                " ", "_"
            )
            text = self._signal_to_text(signal)
            try:
                embedding = self._embed(text)
                self._collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[text],
                    metadatas={
                        "competitor": signal.competitor,
                        "signal_type": signal.signal_type,
                        "impact_assessment": signal.impact_assessment,
                        "confidence": signal.confidence,
                        "source_url": signal.source_url,
                        "date_detected": signal.date_detected.isoformat(),
                        "run_id": run_id,
                    },
                )
                stored += 1
            except Exception as exc:
                logger.error(
                    "Failed to store signal '%s': %s", signal.title, exc
                )

        logger.info(
            "Stored %d/%d signals for run_id=%s", stored, len(signals), run_id
        )
        return stored

    def count(self) -> int:
        """Return the total number of signals currently stored in ChromaDB."""
        return self._collection.count()
