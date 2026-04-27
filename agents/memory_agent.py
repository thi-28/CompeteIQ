"""
Memory agent — reads from and writes to episodic memory (ChromaDB).

This node is invoked twice per pipeline run:
- READ mode (before extraction): retrieves top-K relevant past signals to
  provide context that helps the extractor identify truly novel findings.
- WRITE mode (after synthesis): embeds and stores all new extracted signals
  so future runs can build on today's intelligence.
"""

from __future__ import annotations

import logging
from typing import Any

from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from models.schemas import AgentState, Signal

logger = logging.getLogger(__name__)


class MemoryAgent:
    """
    Dual-mode memory agent that bridges the pipeline to persistent stores.

    Delegates vector operations to EpisodicMemory (ChromaDB) and structured
    persistence to SemanticMemory (SQLite).
    """

    def __init__(
        self,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
    ) -> None:
        """
        Initialise with optional pre-constructed memory stores.

        Args:
            episodic: EpisodicMemory instance (ChromaDB-backed).
            semantic: SemanticMemory instance (SQLite-backed).
        """
        self._episodic = episodic or EpisodicMemory()
        self._semantic = semantic or SemanticMemory()
        logger.info("MemoryAgent initialised.")

    def _read(self, state: AgentState) -> dict[str, Any]:
        """
        Retrieve relevant past signals as LLM context.

        Args:
            state: Current pipeline state.

        Returns:
            Partial state update with ``memory_context`` and routing.
        """
        competitors: list[str] = state.get("competitors", [])
        errors: list[str] = list(state.get("errors", []))
        try:
            context = self._episodic.retrieve_context(competitors)
            logger.info(
                "MemoryAgent READ: retrieved %d context snippets.", len(context)
            )
        except Exception as exc:
            msg = f"memory_agent:read: {exc}"
            logger.error(msg)
            errors.append(msg)
            context = []

        return {
            "memory_context": context,
            "next_agent": "extractor",
            "errors": errors,
        }

    def _write(self, state: AgentState) -> dict[str, Any]:
        """
        Embed and persist new extracted signals to ChromaDB and SQLite.

        Args:
            state: Current pipeline state containing ``extracted_signals``.

        Returns:
            Partial state update with ``next_agent`` set to ``end``.
        """
        run_id: str = state.get("run_id", "unknown")
        raw_signals: list[dict] = state.get("extracted_signals", [])
        errors: list[str] = list(state.get("errors", []))

        # Deserialise dicts back into Signal objects for type-safe operations
        signals: list[Signal] = []
        for raw in raw_signals:
            try:
                signals.append(Signal.model_validate(raw))
            except Exception as exc:
                msg = f"memory_agent:write:deserialise: {exc}"
                logger.warning(msg)
                errors.append(msg)

        try:
            written_chroma = self._episodic.store_signals(signals, run_id)
            logger.info(
                "MemoryAgent WRITE: stored %d signals in ChromaDB.", written_chroma
            )
        except Exception as exc:
            msg = f"memory_agent:write:chroma: {exc}"
            logger.error(msg)
            errors.append(msg)

        try:
            written_sql = self._semantic.save_signals(signals, run_id)
            logger.info(
                "MemoryAgent WRITE: stored %d signals in SQLite.", written_sql
            )
        except Exception as exc:
            msg = f"memory_agent:write:sqlite: {exc}"
            logger.error(msg)
            errors.append(msg)

        return {
            "next_agent": "end",
            "errors": errors,
        }

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Dispatch to read or write mode based on ``state["memory_mode"]``.

        Args:
            state: Current pipeline state.

        Returns:
            Partial state update dict.
        """
        mode = state.get("memory_mode", "read")
        logger.info("MemoryAgent: running in mode='%s'.", mode)
        if mode == "write":
            return self._write(state)
        return self._read(state)


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

_memory_agent: MemoryAgent | None = None


def memory_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node wrapper for the MemoryAgent.

    The agent singleton is lazily initialised on first call so that importing
    this module does not require OPENAI_API_KEY or ChromaDB to be initialised.

    Args:
        state: Shared pipeline state passed by LangGraph.

    Returns:
        Partial state update dict.
    """
    global _memory_agent
    if _memory_agent is None:
        _memory_agent = MemoryAgent()
    return _memory_agent.run(state)
