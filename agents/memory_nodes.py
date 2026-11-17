"""
Graph nodes that read from and write to memory.

- MemoryReadNode: per-competitor retrieval from ChromaDB, using this run's
  snippet titles as the query. Only targets the supervisor's low-signal
  competitors on a retry pass.
- MemoryWriteNode: persists signals (SQLite + ChromaDB with dedup) and the
  briefing (SQLite + markdown file).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from models.schemas import AgentState, Briefing, Signal

logger = logging.getLogger(__name__)


class MemoryReadNode:
    def __init__(self, episodic: EpisodicMemory) -> None:
        self.episodic = episodic

    def __call__(self, state: AgentState) -> dict[str, Any]:
        snippets = state.get("snippets", {})
        targets = state.get("low_signal_competitors") or state.get("competitors", [])
        context = dict(state.get("memory_context", {}))
        errors: list[str] = []
        for comp in targets:
            titles = [s.get("title") or s["content"][:120] for s in snippets.get(comp, [])]
            try:
                context[comp] = self.episodic.retrieve(comp, titles)
            except Exception as exc:  # memory is an enhancement; never fail the run on it
                errors.append(f"memory_read:{comp}: {exc}")
                context.setdefault(comp, [])
        return {"memory_context": context, "errors": errors}


class MemoryWriteNode:
    def __init__(self, semantic: SemanticMemory, episodic: EpisodicMemory, briefings_dir: str | Path) -> None:
        self.semantic = semantic
        self.episodic = episodic
        self.briefings_dir = Path(briefings_dir)

    def __call__(self, state: AgentState) -> dict[str, Any]:
        run_id = state["run_id"]
        signals = [Signal.model_validate(s) for s in state.get("extracted_signals", [])]
        errors: list[str] = []

        self.semantic.save_signals(signals, run_id)
        self.semantic.save_briefing(Briefing(
            run_id=run_id,
            competitors_monitored=state["competitors"],
            signal_count=len(signals),
            content=state.get("briefing", ""),
        ))
        self.briefings_dir.mkdir(parents=True, exist_ok=True)
        (self.briefings_dir / f"{run_id}.md").write_text(state.get("briefing", ""), encoding="utf-8")

        try:
            result = self.episodic.store(signals, run_id)
            logger.info("Memory write: %d new, %d already known", len(result["stored"]), len(result["duplicates"]))
        except Exception as exc:  # SQLite is the source of truth; vector store failure is non-fatal
            errors.append(f"memory_write:episodic: {exc}")
        return {"errors": errors}
