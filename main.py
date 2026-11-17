"""
CompeteIQ pipeline entry point: LangGraph wiring and CLI.

    START -> plan -> collect -> memory_read -> extract -> review
    review --(low-signal competitors with retries left)--> rewrite -> collect ...
    review --(otherwise)--> synthesize -> memory_write -> END

Usage:
    python main.py                       # default competitors from settings
    python main.py --competitors "Anthropic,Mistral"
"""

from __future__ import annotations

import argparse
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.collector import CollectorAgent
from agents.extractor import ExtractorAgent
from agents.memory_nodes import MemoryReadNode, MemoryWriteNode
from agents.supervisor import SupervisorAgent
from agents.synthesizer import SynthesizerAgent
from config import Settings, get_settings
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from models.schemas import AgentState, RunResult, utc_now

logger = logging.getLogger("competeiq")


@dataclass
class Dependencies:
    supervisor: SupervisorAgent
    collector: CollectorAgent
    extractor: ExtractorAgent
    synthesizer: SynthesizerAgent
    semantic: SemanticMemory
    episodic: EpisodicMemory
    settings: Settings

    @classmethod
    def default(cls, settings: Settings | None = None) -> "Dependencies":
        settings = settings or get_settings()
        return cls(
            supervisor=SupervisorAgent(settings=settings),
            collector=CollectorAgent(settings=settings),
            extractor=ExtractorAgent(settings=settings),
            synthesizer=SynthesizerAgent(settings=settings),
            semantic=SemanticMemory(settings.sqlite_db_path),
            episodic=EpisodicMemory(settings.chroma_persist_dir, settings=settings),
            settings=settings,
        )


def build_graph(deps: Dependencies):
    g = StateGraph(AgentState)
    g.add_node("plan", deps.supervisor.plan)
    g.add_node("collect", deps.collector)
    g.add_node("memory_read", MemoryReadNode(deps.episodic))
    g.add_node("extract", deps.extractor)
    g.add_node("review", deps.supervisor.review)
    g.add_node("rewrite", deps.supervisor.rewrite)
    g.add_node("synthesize", deps.synthesizer)
    g.add_node(
        "memory_write",
        MemoryWriteNode(deps.semantic, deps.episodic, Path(deps.settings.data_dir) / "briefings"),
    )

    g.add_edge(START, "plan")
    g.add_edge("plan", "collect")
    g.add_edge("collect", "memory_read")
    g.add_edge("memory_read", "extract")
    g.add_edge("extract", "review")
    g.add_conditional_edges("review", SupervisorAgent.route, {"rewrite": "rewrite", "synthesize": "synthesize"})
    g.add_edge("rewrite", "collect")
    g.add_edge("synthesize", "memory_write")
    g.add_edge("memory_write", END)
    return g.compile()


def recursion_limit(settings: Settings) -> int:
    # plan + first pass (4) + each retry (rewrite + 4) + synthesize + memory_write, with headroom
    return 1 + 4 + 5 * settings.max_research_retries + 2 + 5


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def run_pipeline(
    competitors: list[str] | None = None,
    deps: Dependencies | None = None,
    run_id: str | None = None,
) -> RunResult:
    deps = deps or Dependencies.default()
    settings = deps.settings
    competitors = competitors or settings.default_competitors
    run_id = run_id or new_run_id()
    started_at = utc_now()

    initial: AgentState = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "competitors": competitors,
        "errors": [],
    }
    final: dict[str, Any] = {}
    success = True
    try:
        final = build_graph(deps).invoke(initial, {"recursion_limit": recursion_limit(settings)})
    except Exception as exc:
        logger.exception("Pipeline run %s failed", run_id)
        final = {"errors": [f"fatal: {exc}"]}
        success = False

    result = RunResult(
        run_id=run_id,
        started_at=started_at,
        competitors=competitors,
        signal_count=len(final.get("extracted_signals", [])),
        error_count=len(final.get("errors", [])),
        research_retries={c: n for c, n in final.get("retry_counts", {}).items() if n},
        briefing_id=run_id if final.get("briefing") else None,
        success=success,
    )
    deps.semantic.save_run(result)
    for err in final.get("errors", []):
        logger.warning("run %s: %s", run_id, err)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CompeteIQ pipeline once.")
    parser.add_argument("--competitors", help="Comma-separated list (default: DEFAULT_COMPETITORS)")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    competitors = [c.strip() for c in args.competitors.split(",") if c.strip()] if args.competitors else None

    result = run_pipeline(competitors, Dependencies.default(settings))
    print(result.model_dump_json(indent=2))
    if result.briefing_id:
        print(f"Briefing: {Path(settings.data_dir) / 'briefings' / f'{result.briefing_id}.md'}")
    raise SystemExit(0 if result.success else 1)


if __name__ == "__main__":
    main()
