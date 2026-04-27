"""
CompeteIQ — entry point and LangGraph pipeline definition.

Run the full competitive intelligence pipeline:
    python main.py

Environment variables (see .env.example):
    OPENAI_API_KEY, TAVILY_API_KEY, CHROMA_PERSIST_DIR,
    SQLITE_DB_PATH, DEFAULT_COMPETITORS, LOG_LEVEL
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import datetime

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from agents.supervisor import supervisor_node
from agents.collector import collector_node
from agents.extractor import extractor_node
from agents.synthesizer import synthesizer_node
from agents.memory_agent import memory_node
from memory.semantic import SemanticMemory
from models.schemas import AgentState, Briefing, RunResult

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing logic for conditional edges
# ---------------------------------------------------------------------------


def route_from_supervisor(state: AgentState) -> str:
    """
    Conditional edge router: reads ``next_agent`` from state and returns
    the name of the next LangGraph node.

    Args:
        state: Current pipeline state.

    Returns:
        Node name string consumed by LangGraph for edge selection.
    """
    next_node = state.get("next_agent", "end")
    logger.debug("Router: next_agent='%s'", next_node)
    mapping = {
        "collector": "collector",
        "memory_read": "memory_read",
        "extractor": "extractor",
        "synthesizer": "synthesizer",
        "memory_write": "memory_write",
        "end": END,
    }
    return mapping.get(next_node, END)


def route_from_collector(state: AgentState) -> str:
    """Route after the collector node — always proceeds to memory_read."""
    return "supervisor"


def route_from_memory(state: AgentState) -> str:
    """
    Route after a memory node invocation based on the mode that was used.

    READ mode → supervisor (which will route to extractor).
    WRITE mode → END.
    """
    next_node = state.get("next_agent", "end")
    if next_node == "extractor":
        return "supervisor"
    if next_node == "end":
        return END
    return END


def route_after_extraction(state: AgentState) -> str:
    """Route after extractor — always back to supervisor for dispatch to synthesizer."""
    return "supervisor"


def route_after_synthesis(state: AgentState) -> str:
    """Route after synthesizer — always back to supervisor for dispatch to memory_write."""
    return "supervisor"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """
    Construct and compile the CompeteIQ LangGraph StateGraph.

    Graph topology:
        supervisor → collector → supervisor → memory_read → supervisor
        → extractor → supervisor → synthesizer → supervisor → memory_write → END

    Returns:
        A compiled LangGraph application ready to invoke.
    """
    checkpointer = MemorySaver()
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("collector", collector_node)
    graph.add_node("memory_read", memory_node)
    graph.add_node("memory_write", memory_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Edges from supervisor (conditional based on next_agent)
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "collector": "collector",
            "memory_read": "memory_read",
            "extractor": "extractor",
            "synthesizer": "synthesizer",
            "memory_write": "memory_write",
            END: END,
        },
    )

    # Collector → back to supervisor (supervisor will route to memory_read)
    graph.add_edge("collector", "supervisor")

    # Memory read → back to supervisor (supervisor will route to extractor)
    graph.add_edge("memory_read", "supervisor")

    # Extractor → back to supervisor (supervisor will route to synthesizer)
    graph.add_edge("extractor", "supervisor")

    # Synthesizer → back to supervisor (supervisor will route to memory_write)
    graph.add_edge("synthesizer", "supervisor")

    # Memory write → END
    graph.add_edge("memory_write", END)

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(
    competitors: list[str] | None = None,
    run_id: str | None = None,
) -> dict:
    """
    Execute the full CompeteIQ competitive intelligence pipeline.

    Args:
        competitors: List of competitor names to monitor.
                     Defaults to DEFAULT_COMPETITORS env var or built-in list.
        run_id: Optional UUID to use as the run identifier and LangGraph thread_id.
                Defaults to a fresh UUID4.

    Returns:
        Dict containing run_id, briefing text, signal count, and errors.
    """
    if competitors is None:
        env_competitors = os.getenv(
            "DEFAULT_COMPETITORS", "Anthropic,Google,Meta,Mistral,Perplexity"
        )
        competitors = [c.strip() for c in env_competitors.split(",") if c.strip()]

    run_id = run_id or str(uuid.uuid4())
    started_at = datetime.utcnow()

    logger.info("=" * 60)
    logger.info("CompeteIQ pipeline starting — run_id=%s", run_id)
    logger.info("Competitors: %s", competitors)
    logger.info("=" * 60)

    app = build_graph()

    initial_state: AgentState = {
        "competitors": competitors,
        "search_queries": {},
        "raw_signals": {},
        "extracted_signals": [],
        "memory_context": [],
        "memory_mode": "read",
        "briefing": "",
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "next_agent": "supervisor",
        "errors": [],
    }

    config = {"configurable": {"thread_id": run_id}}

    try:
        final_state = app.invoke(initial_state, config=config)
    except Exception as exc:
        logger.exception("Pipeline failed with unhandled exception: %s", exc)
        final_state = {**initial_state, "errors": [f"pipeline:fatal: {exc}"]}

    # Extract results
    briefing_text: str = final_state.get("briefing", "")
    extracted: list[dict] = final_state.get("extracted_signals", [])
    errors: list[str] = final_state.get("errors", [])

    # Persist run and briefing to SQLite
    db = SemanticMemory()
    completed_at = datetime.utcnow()

    run_result = RunResult(
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        competitors=competitors,
        signal_count=len(extracted),
        error_count=len(errors),
        briefing_id=run_id if briefing_text else None,
        success=len([e for e in errors if "fatal" in e]) == 0,
    )
    db.save_run(run_result)

    if briefing_text:
        briefing_obj = Briefing(
            run_id=run_id,
            created_at=completed_at,
            competitors_monitored=competitors,
            signal_count=len(extracted),
            content=briefing_text,
        )
        db.save_briefing(briefing_obj)

    logger.info("=" * 60)
    logger.info(
        "Pipeline complete — signals=%d, errors=%d, run_id=%s",
        len(extracted),
        len(errors),
        run_id,
    )
    logger.info("=" * 60)

    if errors:
        logger.warning("Errors encountered:\n  %s", "\n  ".join(errors))

    if briefing_text:
        print("\n" + briefing_text)

    return {
        "run_id": run_id,
        "briefing": briefing_text,
        "signal_count": len(extracted),
        "error_count": len(errors),
        "errors": errors,
        "competitors": competitors,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CompeteIQ — multi-agent competitive intelligence pipeline"
    )
    parser.add_argument(
        "--competitors",
        nargs="+",
        help="Competitors to monitor (overrides DEFAULT_COMPETITORS env var)",
    )
    parser.add_argument(
        "--run-id",
        help="Optional run ID (UUID). Auto-generated if not provided.",
    )
    args = parser.parse_args()

    result = run_pipeline(
        competitors=args.competitors,
        run_id=args.run_id,
    )
    sys.exit(0 if result["error_count"] == 0 else 1)
