"""
CompeteIQ Streamlit Dashboard.

Four-page intelligence dashboard:
  Page 1 — Latest Briefing: formatted markdown of the most recent briefing.
  Page 2 — Signal Explorer: filterable, sortable table of all signals.
  Page 3 — Trend Charts: Plotly charts for signal frequency, type distribution,
            and confidence trends.
  Page 4 — Run History: table of all pipeline runs with key metrics.

Sidebar: "Run Pipeline Now" button that calls POST /run.

Start with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="CompeteIQ",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _api_get(path: str, params: dict | None = None) -> list | dict | None:
    """
    Make a GET request to the CompeteIQ API.

    Args:
        path: API path (e.g. '/briefings').
        params: Optional query parameters.

    Returns:
        Parsed JSON response, or None on error.
    """
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to CompeteIQ API at {API_BASE}. "
            "Is the server running? Start it with: uvicorn api.main:app --reload"
        )
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def _api_post(path: str, payload: dict) -> dict | None:
    """
    Make a POST request to the CompeteIQ API.

    Args:
        path: API path.
        payload: JSON-serialisable request body.

    Returns:
        Parsed JSON response, or None on error.
    """
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.shields.io/badge/CompeteIQ-v1.0-blue", width=200)
    st.title("CompeteIQ")
    st.caption("Multi-Agent Competitive Intelligence")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Latest Briefing", "Signal Explorer", "Trend Charts", "Run History"],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Run Pipeline")
    competitor_input = st.text_input(
        "Competitors (comma-separated)",
        value="Anthropic,Google,Meta,Mistral,Perplexity",
        help="Override the default competitor list for this run.",
    )
    if st.button("▶ Run Pipeline Now", type="primary", use_container_width=True):
        competitors = [c.strip() for c in competitor_input.split(",") if c.strip()]
        with st.spinner("Enqueueing pipeline run..."):
            result = _api_post("/run", {"competitors": competitors})
        if result:
            st.success(f"Run started!\nID: `{result.get('run_id', '?')}`")
            st.info(result.get("message", ""))

    st.divider()
    # API connectivity indicator
    try:
        _r = requests.get(f"{API_BASE}/health", timeout=3)
        _r.raise_for_status()
        st.success("API connected", icon="✅")
    except Exception:
        st.error(f"API unreachable at {API_BASE}\nStart it with:\n`uvicorn api.main:app --reload`", icon="❌")
    st.caption(f"API: {API_BASE}")

# ---------------------------------------------------------------------------
# Page 1 — Latest Briefing
# ---------------------------------------------------------------------------

def _importance_badge(signal_count: int) -> str:
    """Return a colour-coded importance label based on signal count."""
    if signal_count >= 7:
        return "🔴 High"
    if signal_count >= 4:
        return "🟡 Medium"
    return "🟢 Low"


def _format_age(created_at: str) -> str:
    """Return a human-readable relative age string (e.g. '2 days ago')."""
    try:
        dt = datetime.fromisoformat(created_at)
        delta = datetime.utcnow() - dt
        days = delta.days
        if days == 0:
            hours = delta.seconds // 3600
            return f"{hours}h ago" if hours > 0 else "just now"
        if days == 1:
            return "yesterday"
        return f"{days} days ago"
    except Exception:
        return created_at


if page == "Latest Briefing":
    st.title("Intelligence Briefings")

    all_briefings = _api_get("/briefings", params={"limit": 10})

    if all_briefings and isinstance(all_briefings, list) and len(all_briefings) > 0:
        # Sort by created_at descending (API already does this, but be safe)
        def _sort_key(b: dict) -> str:
            return b.get("created_at", "") or ""

        all_briefings = sorted(all_briefings, key=_sort_key, reverse=True)

        # --- Summary row metrics ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Briefings", len(all_briefings))
        col2.metric("Latest", _format_age(all_briefings[0].get("created_at", "")))
        total_signals = sum(b.get("signal_count", 0) for b in all_briefings)
        col3.metric("Total Signals", total_signals)
        competitors_set = set()
        for b in all_briefings:
            competitors_set.update(b.get("competitors_monitored", []))
        col4.metric("Competitors Tracked", len(competitors_set))

        st.divider()

        # --- Briefing index (date-sorted list) ---
        st.subheader("Briefing History")
        briefing_options: dict[str, dict] = {}
        for b in all_briefings:
            created = b.get("created_at", "")
            try:
                dt = datetime.fromisoformat(created)
                date_label = dt.strftime("%b %d, %Y")
            except Exception:
                date_label = created[:10]

            sig_count = b.get("signal_count", 0)
            badge = _importance_badge(sig_count)
            age = _format_age(created)
            label = f"{date_label} — {sig_count} signals — {badge}  ({age})"
            briefing_options[label] = b

        selected_label = st.radio(
            "Select a briefing to read:",
            options=list(briefing_options.keys()),
            index=0,
        )
        selected = briefing_options[selected_label]

        st.divider()

        # --- Selected briefing detail ---
        created = selected.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created)
            date_str = dt.strftime("%B %d, %Y at %H:%M UTC")
        except Exception:
            date_str = created

        c1, c2, c3 = st.columns(3)
        c1.metric("Date", date_str)
        c2.metric("Signals", selected.get("signal_count", 0))
        c3.metric("Importance", _importance_badge(selected.get("signal_count", 0)))

        competitors_str = ", ".join(selected.get("competitors_monitored", []))
        if competitors_str:
            st.caption(f"Competitors monitored: {competitors_str}")

        st.divider()
        st.markdown(selected.get("content", "*No content available.*"))

        with st.expander("Metadata"):
            st.json({k: v for k, v in selected.items() if k != "content"})

    else:
        st.info(
            "No briefings found. Run the pipeline to generate your first briefing.",
            icon="ℹ️",
        )

# ---------------------------------------------------------------------------
# Page 2 — Signal Explorer
# ---------------------------------------------------------------------------

elif page == "Signal Explorer":
    st.title("Signal Explorer")

    # Filter controls
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        filter_competitor = st.text_input("Filter by Competitor", placeholder="e.g. Anthropic")
    with col2:
        filter_type = st.selectbox(
            "Signal Type",
            options=[
                "",
                "product_launch",
                "pricing_change",
                "partnership",
                "research_release",
                "executive_move",
                "market_expansion",
                "other",
            ],
            format_func=lambda x: "All types" if x == "" else x.replace("_", " ").title(),
        )
    with col3:
        limit = st.number_input("Max results", min_value=10, max_value=500, value=100)

    params: dict = {"limit": limit}
    if filter_competitor:
        params["competitor"] = filter_competitor
    if filter_type:
        params["signal_type"] = filter_type

    data = _api_get("/signals", params=params)
    if data and isinstance(data, list):
        df = pd.DataFrame(data)
        if not df.empty:
            # Reorder and rename columns for display
            display_cols = [
                "competitor",
                "signal_type",
                "title",
                "impact_assessment",
                "confidence",
                "date_detected",
                "source_url",
            ]
            existing = [c for c in display_cols if c in df.columns]
            df = df[existing]

            # Format columns
            if "confidence" in df.columns:
                df["confidence"] = df["confidence"].apply(lambda x: f"{x:.0%}")
            if "signal_type" in df.columns:
                df["signal_type"] = df["signal_type"].str.replace("_", " ").str.title()
            if "impact_assessment" in df.columns:
                df["impact_assessment"] = df["impact_assessment"].str.upper()

            st.metric("Signals shown", len(df))
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "source_url": st.column_config.LinkColumn("Source URL"),
                    "title": st.column_config.TextColumn("Title", width="large"),
                },
                hide_index=True,
            )
        else:
            st.info("No signals match your filters.", icon="ℹ️")
    else:
        st.info("No signals in the database yet. Run the pipeline first.", icon="ℹ️")

# ---------------------------------------------------------------------------
# Page 3 — Trend Charts
# ---------------------------------------------------------------------------

elif page == "Trend Charts":
    st.title("Trend Analysis")

    data = _api_get("/signals", params={"limit": 500})
    if data and isinstance(data, list) and len(data) > 0:
        df = pd.DataFrame(data)

        if "date_detected" in df.columns:
            df["date_detected"] = pd.to_datetime(df["date_detected"], errors="coerce")
            df["week"] = df["date_detected"].dt.to_period("W").astype(str)

        col1, col2 = st.columns(2)

        # Chart 1 — Signal frequency by competitor
        with col1:
            st.subheader("Signals by Competitor")
            if "competitor" in df.columns:
                comp_counts = df["competitor"].value_counts().reset_index()
                comp_counts.columns = ["competitor", "count"]
                fig1 = px.bar(
                    comp_counts,
                    x="competitor",
                    y="count",
                    color="competitor",
                    labels={"count": "Signal Count", "competitor": "Competitor"},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig1.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig1, use_container_width=True)

        # Chart 2 — Signal type distribution
        with col2:
            st.subheader("Signal Type Distribution")
            if "signal_type" in df.columns:
                type_counts = df["signal_type"].value_counts().reset_index()
                type_counts.columns = ["signal_type", "count"]
                type_counts["label"] = type_counts["signal_type"].str.replace("_", " ").str.title()
                fig2 = px.pie(
                    type_counts,
                    names="label",
                    values="count",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    hole=0.35,
                )
                fig2.update_layout(height=350)
                st.plotly_chart(fig2, use_container_width=True)

        # Chart 3 — Signal frequency over time
        st.subheader("Signal Volume Over Time")
        if "week" in df.columns and "competitor" in df.columns:
            weekly = (
                df.groupby(["week", "competitor"])
                .size()
                .reset_index(name="count")
            )
            fig3 = px.line(
                weekly,
                x="week",
                y="count",
                color="competitor",
                markers=True,
                labels={"count": "Signals", "week": "Week"},
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig3.update_layout(height=350, xaxis_tickangle=-30)
            st.plotly_chart(fig3, use_container_width=True)

        # Chart 4 — Average confidence trend
        st.subheader("Average Confidence Score by Competitor")
        if "confidence" in df.columns and "competitor" in df.columns:
            conf_df = df.groupby("competitor")["confidence"].mean().reset_index()
            conf_df.columns = ["competitor", "avg_confidence"]
            fig4 = px.bar(
                conf_df.sort_values("avg_confidence", ascending=False),
                x="competitor",
                y="avg_confidence",
                color="competitor",
                range_y=[0, 1],
                labels={"avg_confidence": "Avg Confidence", "competitor": "Competitor"},
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig4.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig4, use_container_width=True)

    else:
        st.info("No signal data available for charts. Run the pipeline first.", icon="ℹ️")

# ---------------------------------------------------------------------------
# Page 4 — Run History
# ---------------------------------------------------------------------------

elif page == "Run History":
    st.title("Pipeline Run History")

    data = _api_get("/runs", params={"limit": 50})
    if data and isinstance(data, list) and len(data) > 0:
        df = pd.DataFrame(data)

        # Summary metrics
        total_runs = len(df)
        success_runs = df["success"].sum() if "success" in df.columns else 0
        total_signals = df["signal_count"].sum() if "signal_count" in df.columns else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Runs", total_runs)
        col2.metric("Successful", int(success_runs))
        col3.metric("Total Signals", int(total_signals))
        col4.metric(
            "Success Rate",
            f"{100 * success_runs / total_runs:.0f}%" if total_runs > 0 else "N/A",
        )

        st.divider()

        # Format display columns
        display_cols = [
            "run_id",
            "started_at",
            "completed_at",
            "competitors",
            "signal_count",
            "error_count",
            "success",
        ]
        existing = [c for c in display_cols if c in df.columns]
        display_df = df[existing].copy()
        if "competitors" in display_df.columns:
            display_df["competitors"] = display_df["competitors"].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else x
            )

        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "run_id": st.column_config.TextColumn("Run ID"),
                "success": st.column_config.CheckboxColumn("Success"),
                "signal_count": st.column_config.NumberColumn("Signals"),
                "error_count": st.column_config.NumberColumn("Errors"),
            },
            hide_index=True,
        )
    else:
        st.info("No run history found. Trigger a pipeline run to get started.", icon="ℹ️")
