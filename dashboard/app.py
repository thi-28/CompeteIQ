"""
CompeteIQ Streamlit dashboard: briefings, signal explorer, trends, run history.

    streamlit run dashboard/app.py

Reads everything through the FastAPI service at API_BASE_URL.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `streamlit run dashboard/app.py`

from dashboard.data import (  # noqa: E402
    SINGLE_SERIES,
    SIGNAL_TYPES,
    ApiClient,
    ApiError,
    color_map,
    ordered_competitors,
    runs_frame,
    signals_frame,
    type_counts,
    week_tick_step,
    weekly_counts,
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
CONFIGURED = [c.strip() for c in os.getenv("DEFAULT_COMPETITORS", "Anthropic,Google,Meta,Mistral,Perplexity").split(",") if c.strip()]

st.set_page_config(page_title="CompeteIQ", layout="wide")
client = ApiClient(API_BASE_URL)


@st.cache_data(ttl=60, show_spinner=False)
def load(kind: str, **kw):
    return getattr(client, kind)(**kw)


def label(t: str) -> str:
    return t.replace("_", " ").title()


st.title("CompeteIQ")

try:
    health = load("health")
except ApiError as exc:
    st.error(f"{exc}\n\nStart the API with `uvicorn api.main:app` or set API_BASE_URL.")
    st.stop()

st.caption(f"API {API_BASE_URL} · model `{health['openai_model']}` · embeddings `{health['embedding_model']}`"
           + (f" · run `{health['active_run']}` in progress" if health.get("active_run") else ""))

tab_brief, tab_signals, tab_trends, tab_runs = st.tabs(["Briefings", "Signal explorer", "Trends", "Run history"])

# --- Briefings ---------------------------------------------------------------
with tab_brief:
    briefings = load("briefings")
    if not briefings:
        st.info("No briefings yet. Start a run from the Run history tab or with `python main.py`.")
    else:
        options = {f"{b['created_at'][:10]} · {b['signal_count']} signals · {b['run_id']}": b["run_id"] for b in briefings}
        choice = st.selectbox("Briefing", list(options))
        briefing = load("briefing", run_id=options[choice])
        if briefing:
            st.download_button("Download markdown", briefing["content"], file_name=f"briefing-{briefing['run_id']}.md")
            st.markdown(briefing["content"])

# --- Signal explorer ---------------------------------------------------------
all_df = signals_frame(load("signals", limit=1000))
competitors = ordered_competitors(all_df, CONFIGURED)
colors = color_map(competitors)

with tab_signals:
    c1, c2, c3, c4 = st.columns(4)
    f_comp = c1.multiselect("Competitor", competitors)
    f_type = c2.multiselect("Signal type", SIGNAL_TYPES, format_func=label)
    f_impact = c3.multiselect("Impact", ["high", "medium", "low"])
    f_conf = c4.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

    df = all_df
    if f_comp:
        df = df[df["competitor"].isin(f_comp)]
    if f_type:
        df = df[df["signal_type"].isin(f_type)]
    if f_impact:
        df = df[df["impact_assessment"].isin(f_impact)]
    df = df[df["confidence"] >= f_conf]

    st.caption(f"{len(df)} of {len(all_df)} signals")
    st.dataframe(
        df.sort_values("date_detected", ascending=False),
        hide_index=True,
        width="stretch",
        column_config={
            "date_detected": st.column_config.DatetimeColumn("Detected (UTC)", format="YYYY-MM-DD"),
            "competitor": "Competitor",
            "signal_type": "Type",
            "impact_assessment": "Impact",
            "title": "Title",
            "summary": "Summary",
            "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
            "source_url": st.column_config.LinkColumn("Source", display_text="open"),
            "run_id": "Run",
        },
    )

# --- Trends ------------------------------------------------------------------
with tab_trends:
    if all_df.empty:
        st.info("Trends appear after the first run with signals.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Signals stored", len(all_df))
        m2.metric("High impact", int((all_df["impact_assessment"] == "high").sum()))
        m3.metric("Competitors with signals", len(competitors))

        weekly = weekly_counts(all_df)
        fig = go.Figure()
        for comp in competitors:
            series = weekly[weekly["competitor"] == comp]
            fig.add_trace(go.Scatter(
                x=series["week"], y=series["signals"], name=comp, mode="lines+markers",
                line={"color": colors[comp], "width": 2}, marker={"size": 8},
                hovertemplate="%{y} signals<extra>" + comp + "</extra>",
            ))
        fig.update_layout(
            title="Signals per week by competitor", hovermode="x unified",
            yaxis={"title": "Signals", "rangemode": "tozero", "gridcolor": "rgba(128,128,128,0.15)"},
            # Ticks on the Monday week starts the points sit on (Plotly defaults to Sundays).
            xaxis={"title": "Week starting (UTC)", "showgrid": False, "tickformat": "%b %d",
                   "tick0": weekly["week"].min(), "dtick": week_tick_step(weekly)},
            legend={"orientation": "h", "y": -0.2}, margin={"t": 50, "l": 10, "r": 10},
        )
        st.plotly_chart(fig, width="stretch")

        trend_comp = st.selectbox("Signal types for", ["All competitors"] + competitors)
        scoped = all_df if trend_comp == "All competitors" else all_df[all_df["competitor"] == trend_comp]
        types = type_counts(scoped)
        bar = go.Figure(go.Bar(
            x=types["signals"], y=types["signal_type"].map(label), orientation="h",
            marker={"color": SINGLE_SERIES, "cornerradius": 4}, text=types["signals"], textposition="outside",
            hovertemplate="%{y}: %{x} signals<extra></extra>",
        ))
        bar.update_layout(title=f"Signal types: {trend_comp}", yaxis={"autorange": "reversed"},
                          xaxis={"title": "Signals", "gridcolor": "rgba(128,128,128,0.15)"},
                          bargap=0.35, margin={"t": 50, "l": 10, "r": 10})
        st.plotly_chart(bar, width="stretch")

        with st.expander("Table view"):
            st.dataframe(weekly.pivot(index="week", columns="competitor", values="signals"), width="stretch")
            st.dataframe(types.assign(signal_type=types["signal_type"].map(label)), hide_index=True)

# --- Run history ---------------------------------------------------------------
with tab_runs:
    with st.expander("Start a run"):
        run_comps = st.multiselect("Competitors (empty = defaults)", CONFIGURED)
        key = st.text_input("API key (only if the API requires one)", type="password")
        if st.button("Start run", type="primary"):
            try:
                started = client.start_run(run_comps or None, key or None)
                st.success(f"Run `{started['run_id']}` started. Refresh in a few minutes.")
                load.clear()
            except ApiError as exc:
                st.error(str(exc))

    runs = runs_frame(load("runs"))
    if runs.empty:
        st.info("No runs recorded yet.")
    else:
        st.dataframe(
            runs[["started_at", "status", "run_id", "competitors", "signal_count", "error_count",
                  "research_retries", "duration_s"]],
            hide_index=True,
            width="stretch",
            column_config={
                "started_at": st.column_config.DatetimeColumn("Started (UTC)", format="YYYY-MM-DD HH:mm"),
                "status": "Status",
                "run_id": "Run",
                "competitors": "Competitors",
                "signal_count": "Signals",
                "error_count": "Errors",
                "research_retries": "Re-searches",
                "duration_s": st.column_config.NumberColumn("Duration (s)", format="%d"),
            },
        )
