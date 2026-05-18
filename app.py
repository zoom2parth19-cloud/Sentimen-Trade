"""
dashboard/app.py
=================
SentimenTrade — Interactive Streamlit Dashboard  [Phase 2 Scaffold]
--------------------------------------------------------------------
This file contains the skeleton of the Streamlit dashboard.
Full implementation of charts, event-study plots, and correlation
heatmaps will be added in Phase 2.

Run locally:
    streamlit run dashboard/app.py
"""

import streamlit as st

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimenTrade",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 SentimenTrade")
    st.caption("AI-Driven Event Study & Cross-Asset Correlation Engine")
    st.divider()

    ticker = st.text_input("Primary Ticker", value="AAPL").upper()
    benchmark = st.text_input("Benchmark Ticker", value="SPY").upper()
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    st.divider()
    st.markdown("**Sentiment Input**")
    headlines_raw = st.text_area(
        "Paste news headlines (one per line)",
        height=200,
        placeholder="Apple beats earnings estimates…\nRecession fears mount…",
    )
    run_button = st.button("▶ Run Analysis", type="primary", use_container_width=True)

# ── Main Panel ───────────────────────────────────────────────────────────────
st.title("SentimenTrade Dashboard")
st.markdown(
    "Enter a ticker and headlines in the sidebar, then click **Run Analysis**."
)

if run_button:
    headlines: list[str] = [
        h.strip() for h in headlines_raw.split("\n") if h.strip()
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Price & Returns")
        st.info("Price chart — Phase 2", icon="🔧")

    with col2:
        st.subheader("🧠 Sentiment Scores")
        st.info("Sentiment bar chart — Phase 2", icon="🔧")

    st.subheader("📉 Event Study — Cumulative Abnormal Returns")
    st.info("CAR plot — Phase 2", icon="🔧")

    st.subheader("🔗 Cross-Asset Correlation Matrix")
    st.info("Heatmap — Phase 2", icon="🔧")
