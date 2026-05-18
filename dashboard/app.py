"""
dashboard/app.py
=================
SentimenTrade — Interactive Streamlit Dashboard
------------------------------------------------
Real-time sentiment analysis and event study visualization.

Run locally:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.sentiment_analyzer import SentimentAnalyzer

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimenTrade",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 SentimenTrade")
    st.caption("AI-Driven Event Study & Cross-Asset Correlation Engine")
    st.divider()

    # User inputs
    ticker = st.text_input("Primary Ticker", value="AAPL").upper()
    benchmark = st.text_input("Benchmark Ticker", value="SPY").upper()
    
    # Date range
    end_date = st.date_input("End Date", value=datetime.now())
    start_date = st.date_input("Start Date", value=end_date - timedelta(days=365))

    st.divider()
    st.markdown("**Sentiment Input**")
    headlines_raw = st.text_area(
        "Paste news headlines (one per line)",
        height=200,
        placeholder="Apple beats earnings estimates…\nRecession fears mount…\nStock rallies on positive guidance…",
    )
    run_button = st.button("▶ Run Analysis", type="primary", use_container_width=True)

# ── Main Panel ───────────────────────────────────────────────────────────────
st.title("SentimenTrade Dashboard")
st.markdown("Analyze financial sentiment and market impact in real-time.")

if run_button:
    if not headlines_raw.strip():
        st.error("❌ Please enter at least one headline to analyze.")
        st.stop()

    # Parse headlines
    headlines: list[str] = [
        h.strip() for h in headlines_raw.split("\n") if h.strip()
    ]

    # Initialize analyzers
    with st.spinner("⏳ Loading FinBERT model..."):
        sentiment_analyzer = SentimentAnalyzer()

    with st.spinner(f"📊 Fetching price data for {ticker} and {benchmark}..."):
        data_loader = DataLoader(ticker=ticker, benchmark=benchmark)
        try:
            price_data = data_loader.fetch(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )
        except Exception as e:
            st.error(f"❌ Error fetching price data: {e}")
            st.stop()

    # Sentiment Analysis
    with st.spinner(f"🧠 Analyzing sentiment for {len(headlines)} headline(s)..."):
        try:
            sentiment_results = sentiment_analyzer.score(headlines)
        except Exception as e:
            st.error(f"❌ Error analyzing sentiment: {e}")
            st.stop()

    # ── Display Results ──────────────────────────────────────────────────────
    st.success("✅ Analysis complete!")
    st.divider()

    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_sentiment = sentiment_results["sentiment_score"].mean()
        st.metric("Avg Sentiment Score", f"{avg_sentiment:.2f}", "-1 to +1")
    
    with col2:
        positive_count = (sentiment_results["label"] == "positive").sum()
        st.metric("Positive Headlines", f"{positive_count}/{len(headlines)}")
    
    with col3:
        negative_count = (sentiment_results["label"] == "negative").sum()
        st.metric("Negative Headlines", f"{negative_count}/{len(headlines)}")
    
    with col4:
        neutral_count = (sentiment_results["label"] == "neutral").sum()
        st.metric("Neutral Headlines", f"{neutral_count}/{len(headlines)}")

    st.divider()

    # Sentiment Scores Visualization
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Price & Returns")
        
        # Prepare price chart data
        price_chart_data = price_data[[f"close_{ticker}"]].copy()
        price_chart_data.columns = [ticker]
        
        fig_price = px.line(
            price_chart_data,
            title=f"{ticker} Closing Price",
            labels={"value": "Price ($)", "index": "Date"},
        )
        fig_price.update_layout(hovermode="x unified", height=400)
        st.plotly_chart(fig_price, use_container_width=True)

    with col2:
        st.subheader("🧠 Sentiment Scores")
        
        # Sentiment score bar chart
        sentiment_display = sentiment_results.copy()
        sentiment_display["index"] = range(1, len(sentiment_display) + 1)
        sentiment_display["Sentiment"] = sentiment_display["sentiment_score"]
        
        fig_sentiment = px.bar(
            sentiment_display,
            x="index",
            y="Sentiment",
            color="label",
            title="Sentiment Score by Headline",
            labels={"index": "Headline #", "Sentiment": "Sentiment Score"},
            color_discrete_map={"positive": "#28a745", "negative": "#dc3545", "neutral": "#6c757d"}
        )
        fig_sentiment.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_sentiment, use_container_width=True)

    st.divider()

    # Log Returns Analysis
    st.subheader("📉 Daily Log Returns")
    
    log_ret_cols = [col for col in price_data.columns if col.startswith("log_ret_")]
    if log_ret_cols:
        returns_data = price_data[log_ret_cols].copy()
        returns_data.columns = [col.replace("log_ret_", "") for col in log_ret_cols]
        
        fig_returns = px.line(
            returns_data,
            title="Daily Log Returns Comparison",
            labels={"value": "Log Return", "index": "Date"},
        )
        fig_returns.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        fig_returns.update_layout(hovermode="x unified", height=400)
        st.plotly_chart(fig_returns, use_container_width=True)

    st.divider()

    # Detailed Results Table
    st.subheader("📋 Detailed Sentiment Analysis")
    
    display_df = sentiment_results[[
        "headline",
        "sentiment_score",
        "prob_positive",
        "prob_negative",
        "prob_neutral",
        "label"
    ]].copy()
    
    display_df.columns = [
        "Headline",
        "Score",
        "Positive %",
        "Negative %",
        "Neutral %",
        "Label"
    ]
    
    # Format percentages
    display_df["Positive %"] = (display_df["Positive %"] * 100).round(1).astype(str) + "%"
    display_df["Negative %"] = (display_df["Negative %"] * 100).round(1).astype(str) + "%"
    display_df["Neutral %"] = (display_df["Neutral %"] * 100).round(1).astype(str) + "%"
    display_df["Score"] = display_df["Score"].round(3)
    
    st.dataframe(display_df, use_container_width=True)

    st.divider()

    # Statistics
    st.subheader("📈 Statistical Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Sentiment Score", f"{sentiment_results['sentiment_score'].mean():.3f}")
        st.metric("Median Sentiment Score", f"{sentiment_results['sentiment_score'].median():.3f}")
    
    with col2:
        st.metric("Std Dev Sentiment", f"{sentiment_results['sentiment_score'].std():.3f}")
        st.metric("Min Sentiment", f"{sentiment_results['sentiment_score'].min():.3f}")
    
    with col3:
        st.metric("Max Sentiment", f"{sentiment_results['sentiment_score'].max():.3f}")
        st.metric("Price Volatility", f"{price_data[[col for col in price_data.columns if col.startswith('log_ret_')]].std().mean():.4f}")

else:
    # Default homepage
    st.info("👈 Enter a ticker, headline(s), and click **Run Analysis** to begin!", icon="ℹ️")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🚀 Quick Start
        1. **Enter ticker symbols** (e.g., AAPL, MSFT)
        2. **Paste financial headlines** (one per line)
        3. **Click Run Analysis**
        4. **Explore sentiment scores** and price correlations
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 What You'll See
        - **Sentiment Scores** (-1 to +1 scale)
        - **Price Charts** with historical data
        - **Log Returns** analysis
        - **Probability Distributions** (positive/negative/neutral)
        - **Statistical Summary** of results
        """)

st.sidebar.divider()
st.sidebar.caption("Built with ❤️ using FinBERT, Streamlit, and quantitative finance")
