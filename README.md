# SentimenTrade 📈🧠
### An AI-Driven Event Study & Cross-Asset Correlation Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square&logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FinBERT-yellow?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

**SentimenTrade** is a quantitative research platform that combines financial NLP with classical
event-study methodology. It uses **FinBERT** — a BERT model fine-tuned on financial text — to
score news headlines, then measures whether sentiment shocks produce statistically significant
abnormal returns in equity prices.

The engine answers questions like:
- *Did negative earnings headlines cause a measurable price dislocation?*
- *How correlated is a stock's sentiment score with its benchmark-adjusted returns?*
- *What is the average cumulative abnormal return (CAR) over a [-5, +5] day event window?*

---

## Project Structure

```
SentimenTrade/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Market data pipeline (yfinance + log returns)
│   ├── sentiment_analyzer.py   # FinBERT NLP engine (Hugging Face)
│   ├── event_study.py          # Abnormal return & CAR computation  [Phase 2]
│   └── correlation_engine.py   # Cross-asset correlation matrix     [Phase 2]
│
├── dashboard/
│   └── app.py                  # Streamlit interactive dashboard     [Phase 2]
│
├── data/
│   ├── raw/                    # Raw OHLCV price downloads
│   └── processed/              # Cleaned returns & sentiment scores
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── tests/
│   ├── test_data_loader.py
│   └── test_sentiment_analyzer.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Methodology

### 1. Data Pipeline
Historical OHLCV data is downloaded via `yfinance`. Daily **log returns** are computed as:

```
r_t = ln(P_t / P_{t-1})
```

Log returns are preferred over simple returns in quantitative finance because they are
time-additive, approximately normally distributed, and more numerically stable for modeling.

### 2. Sentiment Engine (FinBERT)
Each headline is passed through `ProsusAI/finbert`, a BERT-based model fine-tuned on ~10,000
financial news sentences. The output is a probability distribution over three classes:
`{positive, negative, neutral}`. A scalar **sentiment score** is derived as:

```
score = P(positive) - P(negative)  ∈ [-1, +1]
```

### 3. Event Study (Phase 2)
Using the Market Model, expected returns are estimated from a pre-event estimation window.
Abnormal returns (AR) and Cumulative Abnormal Returns (CAR) are computed across the event window.

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/SentimenTrade.git
cd SentimenTrade

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit dashboard
streamlit run dashboard/app.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Market data download |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `transformers` | Hugging Face FinBERT model |
| `torch` | PyTorch backend for inference |
| `streamlit` | Interactive dashboard |
| `scipy` | Statistical testing |
| `plotly` | Visualization |

---

## License
MIT License — free to use, modify, and distribute with attribution.
