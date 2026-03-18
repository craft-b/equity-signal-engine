# ML Asset Prediction

A production-style ML trading system demonstrating end-to-end ML engineering in finance — from raw market data to a live interactive dashboard.

## Overview

This project builds a supervised learning pipeline for short-term equity direction prediction, backed by a realistic event-driven backtest engine. The goal is to showcase ML engineering patterns applicable to systematic trading infrastructure.

**Capabilities:**
- Sklearn-compatible data pipeline with configurable technical indicator computation, outlier capping, missing-data handling, and feature selection
- Binary classifier (Random Forest or Logistic Regression) trained with proper time-series cross-validation (no look-ahead)
- Event-driven backtest engine with position sizing, stop-loss/take-profit, transaction costs, and long/short support
- Streamlit dashboard for interactive strategy exploration

## Project Structure

```
ml-asset-prediction/
├── stock_dashboard/
│   ├── app.py             # Streamlit dashboard (entry point)
│   ├── data_pipeline.py   # Data fetching + sklearn Pipeline
│   ├── models.py          # Model training + evaluation
│   ├── backtest.py        # Event-driven backtest engine
│   └── utils.py           # Plotly chart builders
├── notebooks/
│   └── jane_trading.ipynb # Exploratory analysis
├── data_cache/            # Cached raw data (gitignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows bash
# or
source .venv/bin/activate        # macOS/Linux

pip install -r requirements.txt
```

## Running the Dashboard

```bash
cd stock_dashboard
streamlit run app.py
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Data | Yahoo Finance (`yfinance`) |
| Feature engineering | `pandas`, `numpy`, custom sklearn transformers |
| ML models | `scikit-learn` (Random Forest, Logistic Regression) |
| Backtesting | Custom event-driven engine |
| Dashboard | `streamlit`, `plotly` |

## Key Design Decisions

- **No look-ahead bias**: target variable is a forward-shifted return; pipeline steps are fit only on training data
- **Price columns never scaled**: Close/Open/High/Low remain in dollar terms for backtest accuracy
- **Time-series CV**: `TimeSeriesSplit` throughout — no random shuffling of temporal data
- **Depth-constrained Random Forest**: `max_depth=5`, `min_samples_leaf=25` to resist overfitting on noisy financial data

## Disclaimer

For educational and research purposes only. Not financial advice.
