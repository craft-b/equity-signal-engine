"""
Trading System Analysis — main entry point.

Streamlit multi-page app. This file acts as the landing page / home screen.
Individual analyses live under pages/:
    1_Signal_Generator.py  — probabilistic signal generation + calibration
    2_Strategy_Backtest.py — event-driven backtest + performance metrics
"""

import streamlit as st

st.set_page_config(
    page_title="Trading System Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Trading System Analysis")
st.caption("ML-powered equity signal generation and strategy backtesting")

st.markdown("""
---

### Navigation

Use the sidebar to switch between modules:

| Page | Description |
|------|-------------|
| **Signal Generator** | Probabilistic direction prediction with calibration, AUC-ROC, Brier score, and signal quality validation |
| **Strategy Backtest** | Event-driven backtest engine with position sizing, stop-loss/take-profit, and performance attribution |

---

### Architecture

```
data_pipeline.py   sklearn Pipeline: fetch → indicators → outlier cap → scale
models.py          TimeSeriesCV training + isotonic calibration + feature importance
backtest.py        Event-driven simulation: entries, exits, P&L, Sharpe, drawdown
utils.py           Plotly chart builders (shared across pages)
```

---
""")

st.info("Select a page from the sidebar to begin.")
