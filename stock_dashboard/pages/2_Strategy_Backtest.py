"""
Strategy Backtest page.

Runs the event-driven backtest engine and displays:
- Performance metrics: total return, Sharpe, max drawdown, profit factor
- Equity curve vs buy-and-hold
- Drawdown chart
- Trade log
- Optional ML model + feature importance
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest import BacktestConfig, run_backtest
from data_pipeline import PipelineConfig, process_stock_data
from models import ModelConfig, TrainResult, train_model
from utils import (
    build_drawdown_chart,
    build_feature_importance_chart,
    build_performance_chart,
    summarise_trades,
)

st.set_page_config(page_title="Strategy Backtest", layout="wide")
st.title("Strategy Backtest")
st.caption("Event-driven backtest with position sizing, stop-loss/take-profit, and transaction costs")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Configuration")

ticker = st.sidebar.selectbox("Asset", ["SPY", "QQQ", "IWM", "AAPL", "MSFT"], index=0)
period_map = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}
period = period_map[st.sidebar.selectbox("Period", list(period_map.keys()), index=1)]

st.sidebar.subheader("Strategy")
holding_period = st.sidebar.slider("Max Holding (days)", 1, 20, 5)
transaction_cost_pct = st.sidebar.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.01) / 100

st.sidebar.subheader("Risk Management")
max_position_pct = st.sidebar.slider("Max Position Size (%)", 10, 50, 20) / 100
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 1, 10, 3) / 100
take_profit_pct = st.sidebar.slider("Take Profit (%)", 2, 20, 8) / 100

st.sidebar.subheader("Model")
model_type = st.sidebar.selectbox(
    "Signal Source", ["Random Forest", "Logistic Regression", "Technical Only"]
)

with st.sidebar.expander("Advanced"):
    min_confidence = st.slider("Min ML Confidence", 0.50, 0.80, 0.55, 0.05)
    enable_shorts = st.checkbox("Enable Short Positions", value=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, period: str) -> dict:
    config = PipelineConfig(
        sma_periods=[5, 10, 20, 50],
        volatility_windows=[10, 20],
        return_lags=[1, 2, 5],
        min_periods=100,
    )
    return process_stock_data(ticker, period, config)


def add_target(df: pd.DataFrame, holding_period: int) -> pd.DataFrame:
    out = df.copy()
    out["Future_Return"] = out["Close"].pct_change(holding_period).shift(-holding_period)
    out["Target"] = (out["Future_Return"] > 0.01).astype(int)
    return out.dropna(subset=["Target"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if not st.button("Run Backtest", type="primary"):
    st.info("Configure parameters in the sidebar, then click **Run Backtest**.")
    st.stop()

# 1. Load data ---------------------------------------------------------------
with st.spinner("Fetching market data…"):
    result = load_data(ticker, period)

if not result["success"]:
    st.error(f"Data pipeline error: {result.get('error')}")
    st.stop()

df = result["data"]
summary = result["summary"]
st.success(
    f"Loaded {len(df):,} rows · "
    f"{summary['date_range'][0].date()} → {summary['date_range'][1].date()}"
)

# 2. Add target --------------------------------------------------------------
df = add_target(df, holding_period)

# 3. Train model (optional) --------------------------------------------------
train_result: TrainResult | None = None

if model_type != "Technical Only":
    with st.spinner(f"Training {model_type}…"):
        cfg = ModelConfig(model_type=model_type, calibrate=True)
        train_result = train_model(df, cfg)

    if train_result:
        st.success(
            f"Model trained — CV accuracy {train_result.cv.accuracy:.3f} · "
            f"AUC-ROC {train_result.cv.auc_roc:.3f} · "
            f"Brier {train_result.cv.brier:.3f}"
        )
    else:
        st.warning("Model training failed — using technical signals.")

# 4. Run backtest ------------------------------------------------------------
with st.spinner("Running backtest…"):
    bt_config = BacktestConfig(
        max_position_pct=max_position_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_holding_days=holding_period,
        transaction_cost_pct=transaction_cost_pct,
        min_confidence=min_confidence,
        enable_shorts=enable_shorts,
    )
    model_obj = train_result.model if train_result else None
    feature_cols = train_result.feature_cols if train_result else None
    trades_df, metrics, portfolio_history = run_backtest(df, model_obj, feature_cols, bt_config)

if "error" in metrics:
    st.error(f"Backtest error: {metrics['error']}")
    st.stop()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

st.header("Performance Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Total Return", f"{metrics.get('total_return', 0):.2f}%",
    delta=f"{metrics.get('excess_return', 0):.2f}% vs B&H",
)
c2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
c3.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
c4.metric("# Trades", metrics.get("num_trades", 0))

c5, c6, c7, c8 = st.columns(4)
c5.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
c6.metric("Avg Win", f"{metrics.get('avg_win_pct', 0):.2f}%")
c7.metric("Avg Loss", f"{metrics.get('avg_loss_pct', 0):.2f}%")
pf = metrics.get("profit_factor", float("nan"))
c8.metric("Profit Factor", f"{pf:.2f}" if not np.isnan(pf) else "—")

# Sanity warnings
if metrics.get("total_return", 0) > 200:
    st.warning("Returns > 200% — likely overfitting or data error.")
if metrics.get("num_trades", 0) < 5:
    st.warning("Fewer than 5 trades — results not statistically meaningful.")
if abs(metrics.get("max_drawdown", 0)) > 30:
    st.warning("Max drawdown > 30% — strategy carries significant tail risk.")

# Charts
if portfolio_history:
    st.plotly_chart(
        build_performance_chart(df, trades_df, portfolio_history, bt_config.initial_capital),
        use_container_width=True,
    )
    st.plotly_chart(
        build_drawdown_chart(portfolio_history, df["Date"]),
        use_container_width=True,
    )

# Feature importance
if train_result and train_result.feature_importance is not None:
    with st.expander("Feature Importance"):
        st.plotly_chart(
            build_feature_importance_chart(train_result.feature_importance, top_n=15),
            use_container_width=True,
        )

# Trade log
if not trades_df.empty:
    with st.expander(f"Trade Log ({len(trades_df)} trades)"):
        st.dataframe(summarise_trades(trades_df), use_container_width=True)

st.markdown("---")
st.caption("For educational and research purposes only. Not financial advice.")
