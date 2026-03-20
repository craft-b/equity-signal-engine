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
from regimes import classify_vol_regimes, regime_performance
from stats import bootstrap_sharpe_ci, ttest_mean_return
from utils import (
    build_drawdown_chart,
    build_feature_importance_chart,
    build_performance_chart,
    build_regime_chart,
    build_walk_forward_chart,
    summarise_trades,
    summarise_walk_forward_folds,
)
from walk_forward import WalkForwardConfig, walk_forward_backtest

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
threshold_pct = st.sidebar.slider("Return Threshold (%)", 0.1, 3.0, 0.5, 0.1,
                                   help="Minimum forward return to label as 'up' for ML target.")
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

st.sidebar.subheader("Walk-Forward Validation")
run_wf = st.sidebar.checkbox(
    "Run Walk-Forward",
    value=False,
    help="Retrain on each expanding window and test OOS — the honest performance estimate.",
)
if run_wf:
    with st.sidebar.expander("Walk-Forward Settings"):
        wf_min_train = st.slider("Min Train Bars", 100, 500, 252,
                                  help="Minimum training history before first OOS fold.")
        wf_test_bars = st.slider("Test Bars per Fold", 21, 126, 63,
                                  help="OOS bars per fold. 63 ≈ 1 quarter.")
        wf_expanding = st.radio("Window Type", ["Expanding", "Rolling"],
                                 help="Expanding: train on all history. Rolling: fixed lookback.") == "Expanding"


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


def add_target(df: pd.DataFrame, holding_period: int, threshold_pct: float = 0.5) -> pd.DataFrame:
    out = df.copy()
    out["Future_Return"] = out["Close"].pct_change(holding_period).shift(-holding_period)
    out["Target"] = (out["Future_Return"] > threshold_pct / 100).astype(int)
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
df_features = df.copy()  # preserve pre-target df for walk-forward
df = add_target(df, holding_period, threshold_pct)

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

c9, c10, _, _ = st.columns(4)
sortino = metrics.get("sortino_ratio", 0)
calmar = metrics.get("calmar_ratio", float("nan"))
c9.metric("Sortino Ratio", f"{sortino:.2f}",
          help="Annualised return / downside volatility. Higher is better.")
c10.metric("Calmar Ratio", f"{calmar:.2f}" if not (isinstance(calmar, float) and np.isnan(calmar)) else "—",
           help="Annualised return / max drawdown. Higher is better.")

# Sanity warnings
if metrics.get("total_return", 0) > 200:
    st.warning("Returns > 200% — likely overfitting or data error.")
if metrics.get("num_trades", 0) < 5:
    st.warning("Fewer than 5 trades — results not statistically meaningful.")
if abs(metrics.get("max_drawdown", 0)) > 30:
    st.warning("Max drawdown > 30% — strategy carries significant tail risk.")

# Statistical significance
if portfolio_history and len(portfolio_history) >= 20:
    with st.expander("Statistical Significance", expanded=False):
        daily_rets = pd.Series(portfolio_history).pct_change().dropna().values

        with st.spinner("Bootstrapping Sharpe CI…"):
            ci_lo, ci_hi, _ = bootstrap_sharpe_ci(daily_rets, n_boot=1000)
        t_stat, p_value = ttest_mean_return(daily_rets)

        sc1, sc2 = st.columns(2)

        if np.isfinite(ci_lo) and np.isfinite(ci_hi):
            sc1.metric(
                "Sharpe 95% CI",
                f"[{ci_lo:.2f}, {ci_hi:.2f}]",
                help="Bootstrap 95% confidence interval for the annualised Sharpe ratio (1 000 resamples). "
                     "A CI that excludes zero suggests the Sharpe is likely genuine.",
            )
            if ci_lo > 0:
                sc1.success("CI excludes zero — Sharpe likely positive.")
            elif ci_hi < 0:
                sc1.error("CI entirely negative — strategy underperforms risk-free.")
            else:
                sc1.info("CI straddles zero — Sharpe is not reliably positive.")
        else:
            sc1.info("Insufficient data to compute Sharpe CI.")

        if np.isfinite(p_value):
            alpha = 0.05
            sig_label = "significant" if p_value < alpha else "not significant"
            sc2.metric(
                "Mean Return p-value",
                f"{p_value:.4f}",
                delta=f"t = {t_stat:.2f}",
                help="One-sample t-test: H₀ = mean daily return is zero. "
                     "p < 0.05 rejects the null and suggests returns are non-zero.",
            )
            if p_value < alpha:
                sc2.success(f"p = {p_value:.4f} — mean return is statistically {sig_label} (α = {alpha}).")
            else:
                sc2.warning(f"p = {p_value:.4f} — mean return is {sig_label} (α = {alpha}). Interpret results cautiously.")
        else:
            sc2.info("Insufficient data to compute t-test.")

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

# ---------------------------------------------------------------------------
# Volatility Regime Analysis
# ---------------------------------------------------------------------------

st.header("Volatility Regime Analysis")
st.caption(
    "Rolling realised volatility classifies each bar as Low / Medium / High. "
    "Performance broken down by regime reveals whether the strategy thrives "
    "in calm or turbulent markets."
)

vol_window = st.slider("Vol Window (bars)", 5, 60, 20, key="vol_window",
                       help="Rolling window for computing annualised volatility.")

df_regimes, regime_thresh = classify_vol_regimes(df, window=vol_window)

st.plotly_chart(
    build_regime_chart(df_regimes, regime_thresh),
    use_container_width=True,
)

if not trades_df.empty:
    reg_perf = regime_performance(trades_df, df_regimes)
    if not reg_perf.empty:
        r1, r2, r3 = st.columns(3)
        cols = [r1, r2, r3]
        regime_label_color = {"Low": "green", "Medium": "orange", "High": "red"}
        for col, (_, row) in zip(cols, reg_perf.iterrows()):
            label = row["Regime"]
            trades_n = int(row["Trades"])
            wr = row["Win Rate %"]
            pnl = row["Total P&L ($)"]
            col.metric(
                f"{label} Vol — {trades_n} trades",
                f"Win Rate: {wr:.1f}%" if not pd.isna(wr) else "No trades",
                delta=f"P&L: ${pnl:,.0f}" if trades_n > 0 else None,
                help=f"Trades entered during {label.lower()} volatility regime.",
            )

        with st.expander("Per-Regime Breakdown"):
            st.dataframe(
                reg_perf.style.format({
                    "Win Rate %": "{:.1f}",
                    "Avg Return %": "{:.2f}",
                    "Total P&L ($)": "${:,.2f}",
                    "Avg Days Held": "{:.1f}",
                }, na_rep="—"),
                use_container_width=True,
                hide_index=True,
            )
else:
    st.info("Run a backtest with at least one trade to see regime performance.")

# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

if run_wf:
    st.header("Walk-Forward Validation")
    st.caption(
        "Each fold trains on all prior data and tests on the next OOS window — "
        "mimicking live deployment. OOS Sharpe is the most trustworthy performance estimate."
    )

    wf_config = WalkForwardConfig(
        min_train_bars=wf_min_train,
        test_bars=wf_test_bars,
        expanding=wf_expanding,
    )
    wf_bt_config = BacktestConfig(
        max_position_pct=max_position_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_holding_days=holding_period,
        transaction_cost_pct=transaction_cost_pct,
        min_confidence=min_confidence,
        enable_shorts=enable_shorts,
    )
    wf_model_cfg = ModelConfig(model_type=model_type, calibrate=True) if model_type != "Technical Only" else None

    def make_train_fn(hp: int, tp: float, mcfg):
        def _train(df_fold: pd.DataFrame):
            if mcfg is None:
                return None, None
            df_fold_t = add_target(df_fold, hp, tp)
            result = train_model(df_fold_t, mcfg)
            if result is None:
                return None, None
            return result.model, result.feature_cols
        return _train

    with st.spinner(f"Running walk-forward ({wf_test_bars}-bar folds)…"):
        wf_folds, wf_summary = walk_forward_backtest(
            df_features, make_train_fn(holding_period, threshold_pct, wf_model_cfg),
            wf_bt_config, wf_config,
        )

    if "error" in wf_summary:
        st.error(f"Walk-forward error: {wf_summary['error']}")
    else:
        # Summary metrics
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("OOS Sharpe", f"{wf_summary['oos_sharpe']:.2f}",
                  help="Sharpe computed on concatenated OOS daily returns.")
        w2.metric("OOS Sortino", f"{wf_summary['oos_sortino']:.2f}")
        w3.metric("OOS Total Return", f"{wf_summary['oos_total_return']:.2f}%",
                  help="Compounded return across all OOS folds.")
        w4.metric("Profitable Folds", f"{wf_summary['pct_profitable_folds']:.0f}%",
                  help="% of folds with positive return.")

        w5, w6, w7, w8 = st.columns(4)
        w5.metric("Folds", wf_summary["n_folds"])
        w6.metric("Total OOS Trades", wf_summary["total_trades"])
        w7.metric(
            "Mean Fold Return",
            f"{wf_summary['mean_fold_return']:.2f}%",
            f"± {wf_summary['std_fold_return']:.2f}%",
        )
        w8.metric(
            "Mean Fold Sharpe",
            f"{wf_summary['mean_sharpe']:.2f}",
            f"± {wf_summary['std_sharpe']:.2f}",
        )

        # In-sample vs OOS comparison
        is_sharpe = metrics.get("sharpe_ratio", 0)
        oos_sharpe = wf_summary["oos_sharpe"]
        if is_sharpe > 0 and oos_sharpe < is_sharpe * 0.5:
            st.warning(
                f"OOS Sharpe ({oos_sharpe:.2f}) is less than half of in-sample "
                f"({is_sharpe:.2f}) — possible overfitting."
            )
        elif oos_sharpe >= is_sharpe * 0.8:
            st.success(
                f"OOS Sharpe ({oos_sharpe:.2f}) is close to in-sample "
                f"({is_sharpe:.2f}) — strategy generalises well."
            )

        # OOS equity curve
        st.plotly_chart(
            build_walk_forward_chart(wf_folds, bt_config.initial_capital),
            use_container_width=True,
        )

        # Per-fold breakdown
        with st.expander("Per-Fold Breakdown"):
            st.dataframe(summarise_walk_forward_folds(wf_folds), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("For educational and research purposes only. Not financial advice.")
