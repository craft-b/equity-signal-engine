"""
Signal Generator page.

Trains a calibrated ML classifier and displays:
- CV metrics: accuracy, AUC-ROC, Brier score, log-loss
- Probability calibration curve
- Price chart overlaid with signal markers + probability panel
- Signal quality validation (do signals predict forward returns?)
- Feature importance
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline import PipelineConfig, process_stock_data
from models import ModelConfig, TrainResult, generate_signals, train_model
from utils import build_feature_importance_chart

st.set_page_config(page_title="Signal Generator", layout="wide")
st.title("Probabilistic Signal Generator")
st.caption("Calibrated probability estimates for short-term directional price movements")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Configuration")

ASSET_META = {
    "SPY":  {"vol": 0.15, "desc": "S&P 500 — large-cap diversified"},
    "QQQ":  {"vol": 0.25, "desc": "Nasdaq 100 — tech-heavy"},
    "IWM":  {"vol": 0.22, "desc": "Russell 2000 — small-cap"},
    "AAPL": {"vol": 0.28, "desc": "Apple — mega-cap tech"},
    "MSFT": {"vol": 0.26, "desc": "Microsoft — mega-cap tech"},
    "TSLA": {"vol": 0.45, "desc": "Tesla — high-volatility growth"},
}

ticker = st.sidebar.selectbox("Asset", list(ASSET_META.keys()))
if ticker in ASSET_META:
    st.sidebar.caption(ASSET_META[ticker]["desc"])

period_map = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
period = period_map[st.sidebar.selectbox("Period", list(period_map.keys()), index=1)]

st.sidebar.subheader("Signal Parameters")
horizon_days = st.sidebar.slider(
    "Prediction Horizon (days)", 1, 20, 10,
    help="Days forward to predict. Longer = smoother but lagged."
)
threshold_pct = st.sidebar.slider(
    "Movement Threshold (%)", 0.1, 3.0, 0.5, 0.1,
    help="Minimum forward return to label as 'up'."
)

st.sidebar.subheader("Model")
model_type = st.sidebar.selectbox("Model Type", ["Logistic Regression", "Random Forest"])

with st.sidebar.expander("Advanced"):
    min_signal_prob = st.slider("Min Probability for Signal", 0.50, 0.90, 0.55, 0.01)
    enable_calibration = st.checkbox("Isotonic Calibration", value=True,
                                     help="Improves probability reliability.")
    enable_shorts = st.checkbox("Show Down Signals", value=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, period: str) -> dict:
    config = PipelineConfig(
        sma_periods=[10, 20, 50],
        volatility_windows=[10, 20],
        return_lags=[1, 2, 5],
        min_periods=100,
    )
    return process_stock_data(ticker, period, config)


def add_target(df: pd.DataFrame, horizon: int, threshold_pct: float) -> pd.DataFrame:
    out = df.copy()
    future_ret = out["Close"].pct_change(horizon).shift(-horizon)
    out["Target"] = (future_ret > threshold_pct / 100).astype(int)
    out["Future_Return"] = future_ret
    return out.dropna(subset=["Target"]).reset_index(drop=True)


def signal_quality_table(signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean forward return, win rate, and count for each signal class.
    NaN future_return rows (end of series) are excluded.
    """
    rows = []
    val_df = signal_df.dropna(subset=["Future_Return"])
    for sig_val, label in [(1, "Up"), (-1, "Down"), (0, "Neutral")]:
        sub = val_df[val_df["signal"] == sig_val]
        if len(sub) == 0:
            continue
        rows.append({
            "Signal": label,
            "Count": len(sub),
            "Avg Forward Return": f"{sub['Future_Return'].mean():.2%}",
            "Win Rate": f"{(sub['Future_Return'] > 0).mean():.1%}",
            "Median Return": f"{sub['Future_Return'].median():.2%}",
        })
    return pd.DataFrame(rows)


def build_signal_chart(signal_df: pd.DataFrame, min_prob: float, enable_shorts: bool) -> go.Figure:
    recent = signal_df.tail(min(252, len(signal_df)))

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.3, 0.2],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Price & Signals", "P(Up)", "Signal Strength"),
    )

    # Price
    fig.add_trace(
        go.Scatter(x=recent["Date"], y=recent["Close"], name="Price",
                   line=dict(color="#4C9BE8", width=1.5)),
        row=1, col=1,
    )

    # Signal markers
    ups = recent[recent["signal"] == 1]
    downs = recent[recent["signal"] == -1]
    if not ups.empty:
        fig.add_trace(
            go.Scatter(x=ups["Date"], y=ups["Close"], mode="markers", name="Up",
                       marker=dict(symbol="triangle-up", color="green", size=10)),
            row=1, col=1,
        )
    if enable_shorts and not downs.empty:
        fig.add_trace(
            go.Scatter(x=downs["Date"], y=downs["Close"], mode="markers", name="Down",
                       marker=dict(symbol="triangle-down", color="red", size=10)),
            row=1, col=1,
        )

    # Probability
    fig.add_trace(
        go.Scatter(x=recent["Date"], y=recent["prob_up"], name="P(Up)",
                   line=dict(color="steelblue", width=1.5), fill="tozeroy",
                   fillcolor="rgba(70,130,180,0.15)"),
        row=2, col=1,
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    fig.add_hline(y=min_prob, line_dash="dot", line_color="green", opacity=0.7, row=2, col=1)
    if enable_shorts:
        fig.add_hline(y=1.0 - min_prob, line_dash="dot", line_color="red", opacity=0.7, row=2, col=1)

    # Strength
    fig.add_trace(
        go.Bar(x=recent["Date"], y=recent["signal_strength"], name="Strength",
               marker_color="purple", opacity=0.6),
        row=3, col=1,
    )

    fig.update_layout(height=750, template="plotly_dark", showlegend=True,
                      margin=dict(l=40, r=20, t=40, b=20), hovermode="x unified")
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Probability", range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="Strength", row=3, col=1)
    return fig


def build_calibration_chart(cal_true: np.ndarray, cal_pred: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cal_pred, y=cal_true, mode="markers+lines",
        name="Model", marker=dict(size=10, color="steelblue"), line=dict(width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
        line=dict(dash="dash", color="red", width=1.5),
    ))
    fig.update_layout(
        title="Calibration Curve — closer to diagonal is better",
        xaxis_title="Predicted P(Up)",
        yaxis_title="Actual Frequency",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def build_return_distribution(signal_df: pd.DataFrame, horizon: int) -> go.Figure:
    val_df = signal_df.dropna(subset=["Future_Return"])
    fig = go.Figure()
    for sig_val, color, name in [(1, "green", "Up Signal"), (-1, "red", "Down Signal")]:
        sub = val_df[val_df["signal"] == sig_val]
        if len(sub) > 0:
            fig.add_trace(go.Box(
                y=sub["Future_Return"] * 100, name=name,
                marker_color=color, boxmean="sd",
            ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=f"Forward Return Distribution by Signal ({horizon}-day horizon)",
        yaxis_title="Forward Return (%)",
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if not st.button("Generate Signals", type="primary"):
    st.info("Configure parameters in the sidebar, then click **Generate Signals**.")
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

with st.expander("Data Provenance"):
    c1, c2 = st.columns(2)
    c1.write(f"**Ticker:** {ticker}")
    c1.write(f"**Period:** {period}")
    c1.write(f"**Rows:** {len(df):,}")
    c1.write(f"**Features:** {len(summary['features'])}")
    c2.write(f"**Price range:** ${summary['price_range'][0]:.2f} – ${summary['price_range'][1]:.2f}")
    c2.write(f"**Missing values:** {summary['missing_data']}")

# 2. Add target --------------------------------------------------------------
df = add_target(df, horizon_days, threshold_pct)
pos_rate = df["Target"].mean()
st.write(
    f"Target: forward return > {threshold_pct}% over {horizon_days} days · "
    f"positive rate **{pos_rate:.1%}**"
)

# 3. Train model -------------------------------------------------------------
with st.spinner(f"Training {model_type}…"):
    cfg = ModelConfig(
        model_type=model_type,
        calibrate=enable_calibration,
        min_train_samples=100,
    )
    result_model: TrainResult | None = train_model(df, cfg)

if result_model is None:
    st.error("Model training failed — insufficient data or class diversity.")
    st.stop()

# 4. Generate signals --------------------------------------------------------
signal_df = generate_signals(
    df, result_model.model, result_model.feature_cols,
    min_prob=min_signal_prob, enable_shorts=enable_shorts,
)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

st.header("Model Performance")

c1, c2, c3, c4 = st.columns(4)
acc_delta = (result_model.cv.accuracy - 0.5) * 100
c1.metric("CV Accuracy", f"{result_model.cv.accuracy:.3f}",
          f"{acc_delta:+.1f}% vs random")
c2.metric("AUC-ROC", f"{result_model.cv.auc_roc:.3f}",
          help="0.5 = random, 1.0 = perfect")
c3.metric("Brier Score", f"{result_model.cv.brier:.3f}",
          help="Lower is better. < 0.25 is good")
c4.metric("Log-Loss", f"{result_model.cv.log_loss:.3f}",
          help="Lower is better")

c5, c6, c7, c8 = st.columns(4)
c5.metric("CV Precision", f"{result_model.cv.precision:.3f}")
c6.metric("CV Recall", f"{result_model.cv.recall:.3f}")
c7.metric("CV F1", f"{result_model.cv.f1:.3f}")
c8.metric("Train time", f"{result_model.train_time_s:.1f}s")

# Signal stats
st.header("Signal Statistics")
up_n = (signal_df["signal"] == 1).sum()
down_n = (signal_df["signal"] == -1).sum()
flat_n = (signal_df["signal"] == 0).sum()
total_sig = up_n + down_n

s1, s2, s3 = st.columns(3)
s1.metric("Up Signals", int(up_n))
s2.metric("Down Signals", int(down_n))
s3.metric("No Signal", int(flat_n))

if total_sig == 0:
    st.warning("No directional signals generated — try lowering the Min Probability threshold.")
elif total_sig > 0:
    up_pct = up_n / total_sig * 100
    if up_pct > 70:
        st.warning(f"Bullish bias: {up_pct:.0f}% of signals are Up — possible training-period bias.")
    elif up_pct < 30:
        st.warning(f"Bearish bias: {100 - up_pct:.0f}% of signals are Down.")
    else:
        st.success(f"Balanced signals: {up_pct:.0f}% Up / {100 - up_pct:.0f}% Down")

# Current signal
st.header("Current Signal")
latest = signal_df.iloc[-1]
cs1, cs2, cs3 = st.columns(3)
with cs1:
    if latest["signal"] == 1:
        st.success("### UP")
    elif latest["signal"] == -1:
        st.error("### DOWN")
    else:
        st.info("### NO SIGNAL")
cs2.metric("P(Up)", f"{latest['prob_up']:.1%}")
cs2.metric("Signal Strength", f"{latest['signal_strength']:.3f}")
cs3.metric("Price", f"${latest['Close']:.2f}")
cs3.metric("Date", latest["Date"].strftime("%Y-%m-%d"))

# Charts
st.plotly_chart(build_signal_chart(signal_df, min_signal_prob, enable_shorts),
                use_container_width=True)

st.header("Signal Quality Validation")
st.caption("Do signals predict future returns? This is the most important validation.")
st.plotly_chart(build_return_distribution(signal_df, horizon_days), use_container_width=True)
quality_df = signal_quality_table(signal_df)
if not quality_df.empty:
    st.dataframe(quality_df, use_container_width=True, hide_index=True)

# Calibration
st.header("Probability Calibration")
if result_model.calibration_curve is not None:
    cal_true, cal_pred = result_model.calibration_curve
    st.plotly_chart(build_calibration_chart(cal_true, cal_pred), use_container_width=True)
    if enable_calibration:
        st.success("Isotonic calibration applied — probabilities are post-calibration.")
    else:
        st.info("Calibration disabled — raw model probabilities shown.")

# Feature importance
if result_model.feature_importance is not None:
    st.header("Feature Importance")
    st.plotly_chart(
        build_feature_importance_chart(result_model.feature_importance, top_n=15),
        use_container_width=True,
    )

# Detailed CV breakdown
with st.expander("Cross-Validation Details"):
    cv = result_model.cv
    cv_rows = [
        ("Accuracy", f"{cv.accuracy:.3f} ± {cv.accuracy_std:.3f}"),
        ("AUC-ROC",  f"{cv.auc_roc:.3f} ± {cv.auc_roc_std:.3f}"),
        ("Brier",    f"{cv.brier:.3f} ± {cv.brier_std:.3f}"),
        ("Log-Loss", f"{cv.log_loss:.3f}"),
        ("Precision", f"{cv.precision:.3f}"),
        ("Recall",   f"{cv.recall:.3f}"),
        ("F1",       f"{cv.f1:.3f}"),
        ("# Features", str(result_model.n_features)),
        ("# Samples",  f"{result_model.n_samples:,}"),
        ("Positive rate", f"{result_model.positive_rate:.1%}"),
    ]
    st.dataframe(
        pd.DataFrame(cv_rows, columns=["Metric", "Value"]),
        hide_index=True, use_container_width=True,
    )

st.markdown("---")
st.caption("For educational and research purposes only. Not financial advice.")
