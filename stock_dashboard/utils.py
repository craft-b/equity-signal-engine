"""
Shared display utilities for the Streamlit dashboard.

Functions here are pure presentational helpers — they receive data and
return Plotly figures or Pandas DataFrames, with no side-effects.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_performance_chart(
    features_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    portfolio_history: List[float],
    initial_capital: float = 100_000.0,
) -> go.Figure:
    """
    Two-panel chart: price + trade markers (top) and portfolio vs buy-hold (bottom).
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Price & Trade Signals", "Portfolio Value"],
        row_heights=[0.6, 0.4],
        vertical_spacing=0.1,
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=features_df["Date"],
            y=features_df["Close"],
            name="Price",
            line=dict(color="#4C9BE8", width=1.5),
        ),
        row=1, col=1,
    )

    # Trade markers
    if not trades_df.empty:
        longs = trades_df[trades_df["position"] == "LONG"]
        shorts = trades_df[trades_df["position"] == "SHORT"]

        if not longs.empty:
            fig.add_trace(
                go.Scatter(
                    x=longs["entry_date"],
                    y=longs["entry_price"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", color="green", size=10),
                    name="Long Entry",
                ),
                row=1, col=1,
            )
        if not shorts.empty:
            fig.add_trace(
                go.Scatter(
                    x=shorts["entry_date"],
                    y=shorts["entry_price"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", color="red", size=10),
                    name="Short Entry",
                ),
                row=1, col=1,
            )

    # Portfolio equity curve
    port_dates = features_df["Date"].iloc[: len(portfolio_history)]
    fig.add_trace(
        go.Scatter(
            x=port_dates,
            y=portfolio_history,
            name="Strategy",
            line=dict(color="orange", width=2),
        ),
        row=2, col=1,
    )

    # Buy-and-hold benchmark
    bh_values = initial_capital * (
        features_df["Close"] / features_df["Close"].iloc[0]
    )
    fig.add_trace(
        go.Scatter(
            x=features_df["Date"],
            y=bh_values,
            name="Buy & Hold",
            line=dict(color="gray", dash="dash", width=1.5),
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=700,
        showlegend=True,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio ($)", row=2, col=1)

    return fig


def build_drawdown_chart(portfolio_history: List[float], dates: pd.Series) -> go.Figure:
    """Underwater equity (drawdown) chart."""
    port = pd.Series(portfolio_history)
    cum_max = port.cummax()
    drawdown = (port - cum_max) / cum_max * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates.iloc[: len(drawdown)],
            y=drawdown,
            fill="tozeroy",
            fillcolor="rgba(255,50,50,0.3)",
            line=dict(color="red", width=1),
            name="Drawdown %",
        )
    )
    fig.update_layout(
        title="Drawdown",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        height=250,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def build_feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of top-N feature importances."""
    top = importance_df.head(top_n).sort_values("importance")
    fig = go.Figure(
        go.Bar(
            x=top["importance"],
            y=top["feature"],
            orientation="h",
            marker_color="#4C9BE8",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance",
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=20, t=40, b=30),
    )
    return fig


def summarise_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready trades summary (most recent first)."""
    if trades_df.empty:
        return pd.DataFrame()

    display = trades_df[[
        "entry_date", "exit_date", "position", "entry_price",
        "exit_price", "return_pct", "pnl", "days_held", "exit_reason",
    ]].copy()

    display["entry_price"] = display["entry_price"].map("${:.2f}".format)
    display["exit_price"] = display["exit_price"].map("${:.2f}".format)
    display["return_pct"] = display["return_pct"].map("{:.2f}%".format)
    display["pnl"] = display["pnl"].map("${:,.2f}".format)

    return display.sort_values("exit_date", ascending=False).reset_index(drop=True)
