"""
Volatility regime detection using rolling realised volatility.

Each bar is labelled Low / Medium / High based on where its rolling
annualised volatility falls relative to the full-period distribution.
Regime boundaries are set at configurable percentile thresholds so they
adapt to each asset and time period rather than using fixed vol levels.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

REGIME_ORDER = ["Low", "Medium", "High"]
REGIME_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}


def classify_vol_regimes(
    df: pd.DataFrame,
    window: int = 20,
    low_pct: float = 33.0,
    high_pct: float = 67.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Add Rolling_Vol and Vol_Regime columns to df.

    Regimes are determined by where each bar's rolling annualised volatility
    falls relative to the full-period distribution:

        Low    — vol ≤ low_pct  percentile
        High   — vol ≥ high_pct percentile
        Medium — everything in between

    Args:
        df:       DataFrame with a 'Close' column.
        window:   Rolling window for realised vol (bars).
        low_pct:  Percentile below which regime is 'Low'.
        high_pct: Percentile above which regime is 'High'.

    Returns:
        (df_out, thresholds) where thresholds contains the computed vol
        boundaries and the input parameters used to derive them.
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column")

    out = df.copy()
    rolling_vol = out["Close"].pct_change().rolling(window).std() * np.sqrt(252)
    out["Rolling_Vol"] = rolling_vol

    finite_vol = rolling_vol.dropna()
    lo_thresh = float(finite_vol.quantile(low_pct / 100.0))
    hi_thresh = float(finite_vol.quantile(high_pct / 100.0))

    regime = pd.Series("Medium", index=out.index, dtype=object)
    regime[rolling_vol <= lo_thresh] = "Low"
    regime[rolling_vol >= hi_thresh] = "High"
    out["Vol_Regime"] = regime

    thresholds = {
        "low": lo_thresh,
        "high": hi_thresh,
        "low_pct": low_pct,
        "high_pct": high_pct,
        "window": window,
    }
    return out, thresholds


def regime_performance(
    trades_df: pd.DataFrame,
    df_with_regimes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-regime trade performance statistics.

    Each trade is assigned the regime that was active at its entry date.

    Args:
        trades_df:        Completed trades DataFrame from run_backtest.
        df_with_regimes:  DataFrame with 'Date' and 'Vol_Regime' columns.

    Returns:
        DataFrame with one row per regime containing:
        Regime, Trades, Win Rate %, Avg Return %, Total P&L ($), Avg Days Held.
        Regimes with no trades still appear (with NaN metrics) so the caller
        always gets a fixed-schema result.
    """
    if trades_df.empty or "entry_date" not in trades_df.columns:
        return pd.DataFrame()

    regime_map = df_with_regimes.set_index("Date")["Vol_Regime"].to_dict()
    trades = trades_df.copy()
    trades["Regime"] = trades["entry_date"].map(regime_map).fillna("Medium")

    rows = []
    for regime in REGIME_ORDER:
        subset = trades[trades["Regime"] == regime]
        if subset.empty:
            rows.append({
                "Regime": regime,
                "Trades": 0,
                "Win Rate %": float("nan"),
                "Avg Return %": float("nan"),
                "Total P&L ($)": 0.0,
                "Avg Days Held": float("nan"),
            })
            continue

        winners = subset[subset["pnl"] > 0]
        rows.append({
            "Regime": regime,
            "Trades": len(subset),
            "Win Rate %": round(len(winners) / len(subset) * 100, 1),
            "Avg Return %": round(float(subset["return_pct"].mean()), 2),
            "Total P&L ($)": round(float(subset["pnl"].sum()), 2),
            "Avg Days Held": round(float(subset["days_held"].mean()), 1),
        })

    return pd.DataFrame(rows)
