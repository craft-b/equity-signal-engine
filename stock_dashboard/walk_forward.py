"""
Walk-forward backtesting.

Splits an already-processed DataFrame into sequential expanding (or rolling)
train/test windows, retrains the model on each fold, and runs the backtest
on the held-out OOS period.

This provides a statistically honest measure of out-of-sample strategy
performance: each test period is evaluated with a model that has never seen
that data, mimicking how the strategy would perform in live deployment.

Key design:
- The data pipeline runs once on the full dataset (technical indicators need
  full history to initialise correctly).  Walk-forward only splits the output.
- Target labels are added *inside* train_fn, per fold, so no future
  information leaks across the fold boundary.
- train_fn is a caller-supplied callable, keeping this module decoupled from
  ModelConfig / train_model internals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest import BacktestConfig, run_backtest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    min_train_bars: int = 252   # Minimum bars in training window before first fold
    test_bars: int = 63         # OOS bars per fold (~1 quarter of trading days)
    expanding: bool = True      # True = expanding window; False = fixed rolling window
    rolling_train_bars: int = 504  # Only used when expanding=False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    train_start: object
    train_end: object
    test_start: object
    test_end: object
    metrics: Dict
    portfolio_history: List[float]
    trades_df: pd.DataFrame
    n_train_bars: int
    model_trained: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def walk_forward_backtest(
    df: pd.DataFrame,
    train_fn: Callable[[pd.DataFrame], Tuple],
    bt_config: BacktestConfig,
    wf_config: Optional[WalkForwardConfig] = None,
) -> Tuple[List[FoldResult], Dict]:
    """
    Run walk-forward backtesting on a processed DataFrame.

    Args:
        df:        Processed DataFrame with Close, Date, and feature columns.
                   Must NOT include Target/Future_Return — train_fn adds those
                   per fold so no future data leaks into training.
        train_fn:  Callable(df_train) -> (model, feature_cols) or (None, None).
                   Responsible for adding target labels and fitting the model.
                   Return (None, None) to fall back to technical-only signals.
        bt_config: BacktestConfig for the per-fold simulation.
        wf_config: WalkForwardConfig controlling fold sizes and window type.

    Returns:
        folds:   List[FoldResult], one entry per OOS test window.
        summary: Dict of aggregated OOS metrics across all folds.
    """
    wf_config = wf_config or WalkForwardConfig()
    n = len(df)
    need = wf_config.min_train_bars + wf_config.test_bars

    if n < need:
        msg = f"Insufficient data for walk-forward: {n} bars, need at least {need}"
        logger.error(msg)
        return [], {"error": msg}

    # Fold boundaries: test window starts at min_train_bars, steps by test_bars
    test_starts = list(range(wf_config.min_train_bars, n - wf_config.test_bars + 1, wf_config.test_bars))

    folds: List[FoldResult] = []

    for fold_idx, test_start_i in enumerate(test_starts):
        test_end_i = min(test_start_i + wf_config.test_bars, n)

        if wf_config.expanding:
            train_start_i = 0
        else:
            train_start_i = max(0, test_start_i - wf_config.rolling_train_bars)

        df_train = df.iloc[train_start_i:test_start_i].reset_index(drop=True)
        df_test = df.iloc[test_start_i:test_end_i].reset_index(drop=True)

        if len(df_test) < 10:
            logger.debug("Fold %d: test window only %d bars — skipping", fold_idx, len(df_test))
            continue

        # Train model (train_fn is responsible for adding target labels)
        model, feature_cols = train_fn(df_train)

        # Run OOS backtest — falls back to SMA signals if model is None
        trades_df, metrics, portfolio_history = run_backtest(
            df_test, model, feature_cols, bt_config
        )

        folds.append(FoldResult(
            fold=fold_idx,
            train_start=df_train["Date"].iloc[0],
            train_end=df_train["Date"].iloc[-1],
            test_start=df_test["Date"].iloc[0],
            test_end=df_test["Date"].iloc[-1],
            metrics=metrics,
            portfolio_history=portfolio_history,
            trades_df=trades_df,
            n_train_bars=len(df_train),
            model_trained=model is not None,
        ))

        logger.info(
            "Fold %02d | train %s–%s (%d bars) | test %s–%s | "
            "return=%.1f%%  sharpe=%.2f  trades=%d",
            fold_idx,
            df_train["Date"].iloc[0].date(), df_train["Date"].iloc[-1].date(),
            len(df_train),
            df_test["Date"].iloc[0].date(), df_test["Date"].iloc[-1].date(),
            metrics.get("total_return", 0.0),
            metrics.get("sharpe_ratio", 0.0),
            metrics.get("num_trades", 0),
        )

    if not folds:
        return [], {"error": "No valid folds produced — try reducing min_train_bars or test_bars"}

    return folds, _aggregate(folds)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _aggregate(folds: List[FoldResult]) -> Dict:
    """Compute aggregate OOS statistics across all completed folds."""
    valid = [f for f in folds if "error" not in f.metrics]
    if not valid:
        return {"error": "All folds produced errors"}

    def _mean(key: str) -> float:
        return float(np.mean([f.metrics.get(key, 0.0) for f in valid]))

    def _std(key: str) -> float:
        return float(np.std([f.metrics.get(key, 0.0) for f in valid]))

    # Concatenate all OOS daily returns to compute aggregate Sharpe / Sortino
    oos_rets: List[float] = []
    for f in valid:
        ph = pd.Series(f.portfolio_history)
        oos_rets.extend(ph.pct_change().dropna().tolist())
    oos = pd.Series(oos_rets)

    oos_sharpe = float(oos.mean() / oos.std() * np.sqrt(252)) if oos.std() > 0 else 0.0
    downside = oos[oos < 0]
    oos_sortino = (
        float(oos.mean() / downside.std() * np.sqrt(252))
        if len(downside) > 1 and downside.std() > 0 else 0.0
    )

    # Compound per-fold returns for a total OOS return figure
    compound = float(np.prod([(1 + f.metrics.get("total_return", 0.0) / 100) for f in valid]) - 1) * 100

    return {
        "n_folds": len(valid),
        "total_trades": sum(f.metrics.get("num_trades", 0) for f in valid),
        "oos_total_return": compound,
        "oos_sharpe": oos_sharpe,
        "oos_sortino": oos_sortino,
        "mean_fold_return": _mean("total_return"),
        "std_fold_return": _std("total_return"),
        "mean_sharpe": _mean("sharpe_ratio"),
        "std_sharpe": _std("sharpe_ratio"),
        "mean_win_rate": _mean("win_rate"),
        "mean_max_drawdown": _mean("max_drawdown"),
        "pct_profitable_folds": float(
            sum(1 for f in valid if f.metrics.get("total_return", 0.0) > 0) / len(valid) * 100
        ),
    }
