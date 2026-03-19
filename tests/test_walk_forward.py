"""
Tests for stock_dashboard/walk_forward.py

Covers fold generation, date integrity (no train/test leakage), aggregate
metrics, edge cases (insufficient data, all-failing folds, rolling window).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from backtest import BacktestConfig
from walk_forward import FoldResult, WalkForwardConfig, walk_forward_backtest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_processed_df(n: int = 400) -> pd.DataFrame:
    """Minimal processed DataFrame with Close, Date, SMA_10, SMA_50."""
    rng = np.random.default_rng(7)
    prices = 100.0 * (1 + rng.normal(0, 0.01, n)).cumprod()
    return pd.DataFrame({
        "Date": pd.bdate_range("2020-01-01", periods=n),
        "Close": prices,
        "SMA_10": pd.Series(prices).rolling(10, min_periods=1).mean().values,
        "SMA_50": pd.Series(prices).rolling(50, min_periods=1).mean().values,
    })


def technical_train_fn(df_train: pd.DataFrame) -> Tuple:
    """train_fn that always returns (None, None) — uses SMA fallback."""
    return None, None


def _make_ml_train_fn(holding_period: int = 5):
    """train_fn that adds a target and trains a real model."""
    def _train(df_train: pd.DataFrame):
        from models import ModelConfig, train_model
        df = df_train.copy()
        future_ret = df["Close"].pct_change(holding_period).shift(-holding_period)
        df["Target"] = (future_ret > 0.005).astype(int)
        df = df.dropna(subset=["Target"]).reset_index(drop=True)
        cfg = ModelConfig(
            model_type="Random Forest",
            min_train_samples=50,
            n_cv_splits=2,
            min_feature_count=2,  # synthetic df only has SMA_10 / SMA_50
        )
        result = train_model(df, cfg)
        if result is None:
            return None, None
        return result.model, result.feature_cols
    return _train


BT_CONFIG = BacktestConfig(
    max_position_pct=0.20,
    stop_loss_pct=0.05,
    take_profit_pct=0.10,
    max_holding_days=5,
    transaction_cost_pct=0.001,
)


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------

class TestFoldGeneration:

    def test_produces_correct_number_of_folds(self):
        df = make_processed_df(n=400)
        wf_cfg = WalkForwardConfig(min_train_bars=200, test_bars=50)
        folds, summary = walk_forward_backtest(df, technical_train_fn, BT_CONFIG, wf_cfg)
        # 400 bars, first fold at 200, step 50 → folds at 200, 250, 300, 350 = 4
        assert len(folds) == 4

    def test_returns_error_on_insufficient_data(self):
        df = make_processed_df(n=50)
        wf_cfg = WalkForwardConfig(min_train_bars=200, test_bars=63)
        folds, summary = walk_forward_backtest(df, technical_train_fn, BT_CONFIG, wf_cfg)
        assert folds == []
        assert "error" in summary

    def test_fold_results_are_fold_result_instances(self):
        df = make_processed_df(n=400)
        wf_cfg = WalkForwardConfig(min_train_bars=252, test_bars=63)
        folds, _ = walk_forward_backtest(df, technical_train_fn, BT_CONFIG, wf_cfg)
        assert len(folds) > 0
        assert all(isinstance(f, FoldResult) for f in folds)


# ---------------------------------------------------------------------------
# Date integrity — no look-ahead
# ---------------------------------------------------------------------------

class TestDateIntegrity:

    def test_train_end_strictly_before_test_start(self):
        df = make_processed_df(n=400)
        wf_cfg = WalkForwardConfig(min_train_bars=200, test_bars=50)
        folds, _ = walk_forward_backtest(df, technical_train_fn, BT_CONFIG, wf_cfg)
        for f in folds:
            assert f.train_end < f.test_start, (
                f"Fold {f.fold}: train_end {f.train_end} >= test_start {f.test_start}"
            )

    def test_test_windows_do_not_overlap(self):
        df = make_processed_df(n=400)
        wf_cfg = WalkForwardConfig(min_train_bars=200, test_bars=50)
        folds, _ = walk_forward_backtest(df, technical_train_fn, BT_CONFIG, wf_cfg)
        for i in range(len(folds) - 1):
            assert folds[i].test_end <= folds[i + 1].test_start, (
                f"Folds {i} and {i+1} test windows overlap"
            )

    def test_expanding_window_train_grows_monotonically(self):
        df = make_processed_df(n=400)
        wf_cfg = WalkForwardConfig(min_train_bars=200, test_bars=50, expanding=True)
        folds, _ = walk_forward_backtest(df, technical_train_fn, BT_CONFIG, wf_cfg)
        train_sizes = [f.n_train_bars for f in folds]
        assert train_sizes == sorted(train_sizes)

    def test_rolling_window_train_size_stays_fixed(self):
        df = make_processed_df(n=600)
        roll = 200
        wf_cfg = WalkForwardConfig(min_train_bars=200, test_bars=50,
                                    expanding=False, rolling_train_bars=roll)
        folds, _ = walk_forward_backtest(df, technical_train_fn, BT_CONFIG, wf_cfg)
        # All folds (except possibly the first) should have exactly roll bars
        for f in folds[1:]:
            assert f.n_train_bars == roll


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

class TestAggregateSummary:

    def test_summary_contains_expected_keys(self):
        df = make_processed_df(n=400)
        wf_cfg = WalkForwardConfig(min_train_bars=200, test_bars=50)
        _, summary = walk_forward_backtest(df, technical_train_fn, BT_CONFIG, wf_cfg)
        for key in ("n_folds", "oos_sharpe", "oos_sortino", "oos_total_return",
                    "mean_fold_return", "std_fold_return", "mean_sharpe",
                    "pct_profitable_folds", "total_trades"):
            assert key in summary, f"Missing key: {key}"

    def test_n_folds_matches_fold_list_length(self):
        df = make_processed_df(n=400)
        wf_cfg = WalkForwardConfig(min_train_bars=200, test_bars=50)
        folds, summary = walk_forward_backtest(df, technical_train_fn, BT_CONFIG, wf_cfg)
        assert summary["n_folds"] == len(folds)

    def test_pct_profitable_folds_bounded(self):
        df = make_processed_df(n=400)
        wf_cfg = WalkForwardConfig(min_train_bars=200, test_bars=50)
        _, summary = walk_forward_backtest(df, technical_train_fn, BT_CONFIG, wf_cfg)
        assert 0.0 <= summary["pct_profitable_folds"] <= 100.0

    def test_oos_sharpe_is_finite(self):
        df = make_processed_df(n=400)
        wf_cfg = WalkForwardConfig(min_train_bars=200, test_bars=50)
        _, summary = walk_forward_backtest(df, technical_train_fn, BT_CONFIG, wf_cfg)
        assert np.isfinite(summary["oos_sharpe"])


# ---------------------------------------------------------------------------
# ML model integration
# ---------------------------------------------------------------------------

class TestMLIntegration:

    def test_ml_train_fn_marks_model_trained(self):
        df = make_processed_df(n=500)
        wf_cfg = WalkForwardConfig(min_train_bars=300, test_bars=63)
        folds, _ = walk_forward_backtest(df, _make_ml_train_fn(), BT_CONFIG, wf_cfg)
        # At least some folds should successfully train a model
        assert any(f.model_trained for f in folds)

    def test_fallback_to_technical_when_train_fn_returns_none(self):
        df = make_processed_df(n=400)
        wf_cfg = WalkForwardConfig(min_train_bars=200, test_bars=50)
        folds, summary = walk_forward_backtest(df, technical_train_fn, BT_CONFIG, wf_cfg)
        assert "error" not in summary
        assert all(not f.model_trained for f in folds)
