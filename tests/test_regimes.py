"""
Tests for stock_dashboard/regimes.py.

Covers classify_vol_regimes and regime_performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimes import REGIME_ORDER, classify_vol_regimes, regime_performance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 252, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 100.0 * (1 + rng.normal(0, 0.01, n)).cumprod()
    return pd.DataFrame({
        "Date": pd.bdate_range("2022-01-03", periods=n),
        "Close": prices,
    })


def _make_trades(entry_dates, returns=None, pnls=None, days=None) -> pd.DataFrame:
    n = len(entry_dates)
    if returns is None:
        returns = [1.0] * n
    if pnls is None:
        pnls = [100.0] * n
    if days is None:
        days = [3] * n
    return pd.DataFrame({
        "entry_date": entry_dates,
        "exit_date": entry_dates,
        "return_pct": returns,
        "pnl": pnls,
        "days_held": days,
        "position": ["LONG"] * n,
    })


# ---------------------------------------------------------------------------
# classify_vol_regimes
# ---------------------------------------------------------------------------

class TestClassifyVolRegimes:
    def test_adds_rolling_vol_column(self):
        df, _ = classify_vol_regimes(_make_price_df())
        assert "Rolling_Vol" in df.columns

    def test_adds_vol_regime_column(self):
        df, _ = classify_vol_regimes(_make_price_df())
        assert "Vol_Regime" in df.columns

    def test_only_valid_regime_labels(self):
        df, _ = classify_vol_regimes(_make_price_df())
        assert set(df["Vol_Regime"].unique()).issubset(set(REGIME_ORDER))

    def test_all_three_regimes_present(self):
        df, _ = classify_vol_regimes(_make_price_df(n=500))
        assert set(df["Vol_Regime"].unique()) == {"Low", "Medium", "High"}

    def test_thresholds_keys(self):
        _, thresh = classify_vol_regimes(_make_price_df())
        assert {"low", "high", "low_pct", "high_pct", "window"}.issubset(thresh)

    def test_low_threshold_less_than_high(self):
        _, thresh = classify_vol_regimes(_make_price_df())
        assert thresh["low"] < thresh["high"]

    def test_low_regime_has_low_vol(self):
        df, thresh = classify_vol_regimes(_make_price_df(n=500))
        low_vols = df.loc[df["Vol_Regime"] == "Low", "Rolling_Vol"].dropna()
        assert (low_vols <= thresh["low"] + 1e-9).all()

    def test_high_regime_has_high_vol(self):
        df, thresh = classify_vol_regimes(_make_price_df(n=500))
        high_vols = df.loc[df["Vol_Regime"] == "High", "Rolling_Vol"].dropna()
        assert (high_vols >= thresh["high"] - 1e-9).all()

    def test_respects_custom_window(self):
        df, thresh = classify_vol_regimes(_make_price_df(), window=10)
        assert thresh["window"] == 10

    def test_respects_custom_percentiles(self):
        _, thresh = classify_vol_regimes(_make_price_df(), low_pct=20.0, high_pct=80.0)
        assert thresh["low_pct"] == 20.0
        assert thresh["high_pct"] == 80.0

    def test_missing_close_raises(self):
        df = pd.DataFrame({"Date": pd.bdate_range("2022-01-03", periods=10), "Open": range(10)})
        with pytest.raises(ValueError, match="Close"):
            classify_vol_regimes(df)

    def test_preserves_original_columns(self):
        df_in = _make_price_df()
        df_in["Extra"] = 1
        df_out, _ = classify_vol_regimes(df_in)
        assert "Extra" in df_out.columns

    def test_does_not_mutate_input(self):
        df_in = _make_price_df()
        cols_before = set(df_in.columns)
        classify_vol_regimes(df_in)
        assert set(df_in.columns) == cols_before


# ---------------------------------------------------------------------------
# regime_performance
# ---------------------------------------------------------------------------

class TestRegimePerformance:
    def test_returns_dataframe(self):
        df, _ = classify_vol_regimes(_make_price_df(n=500))
        dates = df["Date"].iloc[20:25].tolist()
        trades = _make_trades(dates)
        result = regime_performance(trades, df)
        assert isinstance(result, pd.DataFrame)

    def test_all_regimes_in_output(self):
        df, _ = classify_vol_regimes(_make_price_df(n=500))
        dates = df["Date"].iloc[20:25].tolist()
        trades = _make_trades(dates)
        result = regime_performance(trades, df)
        assert set(result["Regime"]) == {"Low", "Medium", "High"}

    def test_trade_counts_sum_to_total(self):
        df, _ = classify_vol_regimes(_make_price_df(n=500))
        dates = df["Date"].iloc[20:70].tolist()
        trades = _make_trades(dates)
        result = regime_performance(trades, df)
        assert result["Trades"].sum() == len(dates)

    def test_empty_trades_returns_empty(self):
        df, _ = classify_vol_regimes(_make_price_df())
        result = regime_performance(pd.DataFrame(), df)
        assert result.empty

    def test_win_rate_all_winners(self):
        df, _ = classify_vol_regimes(_make_price_df(n=500))
        low_regime_dates = df.loc[df["Vol_Regime"] == "Low", "Date"].iloc[:5].tolist()
        if not low_regime_dates:
            pytest.skip("No Low-regime bars in sample")
        trades = _make_trades(low_regime_dates, pnls=[200.0] * len(low_regime_dates))
        result = regime_performance(trades, df)
        low_row = result[result["Regime"] == "Low"]
        assert float(low_row["Win Rate %"].iloc[0]) == 100.0

    def test_win_rate_all_losers(self):
        df, _ = classify_vol_regimes(_make_price_df(n=500))
        high_regime_dates = df.loc[df["Vol_Regime"] == "High", "Date"].iloc[:5].tolist()
        if not high_regime_dates:
            pytest.skip("No High-regime bars in sample")
        trades = _make_trades(high_regime_dates, pnls=[-50.0] * len(high_regime_dates))
        result = regime_performance(trades, df)
        high_row = result[result["Regime"] == "High"]
        assert float(high_row["Win Rate %"].iloc[0]) == 0.0

    def test_expected_columns_present(self):
        df, _ = classify_vol_regimes(_make_price_df(n=500))
        dates = df["Date"].iloc[20:25].tolist()
        result = regime_performance(_make_trades(dates), df)
        expected = {"Regime", "Trades", "Win Rate %", "Avg Return %", "Total P&L ($)", "Avg Days Held"}
        assert expected.issubset(set(result.columns))

    def test_missing_entry_date_returns_empty(self):
        df, _ = classify_vol_regimes(_make_price_df())
        trades = pd.DataFrame({"pnl": [100.0]})
        result = regime_performance(trades, df)
        assert result.empty
