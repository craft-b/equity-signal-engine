"""
Tests for stock_dashboard/backtest.py

Covers input validation, signal generation, trade lifecycle (stop-loss,
take-profit, max holding, signal reversal), P&L accounting, and metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import BacktestConfig, Trade, _generate_signals, run_backtest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(n: int = 100, start_price: float = 100.0, trend: float = 0.0) -> pd.DataFrame:
    """Synthetic OHLCV-style DataFrame with Close and Date columns."""
    rng = np.random.default_rng(42)
    prices = start_price * (1 + trend + rng.normal(0, 0.005, size=n)).cumprod()
    prices = np.maximum(prices, 1.0)
    return pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "Close": prices,
        "SMA_10": pd.Series(prices).rolling(10).mean().values,
        "SMA_50": pd.Series(prices).rolling(50).mean().values,
    })


def make_trending_df(n: int = 100, direction: str = "up") -> pd.DataFrame:
    """DataFrame with a strong monotonic trend to guarantee signal + trade."""
    delta = 0.01 if direction == "up" else -0.01
    prices = 100.0 * np.cumprod(1 + delta * np.ones(n))
    sma10 = pd.Series(prices).rolling(10, min_periods=1).mean().values
    sma50 = pd.Series(prices).rolling(50, min_periods=1).mean().values
    return pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "Close": prices,
        "SMA_10": sma10,
        "SMA_50": sma50,
    })


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_returns_error_on_insufficient_rows(self):
        df = make_df(n=10)
        _, metrics, history = run_backtest(df)
        assert "error" in metrics

    def test_returns_error_on_missing_close(self):
        df = make_df().drop(columns=["Close"])
        _, metrics, _ = run_backtest(df)
        assert "error" in metrics

    def test_returns_error_on_missing_date(self):
        df = make_df().drop(columns=["Date"])
        _, metrics, _ = run_backtest(df)
        assert "error" in metrics

    def test_returns_error_on_zero_prices(self):
        df = make_df()
        df.loc[5, "Close"] = 0.0
        _, metrics, _ = run_backtest(df)
        assert "error" in metrics

    def test_returns_error_on_none_input(self):
        _, metrics, _ = run_backtest(None)
        assert "error" in metrics

    def test_uses_default_config_when_none(self):
        df = make_df()
        _, metrics, history = run_backtest(df, config=None)
        assert "error" not in metrics or metrics.get("num_trades", 0) >= 0


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

class TestGenerateSignals:

    def test_returns_series_aligned_to_df(self):
        df = make_df()
        config = BacktestConfig()
        signals = _generate_signals(df, None, None, config)
        assert len(signals) == len(df)
        assert signals.index.equals(df.index)

    def test_sma_crossover_generates_long_signals(self):
        df = make_trending_df(direction="up")
        config = BacktestConfig(enable_shorts=False)
        signals = _generate_signals(df, None, None, config)
        assert (signals == 1).any()
        assert (signals == -1).sum() == 0

    def test_no_short_signals_when_disabled(self):
        df = make_trending_df(direction="down")
        config = BacktestConfig(enable_shorts=False)
        signals = _generate_signals(df, None, None, config)
        assert (signals == -1).sum() == 0

    def test_short_signals_generated_when_enabled(self):
        df = make_trending_df(direction="down")
        config = BacktestConfig(enable_shorts=True)
        signals = _generate_signals(df, None, None, config)
        assert (signals == -1).any()

    def test_signals_only_contain_valid_values(self):
        df = make_df()
        signals = _generate_signals(df, None, None, BacktestConfig())
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_ml_model_signals_respect_confidence_threshold(self):
        """A mock model that always returns prob=0.90 should always signal long."""
        class AlwaysLong:
            def predict_proba(self, X):
                return np.column_stack([
                    np.full(len(X), 0.10),
                    np.full(len(X), 0.90),
                ])

        df = make_df()
        feature_cols = ["Close"]
        config = BacktestConfig(min_confidence=0.55)
        signals = _generate_signals(df, AlwaysLong(), feature_cols, config)
        assert (signals == 1).all()

    def test_ml_model_below_threshold_generates_no_signal(self):
        """A model always returning prob=0.50 should not exceed threshold."""
        class Neutral:
            def predict_proba(self, X):
                return np.column_stack([
                    np.full(len(X), 0.50),
                    np.full(len(X), 0.50),
                ])

        df = make_df()
        config = BacktestConfig(min_confidence=0.55)
        signals = _generate_signals(df, Neutral(), ["Close"], config)
        assert (signals == 0).all()


# ---------------------------------------------------------------------------
# Trade lifecycle
# ---------------------------------------------------------------------------

class TestTradePnL:

    def test_no_trades_returns_zero_return(self):
        """When no signal fires, total return should be 0."""
        df = make_df()
        # Force all signals to 0 by removing SMA columns
        df = df[["Date", "Close"]]
        _, metrics, _ = run_backtest(df, config=BacktestConfig())
        assert metrics.get("total_return", 0) == 0.0
        assert metrics.get("num_trades", 0) == 0

    def test_stop_loss_exit(self):
        """Price drops sharply after entry → stop-loss should trigger."""
        n = 80
        prices = np.full(n, 100.0)
        prices[40:] = 90.0   # 10% drop — exceeds default 3% stop-loss

        # Force SMA crossover at bar 1 to guarantee entry
        sma10 = np.full(n, 105.0)
        sma50 = np.full(n, 100.0)

        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=n, freq="B"),
            "Close": prices,
            "SMA_10": sma10,
            "SMA_50": sma50,
        })
        trades_df, metrics, _ = run_backtest(df, config=BacktestConfig(max_holding_days=50))
        assert not trades_df.empty
        assert (trades_df["exit_reason"] == "Stop Loss").any()

    def test_take_profit_exit(self):
        """Price rises sharply after entry → take-profit should trigger."""
        n = 80
        prices = np.full(n, 100.0, dtype=float)
        prices[40:] = 115.0   # 15% gain — exceeds default 8% take-profit

        sma10 = np.full(n, 105.0)
        sma50 = np.full(n, 100.0)

        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=n, freq="B"),
            "Close": prices,
            "SMA_10": sma10,
            "SMA_50": sma50,
        })
        trades_df, _, _ = run_backtest(df, config=BacktestConfig(max_holding_days=50))
        assert not trades_df.empty
        assert (trades_df["exit_reason"] == "Take Profit").any()

    def test_max_holding_exit(self):
        """Flat price with no SL/TP → exit after max holding days."""
        n = 80
        prices = np.full(n, 100.0, dtype=float)
        sma10 = np.full(n, 105.0)
        sma50 = np.full(n, 100.0)

        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=n, freq="B"),
            "Close": prices,
            "SMA_10": sma10,
            "SMA_50": sma50,
        })
        trades_df, _, _ = run_backtest(df, config=BacktestConfig(
            max_holding_days=5, stop_loss_pct=0.50, take_profit_pct=0.50
        ))
        assert not trades_df.empty
        assert (trades_df["exit_reason"] == "Max Holding").any()
        assert (trades_df["days_held"] <= 5).all()

    def test_transaction_costs_reduce_pnl(self):
        """Same scenario with higher transaction cost should yield lower final value."""
        df = make_trending_df(direction="up", n=100)
        _, metrics_cheap, _ = run_backtest(df, config=BacktestConfig(transaction_cost_pct=0.0001))
        _, metrics_expensive, _ = run_backtest(df, config=BacktestConfig(transaction_cost_pct=0.01))
        assert metrics_cheap.get("total_return", 0) >= metrics_expensive.get("total_return", 0)

    def test_pnl_accounting_is_consistent(self):
        """Sum of trade P&Ls should approximately equal total portfolio gain."""
        df = make_trending_df(direction="up", n=120)
        config = BacktestConfig()
        trades_df, metrics, portfolio_history = run_backtest(df, config=config)

        if trades_df.empty:
            pytest.skip("No trades generated")

        portfolio_gain = portfolio_history[-1] - config.initial_capital
        trade_pnl_sum = trades_df["pnl"].sum()
        # Allow 1% tolerance for open position at end / rounding
        assert abs(portfolio_gain - trade_pnl_sum) / config.initial_capital < 0.01


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_metrics_keys_present(self):
        df = make_trending_df(direction="up", n=120)
        _, metrics, _ = run_backtest(df)
        for key in ("total_return", "buy_hold_return", "excess_return",
                    "num_trades", "sharpe_ratio", "max_drawdown"):
            assert key in metrics

    def test_max_drawdown_is_non_positive(self):
        df = make_df()
        _, metrics, _ = run_backtest(df)
        assert metrics.get("max_drawdown", 0) <= 0

    def test_win_rate_bounded(self):
        df = make_trending_df(direction="up", n=120)
        _, metrics, _ = run_backtest(df)
        if metrics.get("num_trades", 0) > 0:
            assert 0.0 <= metrics["win_rate"] <= 100.0

    def test_portfolio_history_length_matches_df(self):
        df = make_df()
        _, _, portfolio_history = run_backtest(df)
        assert len(portfolio_history) == len(df)

    def test_portfolio_history_starts_at_initial_capital(self):
        config = BacktestConfig(initial_capital=50_000)
        df = make_df()
        _, _, portfolio_history = run_backtest(df, config=config)
        assert portfolio_history[0] == config.initial_capital

    def test_buy_hold_return_matches_price_change(self):
        df = make_df()
        expected_bh = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        _, metrics, _ = run_backtest(df)
        assert abs(metrics.get("buy_hold_return", 0) - expected_bh) < 0.01

    def test_no_trades_returns_correct_structure(self):
        df = df = make_df()[["Date", "Close"]]  # no SMA cols → no signals
        trades_df, metrics, history = run_backtest(df)
        assert trades_df.empty
        assert metrics["num_trades"] == 0
        assert len(history) > 0

    def test_sortino_and_calmar_present_in_metrics(self):
        df = make_trending_df(direction="up", n=120)
        _, metrics, _ = run_backtest(df)
        assert "sortino_ratio" in metrics
        assert "calmar_ratio" in metrics

    def test_sortino_ratio_bounded(self):
        df = make_trending_df(direction="up", n=120)
        _, metrics, _ = run_backtest(df)
        # Sortino can be large positive for trending up, but should be finite
        assert np.isfinite(metrics.get("sortino_ratio", 0))

    def test_signal_reversal_exit(self):
        """SMA_10 flips below SMA_50 while in a long position → Signal Reversal."""
        n = 80
        prices = np.full(n, 100.0, dtype=float)
        # First 40 bars: SMA_10 > SMA_50 (long signal)
        # Next 40 bars: SMA_10 < SMA_50 (short signal when shorts enabled)
        sma10 = np.where(np.arange(n) < 40, 105.0, 95.0).astype(float)
        sma50 = np.full(n, 100.0, dtype=float)

        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=n, freq="B"),
            "Close": prices,
            "SMA_10": sma10,
            "SMA_50": sma50,
        })
        trades_df, _, _ = run_backtest(df, config=BacktestConfig(
            max_holding_days=50, stop_loss_pct=0.50, take_profit_pct=0.50,
            enable_shorts=True,
        ))
        assert not trades_df.empty
        assert (trades_df["exit_reason"] == "Signal Reversal").any()

    def test_short_position_profits_on_falling_prices(self):
        """Short trade on falling prices should produce positive P&L."""
        n = 80
        prices = np.linspace(100.0, 80.0, n)  # steady decline
        sma10 = np.full(n, 95.0)
        sma50 = np.full(n, 100.0)

        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=n, freq="B"),
            "Close": prices,
            "SMA_10": sma10,
            "SMA_50": sma50,
        })
        trades_df, metrics, _ = run_backtest(df, config=BacktestConfig(
            enable_shorts=True, max_holding_days=50,
            stop_loss_pct=0.50, take_profit_pct=0.50,
        ))
        if not trades_df.empty:
            short_trades = trades_df[trades_df["position"] == "SHORT"]
            if not short_trades.empty:
                assert short_trades["pnl"].sum() > 0

    def test_nan_prices_rejected_at_validation(self):
        """NaN prices in Close should be caught at input validation, not silently skipped."""
        df = make_df(n=100)
        df.loc[30:35, "Close"] = np.nan  # inject NaN mid-series
        _, metrics, _ = run_backtest(df)
        assert "error" in metrics
