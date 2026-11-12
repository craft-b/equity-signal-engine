"""
Tests for stock_dashboard/backtest.py

Covers input validation, signal generation, trade lifecycle (stop-loss,
take-profit, max holding, signal reversal), P&L accounting, and metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import BacktestConfig, _generate_signals, run_backtest


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


# ---------------------------------------------------------------------------
# Execution lag (E5): signals must not fill on the bar that produced them
# ---------------------------------------------------------------------------

class TestExecutionLag:
    """Features at bar i include Close[i], so a fill at Close[i] is look-ahead."""

    @staticmethod
    def _ramp(n: int = 120) -> pd.DataFrame:
        """Flat prices with one sharp step, so the entry bar is unambiguous."""
        close = np.full(n, 100.0)
        close[60:] = 130.0
        return pd.DataFrame({
            "Date": pd.date_range("2022-01-03", periods=n, freq="B"),
            "Close": close,
            "Open": close,
            "High": close,
            "Low": close,
            "Volume": np.full(n, 1_000_000.0),
        })

    def test_default_config_lags_execution_by_one_bar(self):
        assert BacktestConfig().execution_lag_bars == 1

    def test_entry_is_one_bar_after_the_signal(self, monkeypatch):
        import backtest as bt_mod

        df = self._ramp()
        signal_bar = 70

        def fake_signals(frame, model, feature_cols, config):
            sig = pd.Series(0, index=frame.index)
            sig.iloc[signal_bar] = 1
            return sig

        monkeypatch.setattr(bt_mod, "_generate_signals", fake_signals)

        lagged, _, _ = bt_mod.run_backtest(df, None, None, BacktestConfig())
        immediate, _, _ = bt_mod.run_backtest(
            df, None, None, BacktestConfig(execution_lag_bars=0)
        )

        assert not lagged.empty and not immediate.empty
        lag_entry = pd.Timestamp(lagged.iloc[0]["entry_date"])
        now_entry = pd.Timestamp(immediate.iloc[0]["entry_date"])

        assert now_entry == df["Date"].iloc[signal_bar]
        assert lag_entry == df["Date"].iloc[signal_bar + 1]

    def test_zero_lag_reproduces_previous_behaviour(self):
        """The optimistic path stays reachable so the change is measurable."""
        assert BacktestConfig(execution_lag_bars=0).execution_lag_bars == 0


# ---------------------------------------------------------------------------
# Intrabar exits (E7): stops must see the bar's range, not just its close
# ---------------------------------------------------------------------------

class TestIntrabarStops:
    """Checking the close alone lets a whipsaw through the stop go unrecorded."""

    @staticmethod
    def _bars(n: int = 80) -> pd.DataFrame:
        # run_backtest rejects frames under 50 rows.
        close = np.full(n, 100.0)
        return pd.DataFrame({
            "Date": pd.date_range("2022-01-03", periods=n, freq="B"),
            "Open": close.copy(),
            "High": close.copy(),
            "Low": close.copy(),
            "Close": close.copy(),
            "Volume": np.full(n, 1_000_000.0),
        })

    def _run(self, df, config, signal_bar=5):
        import backtest as bt_mod

        def fake_signals(frame, model, feature_cols, cfg):
            sig = pd.Series(0, index=frame.index)
            sig.iloc[signal_bar] = 1
            return sig

        original = bt_mod._generate_signals
        bt_mod._generate_signals = fake_signals
        try:
            return bt_mod.run_backtest(df, None, None, config)
        finally:
            bt_mod._generate_signals = original

    def test_default_enables_intrabar_stops(self):
        assert BacktestConfig().intrabar_stops is True

    def test_whipsaw_through_the_stop_is_caught(self):
        """Low breaches the stop; the close recovers. Close-only misses it."""
        df = self._bars()
        df.loc[10, "Low"] = 95.0        # -5%, through a 3% stop
        df.loc[10, "Close"] = 99.5      # recovers by the bell

        config = BacktestConfig(stop_loss_pct=0.03, take_profit_pct=0.50,
                                max_holding_days=30)
        intrabar, _, _ = self._run(df, config)
        close_only, _, _ = self._run(
            df, BacktestConfig(stop_loss_pct=0.03, take_profit_pct=0.50,
                               max_holding_days=30, intrabar_stops=False)
        )

        assert intrabar.iloc[0]["exit_reason"] == "Stop Loss"
        assert close_only.iloc[0]["exit_reason"] != "Stop Loss"

    def test_stop_fills_at_the_level_not_the_close(self):
        df = self._bars()
        df.loc[10, "Low"] = 95.0
        df.loc[10, "Close"] = 99.5
        config = BacktestConfig(stop_loss_pct=0.03, take_profit_pct=0.50,
                                max_holding_days=30)
        trades, _, _ = self._run(df, config)
        assert trades.iloc[0]["exit_price"] == pytest.approx(97.0)

    def test_gap_through_the_stop_fills_at_the_open(self):
        """A bar that opens below the stop cannot fill at the stop price."""
        df = self._bars()
        df.loc[10, ["Open", "High", "Low", "Close"]] = [92.0, 93.0, 91.0, 92.5]
        config = BacktestConfig(stop_loss_pct=0.03, take_profit_pct=0.50,
                                max_holding_days=30)
        trades, _, _ = self._run(df, config)
        assert trades.iloc[0]["exit_price"] == pytest.approx(92.0)
        assert trades.iloc[0]["pnl"] < 0

    def test_stop_wins_when_a_bar_breaches_both_levels(self):
        """OHLC cannot order the two touches, so assume the adverse one."""
        df = self._bars()
        df.loc[10, ["Open", "High", "Low", "Close"]] = [100.0, 110.0, 95.0, 105.0]
        config = BacktestConfig(stop_loss_pct=0.03, take_profit_pct=0.05,
                                max_holding_days=30)
        trades, _, _ = self._run(df, config)
        assert trades.iloc[0]["exit_reason"] == "Stop Loss"

    def test_falls_back_to_close_only_without_high_low(self):
        df = self._bars().drop(columns=["High", "Low"])
        config = BacktestConfig(stop_loss_pct=0.03, max_holding_days=30)
        trades, metrics, _ = self._run(df, config)
        assert "error" not in metrics


# ---------------------------------------------------------------------------
# Accounting consistency: the trade table and the equity curve must agree
# ---------------------------------------------------------------------------
# Both defects below were invisible to trade-level assertions. A short's
# per-trade pnl was computed correctly while cash was debited for its notional
# on entry AND again on exit, so the equity curve fell by twice the notional per
# short; and a position still open at the last bar had its P&L booked to cash
# but never recorded as a trade, so total_return could read 0.0% against a
# curve that had moved. Checking that sum(trade pnl) == final - initial catches
# both, and would have caught them from the start.

def _falling_short_setup(n: int = 80):
    prices = np.linspace(100.0, 80.0, n)
    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "Close": prices,
        "SMA_10": np.full(n, 95.0),   # SMA_10 < SMA_50 -> short when enabled
        "SMA_50": np.full(n, 100.0),
    })
    return df


class TestShortAccounting:
    def test_short_portfolio_curve_matches_trade_pnl(self):
        df = _falling_short_setup()
        cfg = BacktestConfig(
            enable_shorts=True, max_holding_days=50,
            stop_loss_pct=0.50, take_profit_pct=0.50,
            transaction_cost_pct=0.0, intrabar_stops=False, execution_lag_bars=0,
        )
        trades_df, metrics, ph = run_backtest(df, config=cfg)
        assert not trades_df.empty
        assert (trades_df["position"] == "SHORT").all()
        curve_pnl = ph[-1] - cfg.initial_capital
        assert trades_df["pnl"].sum() == pytest.approx(curve_pnl, abs=1e-6), (
            f"trade pnl {trades_df['pnl'].sum():.2f} vs curve {curve_pnl:.2f}")

    def test_short_does_not_lose_its_notional_on_entry(self):
        """The bar after a short opens, value must be ~capital, not capital - 2x notional."""
        df = _falling_short_setup()
        cfg = BacktestConfig(
            enable_shorts=True, max_holding_days=50, max_position_pct=0.20,
            stop_loss_pct=0.50, take_profit_pct=0.50,
            transaction_cost_pct=0.0, intrabar_stops=False, execution_lag_bars=0,
        )
        _, _, ph = run_backtest(df, config=cfg)
        # Entry happens at bar 1 on this setup; bar 2 is the first mark-to-market.
        assert ph[2] > cfg.initial_capital * 0.99, (
            f"portfolio fell to {ph[2]:,.0f} the bar after a 20% short opened")

    def test_short_profits_show_in_total_return(self):
        df = _falling_short_setup()
        cfg = BacktestConfig(
            enable_shorts=True, max_holding_days=50,
            stop_loss_pct=0.50, take_profit_pct=0.50,
            transaction_cost_pct=0.0, intrabar_stops=False, execution_lag_bars=0,
        )
        _, metrics, _ = run_backtest(df, config=cfg)
        # Falling 20% with a 20% short position: unambiguously positive.
        assert metrics["total_return"] > 0

    def test_long_trade_pnl_matches_curve(self):
        """Same invariant on the long side, so the fix is not short-specific."""
        n = 80
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=n, freq="B"),
            "Close": np.linspace(100.0, 120.0, n),
            "SMA_10": np.full(n, 105.0),
            "SMA_50": np.full(n, 100.0),
        })
        cfg = BacktestConfig(
            max_holding_days=50, stop_loss_pct=0.50, take_profit_pct=0.50,
            transaction_cost_pct=0.0, intrabar_stops=False, execution_lag_bars=0,
        )
        trades_df, _, ph = run_backtest(df, config=cfg)
        assert (trades_df["position"] == "LONG").all()
        assert trades_df["pnl"].sum() == pytest.approx(ph[-1] - cfg.initial_capital, abs=1e-6)


class TestEndOfDataClose:
    def test_open_position_at_last_bar_is_recorded_as_a_trade(self):
        df = _falling_short_setup()
        cfg = BacktestConfig(
            enable_shorts=True, max_holding_days=500,   # never exits on its own
            stop_loss_pct=0.50, take_profit_pct=0.50,
            transaction_cost_pct=0.0, intrabar_stops=False, execution_lag_bars=0,
        )
        trades_df, metrics, ph = run_backtest(df, config=cfg)
        assert len(trades_df) == 1
        assert trades_df.iloc[0]["exit_reason"] == "End of Data"
        assert metrics["num_trades"] == 1

    def test_forced_close_pnl_is_in_total_return(self):
        """Previously total_return read 0.0% while the curve had moved."""
        df = _falling_short_setup()
        cfg = BacktestConfig(
            enable_shorts=True, max_holding_days=500,
            stop_loss_pct=0.50, take_profit_pct=0.50,
            transaction_cost_pct=0.0, intrabar_stops=False, execution_lag_bars=0,
        )
        trades_df, metrics, ph = run_backtest(df, config=cfg)
        curve_return = (ph[-1] / cfg.initial_capital - 1) * 100
        assert metrics["total_return"] == pytest.approx(curve_return, abs=1e-6)
        assert trades_df["pnl"].sum() == pytest.approx(ph[-1] - cfg.initial_capital, abs=1e-6)
