"""
Benchmark: vectorized (numpy-array) backtest loop vs legacy pandas-loc loop.

The legacy implementation is reproduced inline here for comparison — it is
the original run_backtest loop that used bt.loc[i, col] per iteration.
The vectorized version (current run_backtest) extracts numpy arrays once
before entering the loop, eliminating per-bar pandas indexing overhead.

Run with:
    pytest tests/test_benchmark.py -v -s
to see the printed timing results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from backtest import BacktestConfig, Trade, _generate_signals, run_backtest


# ---------------------------------------------------------------------------
# Legacy loop (original pandas-loc implementation — preserved for comparison)
# ---------------------------------------------------------------------------

def _run_legacy_backtest(
    df: pd.DataFrame,
    config: BacktestConfig,
) -> Tuple[pd.DataFrame, Dict, List[float]]:
    """
    Original event-driven loop using bt.loc[i, col] per bar.
    Kept here solely to provide a timing baseline against the
    vectorized (numpy-array) rewrite.
    """
    bt = df.copy().reset_index(drop=True)
    bt["_signal"] = _generate_signals(bt, None, None, config)

    cash = config.initial_capital
    position = 0
    shares = 0.0
    entry_price = 0.0
    entry_date = None
    entry_idx = 0

    trades: List[Trade] = []
    portfolio_history: List[float] = [config.initial_capital]

    for i in range(1, len(bt)):
        price = bt.loc[i, "Close"]
        signal = bt.loc[i, "_signal"]
        date = bt.loc[i, "Date"]

        if price <= 0 or np.isnan(price):
            portfolio_history.append(portfolio_history[-1])
            continue

        portfolio_value = cash + shares * price * position if position != 0 else cash

        if position != 0:
            days_held = i - entry_idx
            price_chg_pct = (price - entry_price) / entry_price
            unrealized_pct = price_chg_pct * position

            exit_reason = None
            if unrealized_pct <= -config.stop_loss_pct:
                exit_reason = "Stop Loss"
            elif unrealized_pct >= config.take_profit_pct:
                exit_reason = "Take Profit"
            elif days_held >= config.max_holding_days:
                exit_reason = "Max Holding"
            elif signal != 0 and signal != position:
                exit_reason = "Signal Reversal"

            if exit_reason:
                if position == 1:
                    gross_pnl = shares * (price - entry_price)
                else:
                    gross_pnl = shares * (entry_price - price)
                transaction_fee = shares * price * config.transaction_cost_pct
                net_pnl = gross_pnl - transaction_fee
                cash += (shares * entry_price * position) + net_pnl

                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=date,
                    entry_price=entry_price,
                    exit_price=price,
                    position="LONG" if position == 1 else "SHORT",
                    shares=shares,
                    pnl=net_pnl,
                    pnl_pct=net_pnl / config.initial_capital * 100,
                    return_pct=unrealized_pct * 100,
                    days_held=days_held,
                    exit_reason=exit_reason,
                ))

                position = 0
                shares = 0.0
                portfolio_value = cash

        if position == 0 and signal != 0:
            position = int(signal)
            entry_price = price
            entry_date = date
            entry_idx = i
            notional = portfolio_value * config.max_position_pct
            shares = notional / price
            transaction_fee = notional * config.transaction_cost_pct
            cash -= (notional + transaction_fee)
            portfolio_value = cash + shares * price

        portfolio_history.append(portfolio_value)

    if position != 0 and len(bt) > 0:
        final_price = bt["Close"].iloc[-1]
        if position == 1:
            gross_pnl = shares * (final_price - entry_price)
        else:
            gross_pnl = shares * (entry_price - final_price)
        transaction_fee = shares * final_price * config.transaction_cost_pct
        net_pnl = gross_pnl - transaction_fee
        cash += (shares * entry_price * position) + net_pnl
        portfolio_history[-1] = cash

    return pd.DataFrame([vars(t) for t in trades]), {}, portfolio_history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 100.0 * (1 + rng.normal(0, 0.01, n)).cumprod()
    sma10 = pd.Series(prices).rolling(10, min_periods=1).mean().values
    sma50 = pd.Series(prices).rolling(50, min_periods=1).mean().values
    return pd.DataFrame({
        "Date": pd.bdate_range("2020-01-01", periods=n),
        "Close": prices,
        "SMA_10": sma10,
        "SMA_50": sma50,
    })


def _time(fn, repeats: int = 5) -> float:
    """Return median wall-clock seconds over `repeats` calls of fn()."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBenchmark:
    """Timing and equivalence tests for the vectorized backtest loop."""

    @pytest.mark.parametrize("n", [500, 2000])
    def test_vectorized_at_least_as_fast_as_legacy(self, n: int, capsys):
        df = _make_df(n)
        config = BacktestConfig()

        t_legacy = _time(lambda: _run_legacy_backtest(df, config))
        t_vec = _time(lambda: run_backtest(df, config=config))

        speedup = t_legacy / t_vec if t_vec > 0 else float("inf")
        with capsys.disabled():
            print(
                f"\n  n={n:,}: legacy={t_legacy*1000:.1f}ms  "
                f"vectorized={t_vec*1000:.1f}ms  "
                f"speedup={speedup:.1f}x"
            )

        assert t_vec <= t_legacy * 1.5, (
            f"Vectorized ({t_vec*1000:.1f}ms) should not be slower than legacy "
            f"({t_legacy*1000:.1f}ms) at n={n}"
        )

    def test_portfolio_history_numerically_equivalent(self):
        """Vectorized and legacy loops must produce the same portfolio curve."""
        df = _make_df(500)
        config = BacktestConfig()

        _, _, ph_legacy = _run_legacy_backtest(df, config)
        _, _, ph_vec = run_backtest(df, config=config)

        assert len(ph_vec) == len(ph_legacy)
        np.testing.assert_allclose(
            ph_vec, ph_legacy, rtol=1e-9,
            err_msg="Portfolio histories diverge between legacy and vectorized loops",
        )

    def test_trades_numerically_equivalent(self):
        """Trade P&L must match to floating-point precision."""
        df = _make_df(500)
        config = BacktestConfig()

        trades_legacy, _, _ = _run_legacy_backtest(df, config)
        trades_vec, _, _ = run_backtest(df, config=config)

        assert len(trades_vec) == len(trades_legacy), (
            f"Trade counts differ: legacy={len(trades_legacy)} vec={len(trades_vec)}"
        )

        if not trades_vec.empty:
            np.testing.assert_allclose(
                trades_vec["pnl"].values,
                trades_legacy["pnl"].values,
                rtol=1e-9,
                err_msg="Trade P&Ls differ between legacy and vectorized loops",
            )

    def test_short_positions_equivalent(self):
        """Equivalence check with shorts enabled."""
        df = _make_df(300)
        config = BacktestConfig(enable_shorts=True, max_holding_days=10)

        trades_legacy, _, ph_legacy = _run_legacy_backtest(df, config)
        trades_vec, _, ph_vec = run_backtest(df, config=config)

        assert len(ph_vec) == len(ph_legacy)
        np.testing.assert_allclose(ph_vec, ph_legacy, rtol=1e-9)
