"""
Event-driven backtest engine.

Simulates a realistic single-asset trading strategy:
- Entry on signal (ML probability or technical crossover)
- Exit on stop-loss, take-profit, max holding period, or signal reversal
- Fractional position sizing (fixed percentage of portfolio)
- Per-trade transaction cost (round-trip applied at entry and exit)
- Supports both long-only and long/short modes

All prices must be in dollar terms (not scaled) — the data pipeline
guarantees this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    max_position_pct: float = 0.20     # Fraction of portfolio per trade
    stop_loss_pct: float = 0.03        # 3 % adverse move → exit
    take_profit_pct: float = 0.08      # 8 % favourable move → exit
    max_holding_days: int = 5
    transaction_cost_pct: float = 0.001  # 0.1 % one-way (applied both legs)
    min_confidence: float = 0.55        # ML probability threshold for entry
    enable_shorts: bool = False


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_date: object
    exit_date: object
    entry_price: float
    exit_price: float
    position: str          # "LONG" | "SHORT"
    shares: float
    pnl: float
    pnl_pct: float         # P&L as % of initial capital
    return_pct: float      # Trade return % (price-based)
    days_held: int
    exit_reason: str


def _generate_signals(
    df: pd.DataFrame,
    model,
    feature_cols: Optional[List[str]],
    config: BacktestConfig,
) -> pd.Series:
    """Return a Series of signals aligned to df's index: 1=long, -1=short, 0=flat."""
    signals = pd.Series(0, index=df.index, dtype=int)

    if model is not None and feature_cols is not None:
        try:
            X = df[feature_cols].ffill().fillna(0).replace([np.inf, -np.inf], 0)
            probs = model.predict_proba(X)[:, 1]
            signals[probs >= config.min_confidence] = 1
            if config.enable_shorts:
                signals[probs <= (1.0 - config.min_confidence)] = -1
        except Exception as exc:
            logger.warning("ML signal generation failed (%s) — falling back to zero", exc)
    else:
        # Simple SMA crossover
        if "SMA_10" in df.columns and "SMA_50" in df.columns:
            signals[df["SMA_10"] > df["SMA_50"]] = 1
            if config.enable_shorts:
                signals[df["SMA_10"] < df["SMA_50"]] = -1
        elif "SMA_5" in df.columns and "SMA_20" in df.columns:
            signals[df["SMA_5"] > df["SMA_20"]] = 1
            if config.enable_shorts:
                signals[df["SMA_5"] < df["SMA_20"]] = -1

    return signals


def run_backtest(
    df: pd.DataFrame,
    model=None,
    feature_cols: Optional[List[str]] = None,
    config: Optional[BacktestConfig] = None,
) -> Tuple[pd.DataFrame, Dict, List[float]]:
    """
    Run the backtest simulation.

    Args:
        df:            Processed DataFrame with at least 'Close' and 'Date' columns.
        model:         Fitted sklearn estimator (optional).
        feature_cols:  List of feature column names used by the model (optional).
        config:        BacktestConfig instance.

    Returns:
        trades_df:        DataFrame of completed trades (empty if none).
        metrics:          Dict of performance statistics.
        portfolio_history: List of portfolio values (one per bar).
    """
    config = config or BacktestConfig()

    # ---------- Input validation ----------
    if df is None or len(df) < 50:
        return pd.DataFrame(), {"error": "Insufficient data (< 50 rows)"}, []

    required = {"Close", "Date"}
    missing = required - set(df.columns)
    if missing:
        return pd.DataFrame(), {"error": f"Missing columns: {missing}"}, []

    bt = df.copy().reset_index(drop=True)

    if (bt["Close"] <= 0).any() or bt["Close"].isna().any():
        return pd.DataFrame(), {"error": "Invalid prices in backtest input"}, []

    # ---------- Signal generation ----------
    bt["_signal"] = _generate_signals(bt, model, feature_cols, config)

    # ---------- Simulation state ----------
    cash = config.initial_capital
    position = 0        # 1 = long, -1 = short, 0 = flat
    shares = 0.0
    entry_price = 0.0
    entry_date = None
    entry_idx = 0

    trades: List[Trade] = []
    portfolio_history: List[float] = [config.initial_capital]

    # ---------- Main loop ----------
    for i in range(1, len(bt)):
        price = bt.loc[i, "Close"]
        signal = bt.loc[i, "_signal"]
        date = bt.loc[i, "Date"]

        if price <= 0 or np.isnan(price):
            portfolio_history.append(portfolio_history[-1])
            continue

        # Mark-to-market
        portfolio_value = cash + shares * price * position if position != 0 else cash

        # ---------- Exit logic ----------
        if position != 0:
            days_held = i - entry_idx
            price_chg_pct = (price - entry_price) / entry_price
            unrealized_pct = price_chg_pct * position  # positive = profit

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
                # Realise P&L
                if position == 1:
                    gross_pnl = shares * (price - entry_price)
                else:
                    gross_pnl = shares * (entry_price - price)

                transaction_fee = shares * price * config.transaction_cost_pct
                net_pnl = gross_pnl - transaction_fee
                cash += (shares * entry_price * position) + net_pnl  # return notional + profit

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

        # ---------- Entry logic ----------
        if position == 0 and signal != 0:
            position = int(signal)
            entry_price = price
            entry_date = date
            entry_idx = i

            notional = portfolio_value * config.max_position_pct
            shares = notional / price
            transaction_fee = notional * config.transaction_cost_pct
            cash -= (notional + transaction_fee)  # deduct cost basis + fee
            portfolio_value = cash + shares * price

        portfolio_history.append(portfolio_value)

    # ---------- Close any open position at end ----------
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

    final_value = portfolio_history[-1]

    # ---------- Metrics ----------
    if not trades:
        bh_return = (bt["Close"].iloc[-1] / bt["Close"].iloc[0] - 1) * 100
        return pd.DataFrame(), {
            "total_return": 0.0,
            "buy_hold_return": bh_return,
            "excess_return": -bh_return,
            "num_trades": 0,
        }, portfolio_history

    trades_df = pd.DataFrame([vars(t) for t in trades])

    total_return = (final_value - config.initial_capital) / config.initial_capital * 100
    bh_return = (bt["Close"].iloc[-1] / bt["Close"].iloc[0] - 1) * 100

    winners = trades_df[trades_df["pnl"] > 0]
    losers = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(winners) / len(trades_df) * 100

    port_series = pd.Series(portfolio_history)
    daily_returns = port_series.pct_change().dropna()
    sharpe = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std() > 0 else 0.0
    )

    cum_max = port_series.cummax()
    max_drawdown = ((port_series - cum_max) / cum_max).min() * 100

    metrics = {
        "total_return": total_return,
        "buy_hold_return": bh_return,
        "excess_return": total_return - bh_return,
        "num_trades": len(trades_df),
        "win_rate": win_rate,
        "avg_win_pct": winners["pnl_pct"].mean() if len(winners) else 0.0,
        "avg_loss_pct": losers["pnl_pct"].mean() if len(losers) else 0.0,
        "profit_factor": (
            winners["pnl"].sum() / abs(losers["pnl"].sum())
            if len(losers) and losers["pnl"].sum() != 0 else np.nan
        ),
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "final_value": final_value,
    }

    return trades_df, metrics, portfolio_history
