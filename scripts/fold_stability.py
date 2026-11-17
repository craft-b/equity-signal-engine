"""Measure how much the out-of-sample estimate depends on fold placement.

Where the walk-forward boundaries fall is a free parameter. If the result
moves materially when it changes, the result describes the placement rather
than the strategy. Reproduces the stability table in the README.

Usage:
    python scripts/fold_stability.py            # SPY 10y
    python scripts/fold_stability.py QQQ 5y
    python scripts/fold_stability.py --live     # download instead of the pinned snapshot
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stock_dashboard"))

for _name in ("walk_forward", "data_pipeline", "models"):
    logging.getLogger(_name).setLevel(logging.ERROR)

from backtest import BacktestConfig  # noqa: E402
from data_pipeline import (  # noqa: E402
    PipelineConfig, create_pipeline, find_snapshot, process_stock_data,
)
from labels import add_target  # noqa: E402
from models import ModelConfig, train_model  # noqa: E402
from stats import ttest_mean_return  # noqa: E402
from walk_forward import WalkForwardConfig, walk_forward_backtest  # noqa: E402

HOLDING = 5
THRESHOLD = 0.5
TRADING_DAYS = 252

#: (min_train_bars, test_bars). The 250/252 pair is the point: two bars.
GRID = [(250, 60), (252, 63), (252, 60), (250, 63), (252, 126), (300, 63)]

PIPELINE_CONFIG = PipelineConfig(
    sma_periods=[5, 10, 20, 50],
    volatility_windows=[10, 20],
    return_lags=[1, 2, 5],
    min_periods=100,
)


def main(*argv: str) -> int:
    flags = {a for a in argv if a.startswith("--")}
    if flags - {"--live"}:
        print(f"unknown flag(s): {', '.join(sorted(flags - {'--live'}))}")
        return 2
    positional = [a for a in argv if not a.startswith("--")]
    ticker = positional[0] if positional else "SPY"
    period = positional[1] if len(positional) > 1 else "10y"

    # The pinned snapshot by default, so the committed CSV reproduces; --live
    # downloads from Yahoo, whose adjusted prices differ a little every time.
    loaded = process_stock_data(
        ticker, period, PIPELINE_CONFIG,
        snapshot=find_snapshot(ticker, period, live="--live" in flags),
    )
    if not loaded["success"]:
        print(f"data fetch failed: {loaded['error']}")
        return 1
    processed, raw = loaded["data"], loaded["raw"]
    print(f"data: {loaded['source']}")

    model_config = ModelConfig(model_type="Random Forest", calibrate=True,
                               label_horizon=HOLDING)

    def train_fn(df_train: pd.DataFrame):
        result = train_model(add_target(df_train, HOLDING, THRESHOLD), model_config)
        return (result.model, result.feature_cols) if result else (None, None)

    print(f"{ticker} {period} · {len(processed)} bars · holding {HOLDING}d\n")
    print(f"{'min_train':>10}{'test_bars':>11}{'folds':>7}{'trades':>8}"
          f"{'OOS sharpe':>12}{'p':>8}")
    print("-" * 56)

    rows = []
    for min_train, test_bars in GRID:
        folds, _ = walk_forward_backtest(
            processed, train_fn, BacktestConfig(max_holding_days=HOLDING),
            WalkForwardConfig(min_train_bars=min_train, test_bars=test_bars,
                              expanding=True, embargo_bars=HOLDING),
            raw_df=raw,
            pipeline_factory=lambda: create_pipeline(PIPELINE_CONFIG),
        )
        if not folds:
            print(f"{min_train:>10}{test_bars:>11}   no folds")
            continue

        returns = np.concatenate([
            pd.Series(f.portfolio_history).pct_change().dropna().to_numpy()
            for f in folds
        ])
        sharpe = (
            returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)
            if returns.std() > 0 else 0.0
        )
        _, p_value = ttest_mean_return(returns)
        trades = sum(len(f.trades_df) for f in folds)
        rows.append({"min_train": min_train, "test_bars": test_bars,
                     "folds": len(folds), "trades": trades,
                     "sharpe": sharpe, "p_value": p_value})
        print(f"{min_train:>10}{test_bars:>11}{len(folds):>7}{trades:>8}"
              f"{sharpe:>12.2f}{p_value:>8.3f}")

    frame = pd.DataFrame(rows)
    out = ROOT / "docs" / f"fold_stability_{ticker.lower()}_{period}.csv"
    out.parent.mkdir(exist_ok=True)
    frame.to_csv(out, index=False)

    spread = frame["sharpe"].max() - frame["sharpe"].min()
    print("-" * 56)
    print(f"OOS Sharpe spans {spread:.2f} across {len(frame)} fold layouts "
          f"({frame['sharpe'].min():.2f} to {frame['sharpe'].max():.2f})")
    print(f"significant at p<0.05: {int((frame['p_value'] < 0.05).sum())} "
          f"of {len(frame)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
