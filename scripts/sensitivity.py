"""Sweep the strategy's tuning knobs and report the whole OOS surface.

The obvious question about any single reported backtest is "how many
configurations did you try before this one?". Reporting the grid answers it
directly: if every cell is near zero, no amount of knob-turning was going to
produce an edge, and the single reported number was not cherry-picked.

Usage:
    python scripts/sensitivity.py             # SPY 10y
    python scripts/sensitivity.py QQQ 5y
"""

from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stock_dashboard"))

import logging  # noqa: E402

from backtest import BacktestConfig  # noqa: E402
from data_pipeline import PipelineConfig, create_pipeline, process_stock_data  # noqa: E402
from labels import add_target  # noqa: E402
from models import ModelConfig, train_model  # noqa: E402
from stats import ttest_mean_return  # noqa: E402
from walk_forward import WalkForwardConfig, walk_forward_backtest  # noqa: E402

logging.getLogger("walk_forward").setLevel(logging.ERROR)
logging.getLogger("data_pipeline").setLevel(logging.ERROR)
logging.getLogger("models").setLevel(logging.ERROR)

TRADING_DAYS = 252

HOLDING_PERIODS = [3, 5, 10]
THRESHOLDS = [0.25, 0.50, 1.00]
MIN_CONFIDENCES = [0.55, 0.60]

PIPELINE_CONFIG = PipelineConfig(
    sma_periods=[5, 10, 20, 50],
    volatility_windows=[10, 20],
    return_lags=[1, 2, 5],
    min_periods=100,
)


def sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    return float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS))


def evaluate(processed, raw, holding, threshold, min_conf) -> dict:
    bt_config = BacktestConfig(max_holding_days=holding, min_confidence=min_conf)
    model_config = ModelConfig(model_type="Random Forest", calibrate=True,
                               label_horizon=holding)

    def train_fn(df_train: pd.DataFrame):
        result = train_model(add_target(df_train, holding, threshold), model_config)
        return (result.model, result.feature_cols) if result else (None, None)

    folds, _ = walk_forward_backtest(
        processed, train_fn, bt_config,
        WalkForwardConfig(min_train_bars=250, test_bars=126, expanding=True,
                          embargo_bars=holding),
        raw_df=raw,
        pipeline_factory=lambda: create_pipeline(PIPELINE_CONFIG),
    )
    if not folds:
        return {"sharpe": float("nan"), "return_pct": float("nan"),
                "trades": 0, "p_value": float("nan")}

    returns = np.concatenate([
        pd.Series(f.portfolio_history).pct_change().dropna().to_numpy()
        for f in folds
    ])
    _, p_value = ttest_mean_return(returns)
    return {
        "sharpe": sharpe(returns),
        "return_pct": float(np.prod(1 + returns) - 1) * 100,
        "trades": sum(len(f.trades_df) for f in folds),
        "p_value": p_value,
    }


def main(ticker: str = "SPY", period: str = "10y") -> int:
    loaded = process_stock_data(ticker, period, PIPELINE_CONFIG)
    if not loaded["success"]:
        print(f"data fetch failed: {loaded['error']}")
        return 1
    processed, raw = loaded["data"], loaded["raw"]

    grid = list(itertools.product(HOLDING_PERIODS, THRESHOLDS, MIN_CONFIDENCES))
    print(f"{ticker} {period} · {len(processed)} bars · {len(grid)} configurations\n")
    print(f"{'hold':>5}{'thresh':>8}{'minconf':>9}"
          f"{'OOS sharpe':>12}{'return %':>10}{'trades':>8}{'p':>8}")
    print("-" * 60)

    rows = []
    for holding, threshold, min_conf in grid:
        result = evaluate(processed, raw, holding, threshold, min_conf)
        rows.append({"holding": holding, "threshold": threshold,
                     "min_confidence": min_conf, **result})
        print(f"{holding:>5}{threshold:>8.2f}{min_conf:>9.2f}"
              f"{result['sharpe']:>12.2f}{result['return_pct']:>10.2f}"
              f"{result['trades']:>8}{result['p_value']:>8.3f}")

    frame = pd.DataFrame(rows)
    out = ROOT / "docs" / f"sensitivity_{ticker.lower()}_{period}.csv"
    out.parent.mkdir(exist_ok=True)
    frame.to_csv(out, index=False)

    sharpes = frame["sharpe"].dropna()
    significant = int((frame["p_value"] < 0.05).sum())

    # An empty grid is a broken run, not a result. Every fold failing — a
    # scikit-learn incompatibility in the pipeline, say — used to leave every
    # cell NaN and still exit 0, so the CSV published as evidence could contain
    # nothing at all while the command looked like it had succeeded.
    if sharpes.empty:
        print("-" * 60)
        print(f"ERROR: all {len(frame)} configurations produced no tradeable "
              f"folds. This is a failure, not a finding — check the "
              f"walk_forward warnings (this script silences them below ERROR).")
        return 1

    print("-" * 60)
    print(f"OOS Sharpe across {len(sharpes)} configurations: "
          f"min {sharpes.min():.2f} · median {sharpes.median():.2f} · "
          f"max {sharpes.max():.2f}")
    print(f"configurations significant at p<0.05: {significant} of {len(frame)}")
    print(f"\nAt the 5% level you would expect about "
          f"{0.05 * len(frame):.1f} false positives by chance alone.")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
