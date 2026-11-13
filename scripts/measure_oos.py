"""Measure out-of-sample performance and report it honestly.

Runs the same strategy three ways on identical data:

    in-sample        fit on every row, then scored on those same rows
    walk-forward     expanding folds, embargo, preprocessing fit per fold
    buy and hold     the benchmark that matters

The gap between the first two is the whole point of the exercise.

Usage:
    python scripts/measure_oos.py            # SPY, 2y
    python scripts/measure_oos.py QQQ 5y

Two flags reproduce the execution ablation in the README, holding data, model,
folds and seed fixed and relaxing one correction at a time:

    --no-lag        fill at the close of the bar the signal was read from
    --close-stops   check stops against the close, not the bar's range
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stock_dashboard"))

from backtest import BacktestConfig, run_backtest  # noqa: E402
from data_pipeline import PipelineConfig, create_pipeline, process_stock_data  # noqa: E402
from labels import add_target  # noqa: E402
from models import ModelConfig, train_model  # noqa: E402
from stats import bootstrap_sharpe_ci, ttest_mean_return  # noqa: E402
from walk_forward import WalkForwardConfig, walk_forward_backtest  # noqa: E402

HOLDING = 5
THRESHOLD = 0.5
TRADING_DAYS = 252

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


def main(*argv: str) -> int:
    flags = {a for a in argv if a.startswith("--")}
    unknown = flags - {"--no-lag", "--close-stops"}
    if unknown:
        print(f"unknown flag(s): {', '.join(sorted(unknown))}")
        return 2
    positional = [a for a in argv if not a.startswith("--")]
    ticker = positional[0] if positional else "SPY"
    period = positional[1] if len(positional) > 1 else "2y"
    execution_lag_bars = 0 if "--no-lag" in flags else 1
    intrabar_stops = "--close-stops" not in flags

    loaded = process_stock_data(ticker, period, PIPELINE_CONFIG)
    if not loaded["success"]:
        print(f"data fetch failed: {loaded['error']}")
        return 1

    processed, raw = loaded["data"], loaded["raw"]
    print(f"{ticker} · {len(processed)} bars · "
          f"{processed['Date'].min().date()} to {processed['Date'].max().date()}")
    print(f"holding {HOLDING}d · threshold {THRESHOLD}% · "
          f"costs {BacktestConfig().transaction_cost_pct:.1%} per leg")
    print(f"execution lag {execution_lag_bars} bar(s) · "
          f"stops checked {'intrabar' if intrabar_stops else 'on close'}\n")

    bt_config = BacktestConfig(max_holding_days=HOLDING,
                               execution_lag_bars=execution_lag_bars,
                               intrabar_stops=intrabar_stops)
    model_config = ModelConfig(model_type="Random Forest", calibrate=True,
                               label_horizon=HOLDING)

    # ---- in-sample: what the old headline panel reported -------------------
    labelled = add_target(processed, HOLDING, THRESHOLD)
    trained = train_model(labelled, model_config)
    if trained is None:
        print("training failed")
        return 1
    _, in_sample, in_history = run_backtest(
        labelled, trained.model, trained.feature_cols, bt_config
    )
    in_returns = pd.Series(in_history).pct_change().dropna().to_numpy()

    # ---- walk-forward: embargoed, preprocessing refit per fold -------------
    def train_fn(df_train: pd.DataFrame):
        result = train_model(add_target(df_train, HOLDING, THRESHOLD), model_config)
        return (result.model, result.feature_cols) if result else (None, None)

    folds, summary = walk_forward_backtest(
        processed, train_fn, bt_config,
        WalkForwardConfig(min_train_bars=250, test_bars=60, expanding=True,
                          embargo_bars=HOLDING),
        raw_df=raw,
        pipeline_factory=lambda: create_pipeline(PIPELINE_CONFIG),
    )
    if not folds:
        print(f"walk-forward produced no folds: {summary.get('error')}")
        return 1

    # Folds are skipped individually on a preprocessing error, so a partial run
    # is possible without an outright failure. Say so rather than quietly
    # reporting statistics computed from fewer windows than were requested.
    expected = len(range(250, len(processed) - 60 + 1, 60))
    if len(folds) < expected:
        print(f"WARNING: {expected - len(folds)} of {expected} folds were "
              f"skipped; the numbers below cover {len(folds)} folds only.")

    oos_returns = np.concatenate([
        pd.Series(f.portfolio_history).pct_change().dropna().to_numpy()
        for f in folds
    ])
    oos_growth = float(np.prod(1 + oos_returns) - 1) * 100

    # ---- benchmark ---------------------------------------------------------
    first_test = folds[0].test_start
    held = processed[processed["Date"] >= first_test]
    bh_return = float(held["Close"].iloc[-1] / held["Close"].iloc[0] - 1) * 100
    bh_returns = held["Close"].pct_change().dropna().to_numpy()

    line = "-" * 66
    print(line)
    print(f"{'':<22}{'in-sample':>14}{'walk-forward':>14}{'buy & hold':>14}")
    print(line)
    rows = [
        ("total return %", in_sample.get("total_return", float("nan")),
         oos_growth, bh_return),
        ("sharpe", sharpe(in_returns), sharpe(oos_returns), sharpe(bh_returns)),
        ("max drawdown %", in_sample.get("max_drawdown", float("nan")),
         summary.get("max_drawdown", float("nan")), float("nan")),
        ("trades", in_sample.get("num_trades", 0),
         sum(len(f.trades_df) for f in folds), 1),
    ]
    for name, a, b, c in rows:
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
        print(f"{name:<22}{fmt(a):>14}{fmt(b):>14}{fmt(c):>14}")
    print(line)

    print(f"\nmodel: CV AUC {trained.cv.auc_roc:.3f} · "
          f"accuracy {trained.cv.accuracy:.3f} (gap={HOLDING} bars)")
    print(f"walk-forward: {len(folds)} folds, embargo {HOLDING} bars, "
          f"preprocessing refit per fold")
    for f in folds:
        print(f"  fold {f.fold}: train->{f.train_end.date()} "
              f"test {f.test_start.date()}..{f.test_end.date()} "
              f"return={f.metrics.get('total_return', 0):+.2f}% "
              f"trades={len(f.trades_df)}")

    low, high, _ = bootstrap_sharpe_ci(oos_returns, n_boot=1000)
    t_stat, p_value = ttest_mean_return(oos_returns)
    straddles = low < 0 < high

    print(f"\nOOS daily returns: n={len(oos_returns)}")
    print(f"  bootstrap Sharpe 95% CI : [{low:.3f}, {high:.3f}]")
    print(f"  t-test mean return      : t={t_stat:.3f}, p={p_value:.4f}")
    print(f"\nVERDICT: {'no statistically detectable edge' if straddles else 'CI excludes zero'}"
          f" (95% CI {'includes' if straddles else 'excludes'} zero, p={p_value:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
