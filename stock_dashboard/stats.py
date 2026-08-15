"""
Statistical analysis for backtest results.

Provides:
- bootstrap_sharpe_ci: bootstrap 95% CI on annualised Sharpe
- ttest_mean_return: one-sample t-test (H₀: mean daily return = 0)
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _default_block_length(n: int) -> float:
    """Mean block length for the stationary bootstrap.

    n**(1/3) is the standard rule of thumb: long enough to carry the
    short-horizon autocorrelation and volatility clustering in daily returns,
    short enough that blocks stay numerous. For ~2,200 observations this gives
    roughly 13 trading days.
    """
    return max(2.0, float(n ** (1.0 / 3.0)))


def _stationary_indices(
    rng: np.random.Generator, n: int, n_boot: int, mean_block: float
) -> np.ndarray:
    """Index matrix for the Politis-Romano stationary bootstrap.

    Each resample walks forward through the series, and at every step restarts
    at a fresh random position with probability 1/mean_block. Block lengths are
    therefore geometric with the requested mean, and the resampled series is
    stationary — unlike a fixed-block bootstrap, which distorts the ends.
    """
    restart_prob = 1.0 / mean_block
    indices = np.empty((n_boot, n), dtype=np.int64)
    indices[:, 0] = rng.integers(0, n, size=n_boot)

    restarts = rng.random((n_boot, n - 1)) < restart_prob
    fresh = rng.integers(0, n, size=(n_boot, n - 1))
    for t in range(1, n):
        carried = (indices[:, t - 1] + 1) % n
        indices[:, t] = np.where(restarts[:, t - 1], fresh[:, t - 1], carried)
    return indices


def bootstrap_sharpe_ci(
    daily_returns: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
    method: str = "stationary",
    mean_block: float | None = None,
) -> tuple[float, float, np.ndarray]:
    """
    Bootstrap confidence interval for the annualised Sharpe ratio.

    Args:
        daily_returns: Array of daily portfolio returns.
        n_boot:        Number of bootstrap resamples.
        ci:            Confidence level (e.g. 0.95 for 95% CI).
        random_state:  RNG seed for reproducibility.

    Returns:
        (lower, upper, boot_sharpes): CI bounds and the full bootstrap
        distribution.  Returns (nan, nan, empty array) if there is
        insufficient data.
    """
    returns = np.asarray(daily_returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 10:
        return (np.nan, np.nan, np.array([]))

    rng = np.random.default_rng(random_state)
    n = len(returns)
    boot_sharpes = np.empty(n_boot)

    if method == "stationary":
        block = mean_block or _default_block_length(n)
        indices = _stationary_indices(rng, n, n_boot, block)
        samples = returns[indices]
        stds = samples.std(axis=1)
        means = samples.mean(axis=1)
        boot_sharpes = np.where(stds > 0, means / np.where(stds > 0, stds, 1)
                                * np.sqrt(252), 0.0)
    else:
        for i in range(n_boot):
            sample = rng.choice(returns, size=n, replace=True)
            std = sample.std()
            boot_sharpes[i] = sample.mean() / std * np.sqrt(252) if std > 0 else 0.0

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_sharpes, alpha * 100))
    upper = float(np.percentile(boot_sharpes, (1.0 - alpha) * 100))
    return lower, upper, boot_sharpes


def ttest_mean_return(daily_returns: np.ndarray) -> tuple[float, float]:
    """
    One-sample t-test against H₀: mean daily return = 0 (two-tailed).

    Args:
        daily_returns: Array of daily portfolio returns.

    Returns:
        (t_statistic, p_value).  Returns (nan, nan) if there is
        insufficient data.
    """
    returns = np.asarray(daily_returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 10:
        return (np.nan, np.nan)

    t_stat, p_value = stats.ttest_1samp(returns, 0.0)
    return float(t_stat), float(p_value)
