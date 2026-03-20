"""
Statistical analysis for backtest results.

Provides:
- bootstrap_sharpe_ci: bootstrap 95% CI on annualised Sharpe
- ttest_mean_return: one-sample t-test (H₀: mean daily return = 0)
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_sharpe_ci(
    daily_returns: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
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
