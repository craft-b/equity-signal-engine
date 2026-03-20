"""
Tests for stock_dashboard/stats.py.

Covers bootstrap_sharpe_ci and ttest_mean_return.
"""

from __future__ import annotations

import numpy as np
import pytest

from stats import bootstrap_sharpe_ci, ttest_mean_return


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(n: int = 252, mean: float = 0.001, std: float = 0.01, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, n)


# ---------------------------------------------------------------------------
# bootstrap_sharpe_ci
# ---------------------------------------------------------------------------

class TestBootstrapSharpeCi:
    def test_returns_three_tuple(self):
        r = _make_returns()
        result = bootstrap_sharpe_ci(r)
        assert len(result) == 3

    def test_lower_less_than_upper(self):
        r = _make_returns()
        lo, hi, _ = bootstrap_sharpe_ci(r)
        assert lo < hi

    def test_distribution_length_matches_n_boot(self):
        r = _make_returns()
        _, _, dist = bootstrap_sharpe_ci(r, n_boot=500)
        assert len(dist) == 500

    def test_positive_expected_return_yields_positive_upper_bound(self):
        r = _make_returns(mean=0.002)
        _, hi, _ = bootstrap_sharpe_ci(r)
        assert hi > 0

    def test_insufficient_data_returns_nan(self):
        lo, hi, dist = bootstrap_sharpe_ci(_make_returns(n=5))
        assert np.isnan(lo)
        assert np.isnan(hi)
        assert len(dist) == 0

    def test_empty_array_returns_nan(self):
        lo, hi, dist = bootstrap_sharpe_ci(np.array([]))
        assert np.isnan(lo)
        assert np.isnan(hi)

    def test_reproducible_with_same_seed(self):
        r = _make_returns()
        lo1, hi1, _ = bootstrap_sharpe_ci(r, random_state=7)
        lo2, hi2, _ = bootstrap_sharpe_ci(r, random_state=7)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_different_seeds_may_differ(self):
        r = _make_returns()
        lo1, _, _ = bootstrap_sharpe_ci(r, random_state=1)
        lo2, _, _ = bootstrap_sharpe_ci(r, random_state=2)
        # Very unlikely to be equal
        assert lo1 != lo2

    def test_ci_width_narrows_with_more_data(self):
        small = _make_returns(n=100)
        large = _make_returns(n=2000, seed=0)
        lo_s, hi_s, _ = bootstrap_sharpe_ci(small)
        lo_l, hi_l, _ = bootstrap_sharpe_ci(large)
        assert (hi_s - lo_s) > (hi_l - lo_l)

    def test_ignores_inf_values(self):
        r = _make_returns()
        r_with_inf = np.concatenate([r, [np.inf, -np.inf]])
        lo, hi, _ = bootstrap_sharpe_ci(r_with_inf)
        assert np.isfinite(lo)
        assert np.isfinite(hi)


# ---------------------------------------------------------------------------
# ttest_mean_return
# ---------------------------------------------------------------------------

class TestTtestMeanReturn:
    def test_returns_two_values(self):
        r = _make_returns()
        result = ttest_mean_return(r)
        assert len(result) == 2

    def test_strongly_positive_return_is_significant(self):
        r = _make_returns(n=500, mean=0.005, std=0.005)
        t, p = ttest_mean_return(r)
        assert t > 0
        assert p < 0.05

    def test_zero_mean_return_not_significant(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.0, 0.01, 2000)
        _, p = ttest_mean_return(r)
        assert p > 0.05

    def test_pvalue_in_unit_interval(self):
        r = _make_returns()
        _, p = ttest_mean_return(r)
        assert 0.0 <= p <= 1.0

    def test_insufficient_data_returns_nan(self):
        t, p = ttest_mean_return(np.array([0.01, 0.02]))
        assert np.isnan(t)
        assert np.isnan(p)

    def test_empty_array_returns_nan(self):
        t, p = ttest_mean_return(np.array([]))
        assert np.isnan(t)
        assert np.isnan(p)

    def test_negative_return_negative_tstat(self):
        r = _make_returns(n=500, mean=-0.005, std=0.005)
        t, _ = ttest_mean_return(r)
        assert t < 0

    def test_ignores_nan_values(self):
        r = _make_returns()
        r_with_nan = np.concatenate([r, [np.nan]])
        t, p = ttest_mean_return(r_with_nan)
        assert np.isfinite(t)
        assert np.isfinite(p)
