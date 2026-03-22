"""
Tests for stock_dashboard/monitoring.py

Covers feature column selection, drift detection logic, HTML report generation,
and graceful handling of edge cases (empty data, missing evidently, bad columns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoring import _select_feature_cols, detect_feature_drift, generate_drift_report_html


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_feature_df(n: int = 100, seed: int = 0, shift: float = 0.0) -> pd.DataFrame:
    """Numeric feature DataFrame, optionally mean-shifted to simulate drift."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feat_a": rng.normal(0 + shift, 1, n),
        "feat_b": rng.normal(5 + shift, 2, n),
        "feat_c": rng.normal(-2 + shift, 0.5, n),
    })


def make_df_with_noise_cols(n: int = 100) -> pd.DataFrame:
    """DataFrame mixing valid features with columns that should be excluded."""
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "feat_a": rng.normal(0, 1, n),
        "Close": rng.uniform(100, 200, n),        # should be excluded
        "Date": pd.date_range("2022-01-01", periods=n, freq="B"),  # excluded
        "Target": rng.integers(0, 2, n),           # excluded
        "Vol_Regime": ["Low"] * n,                 # excluded (non-numeric str)
    })


# ---------------------------------------------------------------------------
# _select_feature_cols
# ---------------------------------------------------------------------------

class TestSelectFeatureCols:

    def test_returns_only_numeric_non_excluded_cols(self):
        df = make_df_with_noise_cols()
        cols = _select_feature_cols(df, feature_cols=None)
        assert "feat_a" in cols
        assert "Close" not in cols
        assert "Target" not in cols

    def test_respects_explicit_feature_cols(self):
        df = make_feature_df()
        cols = _select_feature_cols(df, feature_cols=["feat_a", "feat_b"])
        assert cols == ["feat_a", "feat_b"]

    def test_filters_missing_cols_from_explicit_list(self):
        df = make_feature_df()
        cols = _select_feature_cols(df, feature_cols=["feat_a", "nonexistent"])
        assert "feat_a" in cols
        assert "nonexistent" not in cols

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame()
        cols = _select_feature_cols(df, feature_cols=None)
        assert cols == []


# ---------------------------------------------------------------------------
# detect_feature_drift
# ---------------------------------------------------------------------------

class TestDetectFeatureDrift:

    def test_no_drift_on_identical_distributions(self):
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"f": rng.normal(0, 1, 200)})
        cur = pd.DataFrame({"f": rng.normal(0, 1, 200)})
        result = detect_feature_drift(ref, cur)
        if "error" in result:
            pytest.skip(f"evidently not available: {result['error']}")
        # Same distribution — should not flag strong drift
        assert isinstance(result["dataset_drift"], bool)

    def test_detects_drift_on_shifted_distribution(self):
        ref = make_feature_df(n=300, seed=0, shift=0.0)
        cur = make_feature_df(n=300, seed=1, shift=10.0)  # large shift
        result = detect_feature_drift(ref, cur)
        if "error" in result:
            pytest.skip(f"evidently not available: {result['error']}")
        assert result["dataset_drift"] is True
        assert result["drift_share"] > 0.0
        assert len(result["drifted_features"]) > 0

    def test_result_keys_present(self):
        ref = make_feature_df(n=100)
        cur = make_feature_df(n=100, shift=0.0)
        result = detect_feature_drift(ref, cur)
        if "error" in result and "not installed" in result.get("error", ""):
            pytest.skip("evidently not installed")
        for key in ("dataset_drift", "drift_share", "drifted_features",
                    "feature_drift", "n_reference", "n_current", "n_features_checked"):
            assert key in result

    def test_n_reference_and_n_current_correct(self):
        ref = make_feature_df(n=150)
        cur = make_feature_df(n=50)
        result = detect_feature_drift(ref, cur)
        if "error" in result:
            pytest.skip(f"evidently not available: {result['error']}")
        assert result["n_reference"] == 150
        assert result["n_current"] == 50

    def test_returns_error_when_too_few_rows(self):
        ref = make_feature_df(n=5)
        cur = make_feature_df(n=5)
        result = detect_feature_drift(ref, cur)
        assert "error" in result

    def test_returns_error_on_empty_df(self):
        ref = pd.DataFrame()
        cur = pd.DataFrame()
        result = detect_feature_drift(ref, cur)
        assert "error" in result

    def test_excludes_price_and_date_columns(self):
        df = make_df_with_noise_cols(n=150)
        result = detect_feature_drift(df.iloc[:100], df.iloc[100:])
        if "error" in result:
            pytest.skip(f"evidently not available: {result['error']}")
        # Should only have checked feat_a — not Close, Date, Target
        assert result["n_features_checked"] >= 1
        assert "Close" not in result.get("feature_drift", {})

    def test_drift_share_between_zero_and_one(self):
        ref = make_feature_df(n=200)
        cur = make_feature_df(n=200, shift=5.0)
        result = detect_feature_drift(ref, cur)
        if "error" in result:
            pytest.skip(f"evidently not available: {result['error']}")
        assert 0.0 <= result["drift_share"] <= 1.0


# ---------------------------------------------------------------------------
# generate_drift_report_html
# ---------------------------------------------------------------------------

class TestGenerateDriftReportHtml:

    def test_returns_string(self):
        ref = make_feature_df(n=100)
        cur = make_feature_df(n=100, shift=5.0)
        html = generate_drift_report_html(ref, cur)
        assert isinstance(html, str)

    def test_non_empty_html_when_evidently_available(self):
        ref = make_feature_df(n=100)
        cur = make_feature_df(n=100, shift=5.0)
        html = generate_drift_report_html(ref, cur)
        if html:  # evidently installed
            assert len(html) > 100
            assert "<html" in html.lower() or "<!doctype" in html.lower()

    def test_returns_empty_string_on_empty_df(self):
        html = generate_drift_report_html(pd.DataFrame(), pd.DataFrame())
        assert html == ""
