"""
Tests for stock_dashboard/data_pipeline.py

Covers each pipeline transformer in isolation and the composed pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline import (
    DataQualityChecker,
    DataScaler,
    FeatureSelector,
    FinalNaNHandler,
    MissingDataHandler,
    OutlierHandler,
    PipelineConfig,
    TechnicalIndicators,
    create_pipeline,
)


# ---------------------------------------------------------------------------
# DataQualityChecker
# ---------------------------------------------------------------------------

class TestDataQualityChecker:

    def test_passes_valid_data(self, raw_ohlcv):
        result = DataQualityChecker(min_periods=50).fit_transform(raw_ohlcv)
        assert len(result) >= 50

    def test_raises_on_insufficient_rows(self, raw_ohlcv):
        tiny = raw_ohlcv.head(10)
        with pytest.raises(ValueError, match="Insufficient data"):
            DataQualityChecker(min_periods=50).fit_transform(tiny)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="Empty DataFrame"):
            DataQualityChecker().fit_transform(pd.DataFrame())

    def test_removes_duplicates_with_ticker(self, raw_ohlcv):
        duped = pd.concat([raw_ohlcv, raw_ohlcv.head(5)], ignore_index=True)
        result = DataQualityChecker(min_periods=50).fit_transform(duped)
        assert len(result) == len(raw_ohlcv)

    def test_works_without_ticker_column(self, raw_ohlcv_no_ticker):
        result = DataQualityChecker(min_periods=50).fit_transform(raw_ohlcv_no_ticker)
        assert len(result) >= 50

    def test_output_is_sorted_by_date(self, raw_ohlcv):
        shuffled = raw_ohlcv.sample(frac=1, random_state=0).reset_index(drop=True)
        result = DataQualityChecker(min_periods=50).fit_transform(shuffled)
        assert result["Date"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# TechnicalIndicators
# ---------------------------------------------------------------------------

class TestTechnicalIndicators:

    def test_adds_sma_columns(self, raw_ohlcv):
        config = PipelineConfig(sma_periods=[10, 20], enable_technical=True)
        result = TechnicalIndicators(config).fit_transform(raw_ohlcv)
        assert "SMA_10" in result.columns
        assert "SMA_20" in result.columns

    def test_adds_rsi(self, raw_ohlcv):
        result = TechnicalIndicators().fit_transform(raw_ohlcv)
        assert "RSI_14" in result.columns
        # RSI should be in [0, 100]
        rsi = result["RSI_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_adds_macd(self, raw_ohlcv):
        result = TechnicalIndicators().fit_transform(raw_ohlcv)
        for col in ("MACD", "MACD_Signal", "MACD_Hist"):
            assert col in result.columns

    def test_adds_bollinger_bands(self, raw_ohlcv):
        result = TechnicalIndicators().fit_transform(raw_ohlcv)
        assert "BB_Width" in result.columns
        assert "BB_Position" in result.columns

    def test_disabled_returns_unchanged(self, raw_ohlcv):
        config = PipelineConfig(enable_technical=False)
        result = TechnicalIndicators(config).fit_transform(raw_ohlcv)
        assert set(result.columns) == set(raw_ohlcv.columns)

    def test_works_without_ticker(self, raw_ohlcv_no_ticker):
        result = TechnicalIndicators().fit_transform(raw_ohlcv_no_ticker)
        assert "RSI_14" in result.columns

    def test_price_ratio_bounded(self, raw_ohlcv):
        config = PipelineConfig(sma_periods=[20])
        result = TechnicalIndicators(config).fit_transform(raw_ohlcv)
        ratio = result["Price_SMA_20_Ratio"].dropna()
        # Price/SMA ratio for typical stocks is in (0.7, 1.5)
        assert ratio.between(0.5, 2.0).all()


# ---------------------------------------------------------------------------
# OutlierHandler
# ---------------------------------------------------------------------------

class TestOutlierHandler:

    def test_caps_extreme_returns(self, raw_ohlcv):
        df = raw_ohlcv.copy()
        # Inject an extreme return
        df.loc[10, "Return"] = 5.0  # 500% daily return

        handler = OutlierHandler(threshold=4.0)
        handler.fit(df)
        result = handler.transform(df)

        assert result.loc[10, "Return"] < 5.0, "Extreme return was not capped"

    def test_does_not_alter_normal_values(self, raw_ohlcv):
        handler = OutlierHandler(threshold=4.0)
        handler.fit(raw_ohlcv)
        result = handler.transform(raw_ohlcv)
        # Most return values should be unchanged
        unchanged = (result["Return"] == raw_ohlcv["Return"]).sum()
        assert unchanged / len(raw_ohlcv) > 0.95


# ---------------------------------------------------------------------------
# MissingDataHandler
# ---------------------------------------------------------------------------

class TestMissingDataHandler:

    def test_does_not_drop_protected_columns(self, raw_ohlcv):
        protected = ["Date", "Close", "Open", "High", "Low", "Volume", "Return"]
        df = raw_ohlcv.copy()
        # Add a high-missing feature column
        df["high_missing_feature"] = np.nan

        handler = MissingDataHandler(max_missing_ratio=0.2)
        handler.fit(df)
        result = handler.transform(df)

        for col in protected:
            assert col in result.columns, f"Protected column '{col}' was dropped"

    def test_drops_high_missing_feature_columns(self, raw_ohlcv):
        df = raw_ohlcv.copy()
        df["junk_feature"] = np.nan

        handler = MissingDataHandler(max_missing_ratio=0.2)
        handler.fit(df)
        result = handler.transform(df)

        assert "junk_feature" not in result.columns

    def test_forward_fills_missing_feature_values(self, raw_ohlcv):
        config = PipelineConfig(sma_periods=[10])
        df = TechnicalIndicators(config).fit_transform(raw_ohlcv)
        # Introduce some NaN in a feature column
        df.loc[5:8, "SMA_10"] = np.nan

        handler = MissingDataHandler()
        handler.fit(df)
        result = handler.transform(df)

        assert result["SMA_10"].isna().sum() == 0


# ---------------------------------------------------------------------------
# FeatureSelector
# ---------------------------------------------------------------------------

class TestFeatureSelector:

    def test_drops_highly_correlated_features(self):
        n = 200
        rng = np.random.default_rng(1)
        base = rng.standard_normal(n)
        df = pd.DataFrame({
            "feat_a": base,
            "feat_b": base + rng.normal(0, 1e-6, n),  # ~perfect correlation
            "feat_c": rng.standard_normal(n),           # independent
            "Close": rng.uniform(50, 200, n),
            "Date": pd.date_range("2022-01-01", periods=n),
        })

        selector = FeatureSelector(correlation_threshold=0.95)
        selector.fit(df)
        result = selector.transform(df)

        # One of feat_a / feat_b should be removed
        assert not ("feat_a" in result.columns and "feat_b" in result.columns)

    def test_never_drops_protected_columns(self, raw_ohlcv):
        selector = FeatureSelector(correlation_threshold=0.5)
        selector.fit(raw_ohlcv)
        result = selector.transform(raw_ohlcv)
        for col in ("Date", "Close", "Return"):
            assert col in result.columns


# ---------------------------------------------------------------------------
# DataScaler
# ---------------------------------------------------------------------------

class TestDataScaler:

    def test_does_not_scale_price_columns(self, processed_df):
        original_close = processed_df["Close"].copy()

        scaler = DataScaler()
        scaler.fit(processed_df)
        result = scaler.transform(processed_df)

        pd.testing.assert_series_equal(result["Close"], original_close, check_names=False)

    def test_scales_feature_columns(self, processed_df):
        # Pick a feature column that is not in the exclude set
        feature_col = None
        for col in processed_df.select_dtypes(include=[np.number]).columns:
            if col not in DataScaler._EXCLUDE and processed_df[col].std() > 0:
                feature_col = col
                break

        if feature_col is None:
            pytest.skip("No scalable feature column found in fixture")

        scaler = DataScaler(scale_by_group=False)
        scaler.fit(processed_df)
        result = scaler.transform(processed_df)

        # After RobustScaler, median should be ~0
        assert abs(result[feature_col].median()) < 1.0


# ---------------------------------------------------------------------------
# FinalNaNHandler
# ---------------------------------------------------------------------------

class TestFinalNaNHandler:

    def test_fills_feature_nans_with_zero(self):
        df = pd.DataFrame({
            "Close": [100.0, 101.0, 102.0],
            "some_feature": [1.0, np.nan, 3.0],
            "Date": pd.date_range("2022-01-01", periods=3),
        })
        result = FinalNaNHandler().fit_transform(df)
        assert result["some_feature"].isna().sum() == 0
        assert result.loc[1, "some_feature"] == 0.0

    def test_drops_rows_with_missing_prices(self):
        df = pd.DataFrame({
            "Close": [100.0, np.nan, 102.0],
            "some_feature": [1.0, 2.0, 3.0],
            "Date": pd.date_range("2022-01-01", periods=3),
        })
        result = FinalNaNHandler().fit_transform(df)
        assert len(result) == 2
        assert result["Close"].isna().sum() == 0


# ---------------------------------------------------------------------------
# End-to-end pipeline (no network call)
# ---------------------------------------------------------------------------

class TestCreatePipeline:

    def test_pipeline_runs_on_synthetic_data(self, raw_ohlcv):
        config = PipelineConfig(min_periods=50, sma_periods=[10, 20])
        pipe = create_pipeline(config)
        result = pipe.fit_transform(raw_ohlcv)

        assert not result.empty
        assert "Close" in result.columns
        assert result["Close"].min() > 0
        assert result.isnull().sum().sum() == 0

    def test_pipeline_preserves_close_prices(self, raw_ohlcv):
        config = PipelineConfig(min_periods=50, sma_periods=[10])
        pipe = create_pipeline(config)
        result = pipe.fit_transform(raw_ohlcv)

        # Close prices should remain in dollar terms (not scaled)
        assert result["Close"].min() > 10.0

    def test_pipeline_works_without_ticker(self, raw_ohlcv_no_ticker):
        config = PipelineConfig(min_periods=50)
        pipe = create_pipeline(config)
        result = pipe.fit_transform(raw_ohlcv_no_ticker)
        assert not result.empty
