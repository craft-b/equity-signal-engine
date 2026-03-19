"""
Tests for stock_dashboard/models.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models import (
    ModelConfig,
    TrainResult,
    evaluate_model,
    generate_signals,
    prepare_ml_dataset,
    train_model,
)


# ---------------------------------------------------------------------------
# prepare_ml_dataset
# ---------------------------------------------------------------------------

class TestPrepareMLDataset:

    def test_returns_none_when_target_missing(self, ml_df):
        df_no_target = ml_df.drop(columns=["Target"])
        X, y, cols = prepare_ml_dataset(df_no_target, ModelConfig())
        assert X is None and y is None and cols == []

    def test_excludes_configured_columns(self, ml_df):
        cfg = ModelConfig()
        X, y, cols = prepare_ml_dataset(ml_df, cfg)
        assert X is not None
        for excluded in cfg.exclude_cols:
            assert excluded not in cols

    def test_returns_none_when_single_class(self, ml_df):
        df = ml_df.copy()
        df["Target"] = 1  # only one class
        X, y, cols = prepare_ml_dataset(df, ModelConfig())
        assert X is None

    def test_returns_none_when_minority_class_too_small(self, ml_df):
        df = ml_df.copy()
        # Set all but 5 rows to class 0 — minority class < 10
        df["Target"] = 0
        df.loc[:4, "Target"] = 1
        X, y, cols = prepare_ml_dataset(df, ModelConfig())
        assert X is None

    def test_returns_none_for_insufficient_samples(self, ml_df):
        cfg = ModelConfig(min_train_samples=10_000)
        X, y, cols = prepare_ml_dataset(ml_df, cfg)
        assert X is None

    def test_returns_none_for_too_few_features(self, ml_df):
        cfg = ModelConfig(min_feature_count=1_000)
        X, y, cols = prepare_ml_dataset(ml_df, cfg)
        assert X is None

    def test_output_shapes_are_consistent(self, ml_df):
        cfg = ModelConfig()
        X, y, cols = prepare_ml_dataset(ml_df, cfg)
        assert X is not None
        assert len(X) == len(y)
        assert list(X.columns) == cols

    def test_no_inf_or_nan_in_X(self, ml_df):
        X, y, cols = prepare_ml_dataset(ml_df, ModelConfig())
        assert X is not None
        assert not X.isin([np.inf, -np.inf]).any().any()
        assert not X.isna().any().any()


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

class TestTrainModel:

    def test_returns_none_for_technical_only(self, ml_df):
        cfg = ModelConfig(model_type="Technical Only")
        assert train_model(ml_df, cfg) is None

    def test_returns_train_result_for_random_forest(self, ml_df):
        cfg = ModelConfig(model_type="Random Forest", min_train_samples=50, n_cv_splits=3)
        result = train_model(ml_df, cfg)
        assert isinstance(result, TrainResult)

    def test_returns_train_result_for_logistic_regression(self, ml_df):
        cfg = ModelConfig(model_type="Logistic Regression", min_train_samples=50, n_cv_splits=3)
        result = train_model(ml_df, cfg)
        assert isinstance(result, TrainResult)

    def test_cv_metrics_are_bounded(self, ml_df):
        cfg = ModelConfig(model_type="Random Forest", min_train_samples=50, n_cv_splits=3)
        result = train_model(ml_df, cfg)
        assert result is not None
        assert 0.0 <= result.cv.accuracy <= 1.0
        assert 0.0 <= result.cv.auc_roc <= 1.0
        assert 0.0 <= result.cv.brier <= 1.0
        assert result.cv.log_loss >= 0.0

    def test_feature_cols_subset_of_df_columns(self, ml_df):
        cfg = ModelConfig(model_type="Random Forest", min_train_samples=50, n_cv_splits=3)
        result = train_model(ml_df, cfg)
        assert result is not None
        assert all(c in ml_df.columns for c in result.feature_cols)

    def test_calibration_curve_shapes(self, ml_df):
        cfg = ModelConfig(model_type="Logistic Regression", calibrate=True, min_train_samples=50, n_cv_splits=3)
        result = train_model(ml_df, cfg)
        assert result is not None
        assert result.calibration_curve is not None
        cal_true, cal_pred = result.calibration_curve
        assert len(cal_true) == len(cal_pred)

    def test_uncalibrated_model_still_returns_result(self, ml_df):
        cfg = ModelConfig(model_type="Random Forest", calibrate=False, min_train_samples=50, n_cv_splits=3)
        result = train_model(ml_df, cfg)
        assert result is not None

    def test_feature_importance_returned_for_rf(self, ml_df):
        cfg = ModelConfig(model_type="Random Forest", min_train_samples=50, n_cv_splits=3)
        result = train_model(ml_df, cfg)
        assert result is not None
        assert result.feature_importance is not None
        assert "feature" in result.feature_importance.columns
        assert "importance" in result.feature_importance.columns

    def test_n_samples_matches_dataset_size(self, ml_df):
        cfg = ModelConfig(model_type="Random Forest", min_train_samples=50, n_cv_splits=3)
        result = train_model(ml_df, cfg)
        assert result is not None
        # n_samples should be close to (but possibly less than) len(ml_df) after dropna
        assert result.n_samples <= len(ml_df)
        assert result.n_samples > 0

    def test_train_time_is_positive(self, ml_df):
        cfg = ModelConfig(model_type="Random Forest", min_train_samples=50, n_cv_splits=3)
        result = train_model(ml_df, cfg)
        assert result is not None
        assert result.train_time_s > 0


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:

    def test_returns_all_expected_keys(self, ml_df):
        cfg = ModelConfig(model_type="Random Forest", min_train_samples=50, n_cv_splits=3)
        result = train_model(ml_df, cfg)
        assert result is not None

        X = ml_df[result.feature_cols].fillna(0)
        y = ml_df["Target"]
        metrics = evaluate_model(result.model, X, y)

        for key in ("accuracy", "precision", "recall", "f1", "brier", "auc_roc"):
            assert key in metrics

    def test_metrics_are_bounded(self, ml_df):
        cfg = ModelConfig(model_type="Random Forest", min_train_samples=50, n_cv_splits=3)
        result = train_model(ml_df, cfg)
        assert result is not None

        X = ml_df[result.feature_cols].fillna(0)
        y = ml_df["Target"]
        metrics = evaluate_model(result.model, X, y)

        for k in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= metrics[k] <= 1.0
        assert 0.0 <= metrics["brier"] <= 1.0


# ---------------------------------------------------------------------------
# generate_signals
# ---------------------------------------------------------------------------

class TestGenerateSignals:

    def test_adds_required_columns(self, ml_df):
        cfg = ModelConfig(model_type="Random Forest", min_train_samples=50, n_cv_splits=3)
        result = train_model(ml_df, cfg)
        assert result is not None

        out = generate_signals(ml_df, result.model, result.feature_cols)
        for col in ("prob_up", "prob_down", "signal_strength", "signal"):
            assert col in out.columns

    def test_probabilities_sum_to_one(self, ml_df):
        cfg = ModelConfig(model_type="Random Forest", min_train_samples=50, n_cv_splits=3)
        result = train_model(ml_df, cfg)
        assert result is not None

        out = generate_signals(ml_df, result.model, result.feature_cols)
        prob_sum = out["prob_up"] + out["prob_down"]
        assert (prob_sum.round(6) == 1.0).all()

    def test_signals_only_contain_valid_values(self, ml_df):
        cfg = ModelConfig(model_type="Random Forest", min_train_samples=50, n_cv_splits=3)
        result = train_model(ml_df, cfg)
        assert result is not None

        out = generate_signals(ml_df, result.model, result.feature_cols, enable_shorts=True)
        assert set(out["signal"].unique()).issubset({-1, 0, 1})

    def test_no_short_signals_when_disabled(self, ml_df):
        cfg = ModelConfig(model_type="Random Forest", min_train_samples=50, n_cv_splits=3)
        result = train_model(ml_df, cfg)
        assert result is not None

        out = generate_signals(ml_df, result.model, result.feature_cols, enable_shorts=False)
        assert -1 not in out["signal"].values
