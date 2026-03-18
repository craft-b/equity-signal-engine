"""
Model training, cross-validation, calibration, and evaluation.

Design decisions:
- TimeSeriesSplit throughout — no random shuffling of temporal data.
- LogisticRegression runs inside a StandardScaler Pipeline.
- RandomForest is depth-constrained (max_depth=5, min_samples_leaf=25)
  to resist overfitting on financial noise.
- CalibratedClassifierCV (isotonic) applied after CV to produce
  well-calibrated probabilities — essential for signal quality analysis.
- Evaluation reports probabilistic metrics (Brier, log-loss, AUC-ROC)
  alongside classification metrics (accuracy, precision, recall, F1)
  because probability quality matters as much as threshold accuracy
  for downstream signal generation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_type: str = "Random Forest"      # "Random Forest" | "Logistic Regression"
    n_cv_splits: int = 5                   # TimeSeriesSplit folds
    min_train_samples: int = 100
    min_feature_count: int = 5
    calibrate: bool = True                 # Apply isotonic calibration
    # Random Forest hyperparameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 5
    rf_min_samples_split: int = 50
    rf_min_samples_leaf: int = 25
    # Logistic Regression hyperparameters
    lr_C: float = 1.0
    random_state: int = 42
    exclude_cols: List[str] = field(default_factory=lambda: [
        "Date", "Ticker", "Open", "High", "Low", "Close", "Volume",
        "Return", "Log_Return", "Future_Return", "Target",
        "Dividends", "Stock Splits", "Capital Gains",
    ])


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CVMetrics:
    """Mean cross-validated metrics across all folds."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    brier: float
    log_loss: float
    auc_roc: float
    # Standard deviations for error bars
    accuracy_std: float = 0.0
    brier_std: float = 0.0
    auc_roc_std: float = 0.0


@dataclass
class TrainResult:
    model: object                  # Final fitted sklearn estimator
    feature_cols: List[str]
    cv: CVMetrics
    n_features: int
    n_samples: int
    train_time_s: float
    calibration_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None
    feature_importance: Optional[pd.DataFrame] = None
    positive_rate: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_base_estimator(config: ModelConfig):
    """Return an uncalibrated sklearn estimator."""
    if config.model_type == "Logistic Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=config.lr_C,
                class_weight="balanced",
                max_iter=1000,
                random_state=config.random_state,
            )),
        ])
    return RandomForestClassifier(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        min_samples_split=config.rf_min_samples_split,
        min_samples_leaf=config.rf_min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced",
        random_state=config.random_state,
        n_jobs=-1,
    )


def _safe_auc(y_true: pd.Series, probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return 0.5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_ml_dataset(
    df: pd.DataFrame,
    config: ModelConfig,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], List[str]]:
    """
    Select feature columns and target from a processed DataFrame.

    Returns (X, y, feature_cols) or (None, None, []) when data is
    insufficient or the target column is missing.
    """
    if "Target" not in df.columns:
        logger.error("Target column missing from DataFrame")
        return None, None, []

    feature_cols = [
        c for c in df.columns
        if c not in config.exclude_cols
        and df[c].notna().mean() > 0.7
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(feature_cols) < config.min_feature_count:
        logger.warning(
            "Only %d features available (minimum %d)",
            len(feature_cols), config.min_feature_count,
        )
        return None, None, []

    ml_df = (
        df[feature_cols + ["Target"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(ml_df) < config.min_train_samples:
        logger.warning(
            "Only %d clean samples (minimum %d)",
            len(ml_df), config.min_train_samples,
        )
        return None, None, []

    counts = ml_df["Target"].value_counts()
    if len(counts) < 2 or counts.min() < 10:
        logger.warning("Class imbalance too severe: %s", counts.to_dict())
        return None, None, []

    return ml_df[feature_cols], ml_df["Target"], feature_cols


def train_model(
    df: pd.DataFrame,
    config: Optional[ModelConfig] = None,
) -> Optional[TrainResult]:
    """
    Train a classifier with time-series cross-validation.

    The model fit on the full dataset is optionally wrapped in
    CalibratedClassifierCV for better probability estimates.

    Returns None if model_type == "Technical Only" or data is insufficient.
    """
    config = config or ModelConfig()

    if config.model_type == "Technical Only":
        return None

    X, y, feature_cols = prepare_ml_dataset(df, config)
    if X is None:
        return None

    n_splits = min(config.n_cv_splits, max(2, len(X) // 50))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_acc, cv_prec, cv_rec, cv_f1 = [], [], [], []
    cv_brier, cv_ll, cv_auc = [], [], []

    base = _build_base_estimator(config)
    t0 = time.perf_counter()

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        if len(y_tr.unique()) < 2:
            continue

        base.fit(X_tr, y_tr)
        y_pred = base.predict(X_te)
        probs = base.predict_proba(X_te)[:, 1]

        cv_acc.append(accuracy_score(y_te, y_pred))
        cv_prec.append(precision_score(y_te, y_pred, zero_division=0))
        cv_rec.append(recall_score(y_te, y_pred, zero_division=0))
        cv_f1.append(f1_score(y_te, y_pred, zero_division=0))
        cv_brier.append(brier_score_loss(y_te, probs))
        cv_ll.append(log_loss(y_te, probs))
        cv_auc.append(_safe_auc(y_te, probs))

    if not cv_acc:
        logger.error("All CV folds skipped — insufficient class diversity")
        return None

    # Final fit: optionally calibrated
    if config.calibrate:
        final_model = CalibratedClassifierCV(
            _build_base_estimator(config), method="isotonic", cv=3
        )
    else:
        final_model = _build_base_estimator(config)

    final_model.fit(X, y)
    elapsed = time.perf_counter() - t0

    # Calibration curve on training data (indicative, not evaluative)
    final_probs = final_model.predict_proba(X)[:, 1]
    cal_true, cal_pred = calibration_curve(y, final_probs, n_bins=10, strategy="quantile")

    # Feature importance (from base estimator)
    imp_df = _extract_importance(base, feature_cols)

    cv_metrics = CVMetrics(
        accuracy=float(np.mean(cv_acc)),
        precision=float(np.mean(cv_prec)),
        recall=float(np.mean(cv_rec)),
        f1=float(np.mean(cv_f1)),
        brier=float(np.mean(cv_brier)),
        log_loss=float(np.mean(cv_ll)),
        auc_roc=float(np.mean(cv_auc)),
        accuracy_std=float(np.std(cv_acc)),
        brier_std=float(np.std(cv_brier)),
        auc_roc_std=float(np.std(cv_auc)),
    )

    return TrainResult(
        model=final_model,
        feature_cols=feature_cols,
        cv=cv_metrics,
        n_features=len(feature_cols),
        n_samples=len(X),
        train_time_s=elapsed,
        calibration_curve=(cal_true, cal_pred),
        feature_importance=imp_df,
        positive_rate=float(y.mean()),
    )


def _extract_importance(estimator, feature_cols: List[str]) -> Optional[pd.DataFrame]:
    """Pull feature importances from the base estimator (RF or LR)."""
    if hasattr(estimator, "feature_importances_"):
        imp = estimator.feature_importances_
    elif hasattr(estimator, "named_steps"):
        clf = estimator.named_steps.get("clf")
        if clf is not None and hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_[0])
        else:
            return None
    else:
        return None

    return (
        pd.DataFrame({"feature": feature_cols, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Return a dict of held-out evaluation metrics for a fitted model."""
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "brier": brier_score_loss(y_test, probs),
        "log_loss": log_loss(y_test, probs),
        "auc_roc": _safe_auc(y_test, probs),
    }


def generate_signals(
    df: pd.DataFrame,
    model,
    feature_cols: List[str],
    min_prob: float = 0.55,
    enable_shorts: bool = False,
) -> pd.DataFrame:
    """
    Attach probability and signal columns to df.

    Adds: prob_up, prob_down, signal_strength, signal (1 / -1 / 0).
    """
    out = df.copy()
    X = out[feature_cols].ffill().fillna(0).replace([np.inf, -np.inf], 0)
    probs = model.predict_proba(X)[:, 1]

    out["prob_up"] = probs
    out["prob_down"] = 1.0 - probs
    out["signal_strength"] = np.abs(probs - 0.5)
    out["signal"] = 0
    out.loc[out["prob_up"] >= min_prob, "signal"] = 1
    if enable_shorts:
        out.loc[out["prob_up"] <= (1.0 - min_prob), "signal"] = -1

    return out


# ---------------------------------------------------------------------------
# Module smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 500
    df_test = pd.DataFrame(rng.standard_normal((n, 8)), columns=[f"feat_{i}" for i in range(8)])
    df_test["Target"] = rng.integers(0, 2, size=n)
    df_test["Date"] = pd.date_range("2022-01-01", periods=n, freq="B")

    cfg = ModelConfig(model_type="Random Forest", min_train_samples=50)
    result = train_model(df_test, cfg)
    if result:
        print(f"CV accuracy={result.cv.accuracy:.3f}  AUC-ROC={result.cv.auc_roc:.3f}  Brier={result.cv.brier:.3f}")
        print("Smoke test passed.")
    else:
        print("train_model returned None.")
