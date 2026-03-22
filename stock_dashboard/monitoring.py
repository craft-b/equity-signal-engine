"""
Feature drift detection using Evidently AI.

Compares feature distributions between a reference window (historical baseline)
and a current window (recent data) to flag potential model staleness or data
distribution shift.

Usage:
    from monitoring import detect_feature_drift, generate_drift_report_html

    summary = detect_feature_drift(reference_df, current_df, feature_cols)
    html    = generate_drift_report_html(reference_df, current_df, feature_cols)
"""

from __future__ import annotations

import logging
import tempfile
import os
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
    _EVIDENTLY_AVAILABLE = True
except ImportError:
    _EVIDENTLY_AVAILABLE = False
    logger.warning("evidently not installed — drift detection unavailable")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _select_feature_cols(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]],
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """Return the intersection of feature_cols and df columns, filtering excludes."""
    _exclude = set(exclude or [
        "Date", "Ticker", "Open", "High", "Low", "Close", "Volume",
        "Return", "Log_Return", "Future_Return", "Target",
        "Dividends", "Stock Splits", "Capital Gains",
        "Rolling_Vol", "Vol_Regime",
    ])
    candidates = feature_cols if feature_cols is not None else list(df.columns)
    return [
        c for c in candidates
        if c in df.columns
        and c not in _exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Detect distribution drift between reference and current feature windows.

    Args:
        reference_df:  Historical baseline DataFrame (e.g. training period).
        current_df:    Recent DataFrame to compare against the baseline.
        feature_cols:  Columns to check. Defaults to all numeric non-price cols.

    Returns:
        dict with keys:
          dataset_drift      bool   — True if overall drift detected
          drift_share        float  — fraction of features that drifted
          drifted_features   list   — names of drifted columns
          feature_drift      dict   — {col: bool} per-column result
          n_reference        int
          n_current          int
          n_features_checked int
          error              str    — set only when detection could not run
    """
    if not _EVIDENTLY_AVAILABLE:
        return {"error": "evidently not installed", "dataset_drift": False}

    cols = _select_feature_cols(reference_df, feature_cols)

    if not cols:
        return {"error": "No valid feature columns found", "dataset_drift": False}

    if len(reference_df) < 10 or len(current_df) < 10:
        return {"error": "Insufficient rows for drift detection (need ≥ 10 each)", "dataset_drift": False}

    try:
        # Use explicit per-column metrics for reliable result parsing across versions
        metrics_list = [DatasetDriftMetric()]
        for col in cols:
            metrics_list.append(ColumnDriftMetric(column_name=col))

        report = Report(metrics=metrics_list)
        report.run(
            reference_data=reference_df[cols].reset_index(drop=True),
            current_data=current_df[cols].reset_index(drop=True),
        )
        result = report.as_dict()
    except Exception as exc:
        logger.error("Evidently drift report failed: %s", exc)
        return {"error": str(exc), "dataset_drift": False}

    metrics_results = result.get("metrics", [])

    # Dataset-level result
    dataset_metric = next(
        (m for m in metrics_results if m.get("metric") == "DatasetDriftMetric"), None
    )
    dataset_drift = False
    drift_share = 0.0
    if dataset_metric:
        dr = dataset_metric.get("result", {})
        dataset_drift = bool(dr.get("dataset_drift", False))
        drift_share = float(dr.get("share_of_drifted_columns", 0.0))

    # Per-column results from explicit ColumnDriftMetric entries
    feature_drift: Dict[str, bool] = {}
    for m in metrics_results:
        if m.get("metric") == "ColumnDriftMetric":
            col_result = m.get("result", {})
            col_name = col_result.get("column_name", "")
            if col_name:
                feature_drift[col_name] = bool(col_result.get("drift_detected", False))

    drifted = [c for c, d in feature_drift.items() if d]

    return {
        "dataset_drift": dataset_drift,
        "drift_share": drift_share,
        "drifted_features": drifted,
        "feature_drift": feature_drift,
        "n_reference": len(reference_df),
        "n_current": len(current_df),
        "n_features_checked": len(cols),
    }


def generate_drift_report_html(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> str:
    """
    Generate a full Evidently HTML drift report as a string.

    Returns an empty string if evidently is not installed or an error occurs.
    """
    if not _EVIDENTLY_AVAILABLE:
        return ""

    cols = _select_feature_cols(reference_df, feature_cols)
    if not cols:
        return ""

    try:
        metrics_list = [DatasetDriftMetric()]
        for col in cols:
            metrics_list.append(ColumnDriftMetric(column_name=col))
        report = Report(metrics=metrics_list)
        report.run(
            reference_data=reference_df[cols].reset_index(drop=True),
            current_data=current_df[cols].reset_index(drop=True),
        )
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            tmp_path = f.name

        report.save_html(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            html = f.read()
        os.unlink(tmp_path)
        return html
    except Exception as exc:
        logger.error("HTML report generation failed: %s", exc)
        return ""
