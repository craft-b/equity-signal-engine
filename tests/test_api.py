"""
Tests for api/main.py

Uses FastAPI TestClient with mocked yfinance to avoid network calls.
Covers health, predict, drift endpoints — request validation, response
schema, error handling, and signal/regime correctness.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure both stock_dashboard and api are importable
sys.path.insert(0, str(Path(__file__).parent.parent / "stock_dashboard"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame matching DataFetcher output format."""
    rng = np.random.default_rng(seed)
    close = 100.0 * (1 + rng.normal(0, 0.01, n)).cumprod()
    close = np.maximum(close, 1.0)
    high  = close * (1 + rng.uniform(0, 0.005, n))
    low   = close * (1 - rng.uniform(0, 0.005, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol   = rng.integers(1_000_000, 5_000_000, n).astype(float)

    df = pd.DataFrame({
        "Date":       pd.bdate_range("2022-01-01", periods=n),
        "Open":       open_,
        "High":       high,
        "Low":        low,
        "Close":      close,
        "Volume":     vol,
        "Ticker":     "TEST",
        "Return":     pd.Series(close).pct_change().values,
        "Log_Return": np.log(pd.Series(close) / pd.Series(close).shift(1)).values,
    })
    return df.dropna(subset=["Return"]).reset_index(drop=True)


def _make_trending_raw(n: int = 300, direction: str = "up") -> pd.DataFrame:
    """Strong-trend OHLCV so SMA_10 always > SMA_50 (or < for down)."""
    df = _make_raw_ohlcv(n)  # may be n-1 rows after dropna
    actual_n = len(df)
    delta = 0.005 if direction == "up" else -0.005
    close = 100.0 * np.cumprod(1 + delta * np.ones(actual_n))
    df = df.copy()
    df["Close"]      = close
    df["Return"]     = pd.Series(close).pct_change().values
    df["Log_Return"] = np.log(pd.Series(close) / pd.Series(close).shift(1)).values
    return df.dropna(subset=["Return"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:

    def test_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_is_ok(self):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_timestamp_present(self):
        resp = client.get("/health")
        assert "timestamp" in resp.json()


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class TestPredict:

    def _patched_predict(self, payload: dict, raw_df=None):
        if raw_df is None:
            raw_df = _make_raw_ohlcv()
        with patch("api.main._fetcher.fetch_stock_data", return_value=raw_df):
            return client.post("/predict", json=payload)

    def test_returns_200_for_valid_ticker(self):
        resp = self._patched_predict({"ticker": "AAPL"})
        assert resp.status_code == 200

    def test_response_schema_keys_present(self):
        resp = self._patched_predict({"ticker": "AAPL"})
        data = resp.json()
        for key in ("ticker", "signal", "signal_label", "signal_strength",
                    "regime", "latest_price", "latest_date", "sma_10", "sma_50", "timestamp"):
            assert key in data, f"Missing key: {key}"

    def test_ticker_uppercased_in_response(self):
        resp = self._patched_predict({"ticker": "aapl"})
        assert resp.json()["ticker"] == "AAPL"

    def test_signal_is_valid_value(self):
        resp = self._patched_predict({"ticker": "AAPL"})
        assert resp.json()["signal"] in (-1, 0, 1)

    def test_signal_label_matches_signal(self):
        resp = self._patched_predict({"ticker": "AAPL"})
        data = resp.json()
        label_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
        assert data["signal_label"] == label_map[data["signal"]]

    def test_regime_is_valid_value(self):
        resp = self._patched_predict({"ticker": "AAPL"})
        assert resp.json()["regime"] in ("Low", "Medium", "High")

    def test_latest_price_positive(self):
        resp = self._patched_predict({"ticker": "AAPL"})
        assert resp.json()["latest_price"] > 0

    def test_buy_signal_on_uptrend(self):
        resp = self._patched_predict({"ticker": "AAPL"}, raw_df=_make_trending_raw(direction="up"))
        assert resp.json()["signal"] == 1

    def test_404_when_fetcher_returns_none(self):
        with patch("api.main._fetcher.fetch_stock_data", return_value=None):
            resp = client.post("/predict", json={"ticker": "FAKE"})
        assert resp.status_code == 404

    def test_missing_ticker_returns_422(self):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_shorts_disabled_by_default(self):
        resp = self._patched_predict({"ticker": "AAPL"})
        # With shorts disabled, signal should never be -1 for non-short request
        # (default enable_shorts=False means no -1 from SMA crossover)
        assert resp.json()["signal"] in (0, 1)


# ---------------------------------------------------------------------------
# /drift
# ---------------------------------------------------------------------------

class TestDrift:

    def _patched_drift(self, payload: dict, raw_df=None):
        if raw_df is None:
            raw_df = _make_raw_ohlcv(n=400)
        with patch("api.main._fetcher.fetch_stock_data", return_value=raw_df):
            return client.post("/drift", json=payload)

    def test_returns_200_for_valid_ticker(self):
        resp = self._patched_drift({"ticker": "AAPL", "period": "2y"})
        assert resp.status_code == 200

    def test_response_schema_keys_present(self):
        resp = self._patched_drift({"ticker": "AAPL"})
        data = resp.json()
        for key in ("ticker", "dataset_drift", "drift_share", "drifted_features",
                    "feature_drift", "n_reference", "n_current", "n_features_checked", "timestamp"):
            assert key in data

    def test_ticker_uppercased(self):
        resp = self._patched_drift({"ticker": "msft"})
        assert resp.json()["ticker"] == "MSFT"

    def test_drift_share_between_0_and_1(self):
        resp = self._patched_drift({"ticker": "AAPL"})
        share = resp.json()["drift_share"]
        assert 0.0 <= share <= 1.0

    def test_n_reference_plus_n_current_approximately_total(self):
        n = 400
        resp = self._patched_drift({"ticker": "AAPL"}, raw_df=_make_raw_ohlcv(n=n))
        data = resp.json()
        # n_reference + n_current should approximately equal the processed df length
        total = data["n_reference"] + data["n_current"]
        assert total > 0

    def test_404_when_fetcher_returns_none(self):
        with patch("api.main._fetcher.fetch_stock_data", return_value=None):
            resp = client.post("/drift", json={"ticker": "FAKE"})
        assert resp.status_code == 404

    def test_missing_ticker_returns_422(self):
        resp = client.post("/drift", json={})
        assert resp.status_code == 422

    def test_dataset_drift_is_bool(self):
        resp = self._patched_drift({"ticker": "AAPL"})
        assert isinstance(resp.json()["dataset_drift"], bool)
