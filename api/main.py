"""
Equity Signal Engine — FastAPI prediction service.

Exposes the ML pipeline as a REST API with two core endpoints:

  POST /predict  — latest trading signal + volatility regime for a ticker
  POST /drift    — feature distribution drift report (reference vs current)

The service is stateless: it fetches and processes data on every request.
No model training on request — signals use the SMA crossover strategy,
which is always available and computationally cheap.

Deployment:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Make stock_dashboard importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "stock_dashboard"))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    DriftRequest,
    DriftResponse,
    PredictRequest,
    PredictResponse,
)
from backtest import BacktestConfig, _generate_signals
from data_pipeline import (
    DataFetcher,
    DataQualityChecker,
    FinalNaNHandler,
    MissingDataHandler,
    PipelineConfig,
    TechnicalIndicators,
)
from monitoring import detect_feature_drift
from regimes import classify_vol_regimes

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Equity Signal Engine API",
    description=(
        "ML-powered trading signal generation and feature drift monitoring. "
        "Signals are computed from SMA crossover with volatility regime context."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_fetcher = DataFetcher()
_pipeline_config = PipelineConfig()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_processed_df(ticker: str, period: str):
    """Fetch and process OHLCV data through the feature pipeline."""
    raw = _fetcher.fetch_stock_data(ticker, period)
    if raw is None or raw.empty:
        raise HTTPException(status_code=404, detail=f"No data found for ticker '{ticker}'")

    df = raw.copy()
    try:
        df = DataQualityChecker(min_periods=50).fit_transform(df)
        df = TechnicalIndicators(_pipeline_config).fit_transform(df)
        df = MissingDataHandler().fit_transform(df)
        df = FinalNaNHandler().fit_transform(df)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Pipeline processing failed: {exc}")

    if df is None or df.empty:
        raise HTTPException(status_code=422, detail=f"Pipeline produced no data for '{ticker}'")

    return df


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "Equity Signal Engine API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {"predict": "POST /predict", "drift": "POST /drift", "health": "GET /health"},
    }


@app.get("/health")
def health():
    """Liveness probe — returns 200 when the service is up."""
    return {"status": "ok", "timestamp": _utc_now()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Compute the latest trading signal and volatility regime for a ticker.

    Signal is derived from SMA-10 / SMA-50 crossover on processed OHLCV data.
    Signal strength is the normalised SMA spread: (SMA_10 - SMA_50) / SMA_50.
    Volatility regime uses a rolling 20-bar annualised vol percentile classifier.
    """
    df = _build_processed_df(req.ticker, req.period)

    config = BacktestConfig(enable_shorts=req.enable_shorts)
    signals = _generate_signals(df, model=None, feature_cols=None, config=config)

    df_reg, _ = classify_vol_regimes(df)
    latest = df_reg.iloc[-1]

    raw_signal = int(signals.iloc[-1])
    label_map = {1: "BUY", -1: "SELL", 0: "HOLD"}

    sma_10 = float(latest.get("SMA_10", np.nan))
    sma_50 = float(latest.get("SMA_50", np.nan))
    if not (np.isnan(sma_10) or np.isnan(sma_50) or sma_50 == 0):
        strength = float((sma_10 - sma_50) / sma_50)
    else:
        strength = 0.0

    date_val = latest.get("Date", "")
    latest_date = str(date_val.date()) if hasattr(date_val, "date") else str(date_val)[:10]

    return PredictResponse(
        ticker=req.ticker.upper(),
        signal=raw_signal,
        signal_label=label_map.get(raw_signal, "HOLD"),
        signal_strength=round(strength, 6),
        regime=str(latest.get("Vol_Regime", "Unknown")),
        latest_price=round(float(latest["Close"]), 4),
        latest_date=latest_date,
        sma_10=round(sma_10, 4) if not np.isnan(sma_10) else 0.0,
        sma_50=round(sma_50, 4) if not np.isnan(sma_50) else 0.0,
        timestamp=_utc_now(),
    )


@app.post("/drift", response_model=DriftResponse)
def drift(req: DriftRequest):
    """
    Detect feature distribution drift between reference and current windows.

    Fetches `period` of history for the ticker, splits 70/30 into
    reference (older) and current (recent) windows, then runs an
    Evidently DataDriftPreset across all numeric feature columns.
    """
    df = _build_processed_df(req.ticker, req.period)

    split = int(len(df) * 0.70)
    if split < 10 or (len(df) - split) < 10:
        raise HTTPException(
            status_code=422,
            detail="Insufficient data for drift analysis — try a longer period.",
        )

    reference_df = df.iloc[:split].reset_index(drop=True)
    current_df = df.iloc[split:].reset_index(drop=True)

    result = detect_feature_drift(reference_df, current_df)

    if "error" in result and result["error"]:
        return DriftResponse(
            ticker=req.ticker.upper(),
            dataset_drift=False,
            drift_share=0.0,
            drifted_features=[],
            feature_drift={},
            n_reference=len(reference_df),
            n_current=len(current_df),
            n_features_checked=0,
            timestamp=_utc_now(),
            error=result["error"],
        )

    return DriftResponse(
        ticker=req.ticker.upper(),
        dataset_drift=result["dataset_drift"],
        drift_share=result["drift_share"],
        drifted_features=result["drifted_features"],
        feature_drift=result["feature_drift"],
        n_reference=result["n_reference"],
        n_current=result["n_current"],
        n_features_checked=result["n_features_checked"],
        timestamp=_utc_now(),
    )
