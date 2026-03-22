"""
Pydantic request / response models for the Equity Signal Engine API.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    ticker: str = Field(..., description="Equity ticker symbol")
    period: str = Field(
        "1y",
        description="History window for signal computation (yfinance format: 1mo 3mo 6mo 1y 2y)",
    )
    enable_shorts: bool = Field(False, description="Allow short-sell signals")

    model_config = {"json_schema_extra": {"example": {"ticker": "AAPL", "period": "1y"}}}


class PredictResponse(BaseModel):
    ticker: str
    signal: int = Field(..., description="Trading signal: 1=BUY, -1=SELL, 0=HOLD")
    signal_label: str = Field(..., description="Human-readable signal: BUY | SELL | HOLD")
    signal_strength: float = Field(..., description="Normalised SMA spread (proxy for confidence)")
    regime: str = Field(..., description="Volatility regime at latest bar: Low | Medium | High")
    latest_price: float
    latest_date: str
    sma_10: float
    sma_50: float
    timestamp: str


# ---------------------------------------------------------------------------
# /drift
# ---------------------------------------------------------------------------

class DriftRequest(BaseModel):
    ticker: str = Field(..., description="Equity ticker symbol")
    period: str = Field(
        "2y",
        description="Total history to fetch; first 70% used as reference, last 30% as current",
    )

    model_config = {"json_schema_extra": {"example": {"ticker": "AAPL", "period": "2y"}}}


class DriftResponse(BaseModel):
    ticker: str
    dataset_drift: bool
    drift_share: float = Field(..., description="Fraction of features with detected drift")
    drifted_features: List[str]
    feature_drift: Dict[str, bool]
    n_reference: int
    n_current: int
    n_features_checked: int
    timestamp: str
    error: Optional[str] = None
