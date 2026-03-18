"""
Stock data preprocessing pipeline.

Fetches OHLCV data from Yahoo Finance, runs it through a sklearn-compatible
Pipeline (quality check → technical indicators → missing data → outlier
capping → feature selection → scaling), and returns a clean DataFrame ready
for model training and backtesting.

Price columns (Close, Open, High, Low) are NEVER scaled — they must remain
in dollar terms for the backtest engine.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    enable_technical: bool = True
    sma_periods: List[int] = field(default_factory=lambda: [5, 20, 50])
    volatility_windows: List[int] = field(default_factory=lambda: [5, 20])
    return_lags: List[int] = field(default_factory=lambda: [1, 2, 5])
    max_missing_ratio: float = 0.2
    outlier_threshold: float = 4.0
    min_periods: int = 50
    correlation_threshold: float = 0.95


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _fetch_cached(ticker: str, period: str) -> Optional[pd.DataFrame]:
    """Module-level cached fetch so the cache survives across DataFetcher instances."""
    try:
        symbol = "^GSPC" if ticker.upper() == "SPX" else ticker
        data = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if data.empty:
            logger.warning("No data returned for %s", ticker)
            return None
        data = data.reset_index()
        data["Ticker"] = ticker.upper().replace("^", "")
        data["Return"] = data["Close"].pct_change()
        data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
        return data.dropna(subset=["Close"])
    except Exception as exc:
        logger.error("Error fetching %s: %s", ticker, exc)
        return None


class DataFetcher:
    """Thin wrapper around the module-level cached fetch function."""

    def fetch_stock_data(self, ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
        logger.info("Fetching %s  period=%s", ticker, period)
        return _fetch_cached(ticker, period)

    def fetch_multiple(self, tickers: List[str], period: str = "2y") -> pd.DataFrame:
        frames = [self.fetch_stock_data(t, period) for t in tickers]
        frames = [f for f in frames if f is not None]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(["Ticker", "Date"])


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

class DataQualityChecker(BaseEstimator, TransformerMixin):
    """Drop duplicates, enforce minimum row count, sort by date."""

    def __init__(self, min_periods: int = 50):
        self.min_periods = min_periods

    def fit(self, X: pd.DataFrame, y=None) -> "DataQualityChecker":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if X.empty:
            raise ValueError("Empty DataFrame passed to DataQualityChecker")

        df = X.dropna(subset=["Date", "Close"])

        # Deduplicate — handle both single-ticker (no Ticker col) and multi-ticker
        if "Ticker" in df.columns:
            df = df.drop_duplicates(["Ticker", "Date"], keep="first")
            df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        else:
            df = df.drop_duplicates(["Date"], keep="first")
            df = df.sort_values("Date").reset_index(drop=True)

        if len(df) < self.min_periods:
            raise ValueError(
                f"Insufficient data after quality check: {len(df)} rows < "
                f"minimum {self.min_periods}"
            )
        return df


class TechnicalIndicators(BaseEstimator, TransformerMixin):
    """Vectorised technical indicator computation."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def fit(self, X: pd.DataFrame, y=None) -> "TechnicalIndicators":
        return self

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute indicators for a single-ticker DataFrame (no groupby needed)."""
        close = df["Close"]
        ret = df["Return"]

        # SMAs
        for p in self.config.sma_periods:
            df[f"SMA_{p}"] = close.rolling(p, min_periods=1).mean()
            df[f"Price_SMA_{p}_Ratio"] = close / df[f"SMA_{p}"]

        # Lagged returns
        for lag in self.config.return_lags:
            df[f"Return_Lag_{lag}"] = ret.shift(lag)

        # Rolling volatility
        for win in self.config.volatility_windows:
            df[f"Volatility_{win}"] = ret.rolling(win, min_periods=1).std()

        # Momentum
        for p in [5, 10]:
            df[f"Price_Momentum_{p}"] = close.pct_change(p)

        # RSI (14-period)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # Bollinger Bands
        sma20 = close.rolling(20, min_periods=1).mean()
        std20 = close.rolling(20, min_periods=1).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        df["BB_Width"] = (bb_upper - bb_lower) / (sma20 + 1e-10)
        df["BB_Position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # Volume
        if "Volume" in df.columns:
            vol_ma = df["Volume"].rolling(20, min_periods=1).mean()
            df["Volume_Ratio"] = df["Volume"] / (vol_ma + 1e-10)

        # Calendar features
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["Month"] = df["Date"].dt.month

        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if X.empty or not self.config.enable_technical:
            return X

        if "Ticker" in X.columns:
            return (
                X.groupby("Ticker", group_keys=False)
                .apply(self._apply, include_groups=False)
                .reset_index(drop=True)
            )
        return self._apply(X.copy())


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Cap extreme return / momentum values using MAD-based z-scores."""

    def __init__(self, threshold: float = 4.0):
        self.threshold = threshold
        self.bounds: Dict[str, tuple] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "OutlierHandler":
        self.bounds = {}
        for col in X.columns:
            if "Return" in col or "Momentum" in col:
                vals = X[col].dropna()
                if len(vals) > 10:
                    med = vals.median()
                    mad = np.median(np.abs(vals - med))
                    if mad > 0:
                        self.bounds[col] = (med, mad)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col, (med, mad) in self.bounds.items():
            if col not in df.columns:
                continue
            z = 0.6745 * (df[col] - med) / mad
            mask = np.abs(z) > self.threshold
            if mask.any():
                lower = med - self.threshold * mad / 0.6745
                upper = med + self.threshold * mad / 0.6745
                df.loc[mask, col] = df.loc[mask, col].clip(lower, upper)
        return df


class MissingDataHandler(BaseEstimator, TransformerMixin):
    """Forward/backward fill, then drop columns that still exceed missing threshold."""

    _PROTECTED = {"Date", "Ticker", "Close", "Open", "High", "Low", "Volume", "Return"}

    def __init__(self, max_missing_ratio: float = 0.2):
        self.max_missing_ratio = max_missing_ratio
        self.drop_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "MissingDataHandler":
        high_missing = X.columns[X.isnull().mean() > self.max_missing_ratio]
        self.drop_cols = [c for c in high_missing if c not in self._PROTECTED]
        if self.drop_cols:
            logger.info("Dropping %d high-missing columns: %s", len(self.drop_cols), self.drop_cols)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.drop(columns=self.drop_cols, errors="ignore").copy()
        numeric = df.select_dtypes(include=[np.number]).columns
        df[numeric] = df[numeric].ffill(limit=5).bfill(limit=5)
        critical = [c for c in ("Close", "Return") if c in df.columns]
        if critical:
            df = df.dropna(subset=critical)
        return df


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Drop highly correlated feature columns (keeps essential price/meta cols)."""

    _PROTECTED = {"Date", "Ticker", "Close", "Open", "High", "Low", "Volume", "Return"}

    def __init__(self, correlation_threshold: float = 0.95):
        self.correlation_threshold = correlation_threshold
        self.keep_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureSelector":
        numeric = X.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric if c not in self._PROTECTED]

        if len(feature_cols) <= 1:
            self.keep_cols = list(X.columns)
            return self

        corr = X[feature_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = {
            col
            for col in upper.columns
            if (upper[col] > self.correlation_threshold).any()
        }
        self.keep_cols = [c for c in X.columns if c not in to_drop]
        if to_drop:
            logger.info("Dropping %d correlated features", len(to_drop))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.keep_cols] if self.keep_cols else X


class DataScaler(BaseEstimator, TransformerMixin):
    """
    RobustScaler applied only to derived features.

    Price columns (Close, Open, High, Low) and raw Volume are intentionally
    excluded so the backtest engine can use real dollar values.
    """

    _EXCLUDE = {"Date", "Ticker", "DayOfWeek", "Month", "Close", "Open", "High", "Low", "Volume"}

    def __init__(self, scale_by_group: bool = True):
        self.scale_by_group = scale_by_group
        self.scalers: Dict[str, RobustScaler] = {}
        self.cols_to_scale: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "DataScaler":
        numeric = X.select_dtypes(include=[np.number]).columns
        self.cols_to_scale = [c for c in numeric if c not in self._EXCLUDE]

        if not self.cols_to_scale:
            logger.warning("No columns available to scale")
            return self

        if self.scale_by_group and "Ticker" in X.columns:
            for ticker, grp in X.groupby("Ticker"):
                if len(grp) > 10:
                    self.scalers[ticker] = RobustScaler().fit(grp[self.cols_to_scale])
        else:
            self.scalers["global"] = RobustScaler().fit(X[self.cols_to_scale])

        logger.info("Scaling %d feature columns (prices excluded)", len(self.cols_to_scale))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.cols_to_scale:
            return X
        df = X.copy()
        if self.scale_by_group and "Ticker" in X.columns:
            for ticker, scaler in self.scalers.items():
                mask = df["Ticker"] == ticker
                if mask.any():
                    df.loc[mask, self.cols_to_scale] = scaler.transform(
                        df.loc[mask, self.cols_to_scale]
                    )
        elif "global" in self.scalers:
            df[self.cols_to_scale] = self.scalers["global"].transform(df[self.cols_to_scale])
        return df


class FinalNaNHandler(BaseEstimator, TransformerMixin):
    """Fill residual NaN in feature columns with 0; drop rows where prices are still NaN."""

    _PRICE_COLS = {"Close", "Open", "High", "Low"}

    def fit(self, X: pd.DataFrame, y=None) -> "FinalNaNHandler":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        fillable = [c for c in df.select_dtypes(include=[np.number]).columns if c not in self._PRICE_COLS]
        df[fillable] = df[fillable].fillna(0)
        price_cols = [c for c in self._PRICE_COLS if c in df.columns]
        if price_cols:
            before = len(df)
            df = df.dropna(subset=price_cols)
            dropped = before - len(df)
            if dropped:
                logger.warning("Dropped %d rows with missing prices", dropped)
        return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_pipeline(config: Optional[PipelineConfig] = None) -> Pipeline:
    config = config or PipelineConfig()
    return Pipeline([
        ("quality_check", DataQualityChecker(min_periods=config.min_periods)),
        ("technical", TechnicalIndicators(config)),
        ("missing_data", MissingDataHandler(max_missing_ratio=config.max_missing_ratio)),
        ("outliers", OutlierHandler(threshold=config.outlier_threshold)),
        ("feature_selection", FeatureSelector(correlation_threshold=config.correlation_threshold)),
        ("scaling", DataScaler(scale_by_group=True)),
        ("final_nan", FinalNaNHandler()),
    ])


def process_stock_data(
    tickers: Union[str, List[str]],
    period: str = "2y",
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """
    Fetch and process stock data for one or more tickers.

    Returns a dict with keys:
        success (bool), data (DataFrame | None), error (str | None), summary (dict | None)
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    fetcher = DataFetcher()
    if len(tickers) == 1:
        raw = fetcher.fetch_stock_data(tickers[0], period)
    else:
        raw = fetcher.fetch_multiple(tickers, period)

    if raw is None or raw.empty:
        return {"success": False, "data": None, "error": "No data retrieved", "summary": None}

    try:
        pipe = create_pipeline(config)
        processed = pipe.fit_transform(raw)

        if processed["Close"].min() <= 0:
            return {
                "success": False,
                "data": None,
                "error": "Invalid (zero/negative) prices after processing",
                "summary": None,
            }

        summary = {
            "tickers": tickers,
            "period": period,
            "raw_shape": raw.shape,
            "processed_shape": processed.shape,
            "features": [c for c in processed.columns if c not in ("Date", "Ticker")],
            "date_range": (processed["Date"].min(), processed["Date"].max()),
            "missing_data": int(processed.isnull().sum().sum()),
            "price_range": (processed["Close"].min(), processed["Close"].max()),
        }

        return {"success": True, "data": processed, "error": None, "summary": summary, "pipeline": pipe}

    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
        return {"success": False, "data": None, "error": str(exc), "summary": None}
