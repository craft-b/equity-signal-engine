"""
FIXED Stock Data Preprocessing Pipeline
Key fix: Don't scale price columns (Close, Open, High, Low)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from typing import List, Union, Dict, Any, Optional
import logging
import yfinance as yf
from functools import lru_cache
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    enable_technical: bool = True
    sma_periods: List[int] = None
    volatility_windows: List[int] = None
    return_lags: List[int] = None
    max_missing_ratio: float = 0.2
    outlier_threshold: float = 4.0
    min_periods: int = 50
    correlation_threshold: float = 0.95

    def __post_init__(self):
        self.sma_periods = self.sma_periods or [5, 20, 50]
        self.volatility_windows = self.volatility_windows or [5, 20]
        self.return_lags = self.return_lags or [1, 2, 5]


class DataFetcher:
    """Fetch stock data with caching."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DataFetcher")

    @lru_cache(maxsize=32)
    def fetch_stock_data(self, ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
        try:
            self.logger.info(f"Fetching data for {ticker}, period: {period}")
            ticker = "^GSPC" if ticker.upper() == "SPX" else ticker
            data = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if data.empty:
                self.logger.warning(f"No data for {ticker}")
                return None
            data = data.reset_index()
            data["Ticker"] = ticker.upper().replace("^", "")
            data["Return"] = data["Close"].pct_change()
            data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
            return data.dropna(subset=["Close"])
        except Exception as e:
            self.logger.error(f"Error fetching {ticker}: {e}")
            return None

    def fetch_multiple(self, tickers: List[str], period: str = "2y") -> pd.DataFrame:
        frames = [self.fetch_stock_data(t, period) for t in tickers]
        frames = [f for f in frames if f is not None]
        return pd.concat(frames, ignore_index=True).sort_values(["Ticker", "Date"]) if frames else pd.DataFrame()


class DataQualityChecker(BaseEstimator, TransformerMixin):
    """Basic data quality checks."""

    def __init__(self, min_periods: int = 50):
        self.min_periods = min_periods

    def fit(self, X, y=None): 
        return self

    def transform(self, X):
        if X.empty:
            raise ValueError("Empty DataFrame")
        df = X.dropna(subset=["Date", "Close"]).drop_duplicates(["Ticker", "Date"], keep="first")
        if len(df) < self.min_periods:
            raise ValueError(f"Insufficient data: {len(df)} < {self.min_periods}")
        return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


class TechnicalIndicators(BaseEstimator, TransformerMixin):
    """Vectorized technical indicators."""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

    def fit(self, X, y=None): 
        return self

    def transform(self, X):
        if X.empty or not self.config.enable_technical:
            return X
        df = X.copy()
        
        # Handle grouping
        if "Ticker" in df.columns:
            groups = df.groupby("Ticker", group_keys=False)
        else:
            groups = [(None, df)]

        # SMA
        for period in self.config.sma_periods:
            if "Ticker" in df.columns:
                df[f"SMA_{period}"] = groups["Close"].rolling(period, min_periods=1).mean().reset_index(level=0, drop=True)
            else:
                df[f"SMA_{period}"] = df["Close"].rolling(period, min_periods=1).mean()

        # Lagged returns
        for lag in self.config.return_lags:
            if "Ticker" in df.columns:
                df[f"Return_Lag_{lag}"] = groups["Return"].shift(lag).reset_index(level=0, drop=True)
            else:
                df[f"Return_Lag_{lag}"] = df["Return"].shift(lag)

        # Rolling volatility
        for win in self.config.volatility_windows:
            if "Ticker" in df.columns:
                df[f"Volatility_{win}"] = groups["Return"].rolling(win, min_periods=1).std().reset_index(level=0, drop=True)
            else:
                df[f"Volatility_{win}"] = df["Return"].rolling(win, min_periods=1).std()

        # Price vs SMA ratios
        for period in self.config.sma_periods:
            sma_col = f"SMA_{period}"
            if sma_col in df.columns:
                df[f"Price_SMA_{period}_Ratio"] = df["Close"] / df[sma_col]

        # Price momentum
        for p in [5, 10]:
            if "Ticker" in df.columns:
                df[f"Price_Momentum_{p}"] = groups["Close"].pct_change(p).reset_index(level=0, drop=True)
            else:
                df[f"Price_Momentum_{p}"] = df["Close"].pct_change(p)

        # Volume features
        if "Volume" in df.columns:
            if "Ticker" in df.columns:
                df["Volume_SMA_20"] = groups["Volume"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
            else:
                df["Volume_SMA_20"] = df["Volume"].rolling(20, min_periods=1).mean()
            df["Volume_Ratio"] = df["Volume"] / (df["Volume_SMA_20"] + 1e-10)

        # Time features
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["Month"] = df["Date"].dt.month
        
        return df


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Cap extreme returns conservatively."""

    def __init__(self, threshold: float = 4.0):
        self.threshold = threshold
        self.bounds = {}

    def fit(self, X, y=None):
        for col in X.columns:
            if "Return" in col or "Momentum" in col:
                vals = X[col].dropna()
                if not vals.empty and len(vals) > 10:
                    med = vals.median()
                    mad = np.median(np.abs(vals - med))
                    if mad > 0:
                        self.bounds[col] = (med, mad)
        return self

    def transform(self, X):
        df = X.copy()
        for col, (med, mad) in self.bounds.items():
            if col in df.columns:
                z = 0.6745 * (df[col] - med) / mad
                mask = np.abs(z) > self.threshold
                if mask.any():
                    lower = med - self.threshold * mad / 0.6745
                    upper = med + self.threshold * mad / 0.6745
                    df.loc[mask, col] = np.clip(df.loc[mask, col], lower, upper)
        return df


class MissingDataHandler(BaseEstimator, TransformerMixin):
    """Fill missing data with ffill/bfill and drop columns exceeding max_missing_ratio."""

    def __init__(self, max_missing_ratio: float = 0.2):
        self.max_missing_ratio = max_missing_ratio
        self.drop_cols = []

    def fit(self, X, y=None):
        # CRITICAL: Essential columns that should NEVER be dropped
        protected_cols = {"Date", "Ticker", "Close", "Open", "High", "Low", "Volume", "Return"}
        
        # Find columns with too many missing values, excluding protected ones
        high_missing = X.columns[X.isnull().mean() > self.max_missing_ratio]
        self.drop_cols = [col for col in high_missing if col not in protected_cols]
        
        if self.drop_cols:
            logger.info(f"Dropping {len(self.drop_cols)} columns with >{self.max_missing_ratio*100}% missing: {self.drop_cols}")
        
        return self

    def transform(self, X):
        df = X.drop(columns=self.drop_cols, errors='ignore').copy()
        
        # Fill numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(method="ffill", limit=5).fillna(method="bfill", limit=5)
        
        # Only drop rows if Close or Return are still missing after filling
        critical_cols = [c for c in ["Close", "Return"] if c in df.columns]
        if critical_cols:
            df = df.dropna(subset=critical_cols)
        
        return df


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Correlation-based feature selection."""

    def __init__(self, correlation_threshold: float = 0.95):
        self.correlation_threshold = correlation_threshold
        self.keep_cols = []
        self.dropped_features = []

    def fit(self, X, y=None):
        # CRITICAL: Protect essential columns from being dropped
        protected_cols = {"Date", "Ticker", "Close", "Open", "High", "Low", "Volume", "Return"}
        
        numeric = X.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric if c not in protected_cols]
        
        if len(feature_cols) <= 1:
            self.keep_cols = list(X.columns)
            return self
        
        # Only check correlation among feature columns
        corr_matrix = X[feature_cols].corr().abs()
        to_drop = set()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        for col in upper.columns:
            highly_corr = upper.index[upper[col] > self.correlation_threshold].tolist()
            if highly_corr:
                to_drop.update(highly_corr)
        
        self.dropped_features = list(to_drop)
        self.keep_cols = [c for c in X.columns if c not in to_drop]
        
        if self.dropped_features:
            logger.info(f"Dropping {len(self.dropped_features)} highly correlated features: {self.dropped_features[:5]}...")
        
        return self

    def transform(self, X):
        return X[self.keep_cols] if self.keep_cols else X


class DataScaler(BaseEstimator, TransformerMixin):
    """
    Scale numeric features per ticker or globally.
    
    CRITICAL FIX: NEVER scale price columns (Close, Open, High, Low)
    These must remain in dollar values for backtesting.
    """

    def __init__(self, scale_by_group=True):
        self.scale_by_group = scale_by_group
        self.scalers = {}
        self.cols_to_scale = []

    def fit(self, X, y=None):
        # CRITICAL: Exclude price columns from scaling
        exclude = {"Date", "Ticker", "DayOfWeek", "Month", 
                  "Close", "Open", "High", "Low",  # ← THE FIX
                  "Volume"}  # Volume can be huge, better to use ratios
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.cols_to_scale = [c for c in numeric_cols if c not in exclude]
        
        if not self.cols_to_scale:
            logger.warning("No columns to scale")
            return self
        
        logger.info(f"Scaling {len(self.cols_to_scale)} features (price columns excluded)")
        
        if self.scale_by_group and "Ticker" in X.columns:
            for ticker, grp in X.groupby("Ticker"):
                if len(grp) > 10:  # Need minimum samples
                    self.scalers[ticker] = RobustScaler().fit(grp[self.cols_to_scale])
        else:
            self.scalers["global"] = RobustScaler().fit(X[self.cols_to_scale])
        
        return self

    def transform(self, X):
        df = X.copy()
        
        if not self.cols_to_scale:
            return df
        
        if self.scale_by_group and "Ticker" in X.columns:
            for ticker, scaler in self.scalers.items():
                mask = df["Ticker"] == ticker
                if mask.any() and ticker in df["Ticker"].values:
                    df.loc[mask, self.cols_to_scale] = scaler.transform(df.loc[mask, self.cols_to_scale])
        else:
            df[self.cols_to_scale] = self.scalers["global"].transform(df[self.cols_to_scale])
        
        return df


class FinalNaNHandler(BaseEstimator, TransformerMixin):
    """Fill any remaining NaNs with 0 (for features only, not prices)."""

    def fit(self, X, y=None): 
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # CRITICAL: Don't fill NaN in price columns with 0
        protected = {"Close", "Open", "High", "Low"}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        fillable_cols = [c for c in numeric_cols if c not in protected]
        
        if fillable_cols:
            df[fillable_cols] = df[fillable_cols].fillna(0)
        
        # If price columns still have NaN, drop those rows
        price_cols = [c for c in protected if c in df.columns]
        if price_cols:
            before = len(df)
            df = df.dropna(subset=price_cols)
            if len(df) < before:
                logger.warning(f"Dropped {before - len(df)} rows with missing prices")
        
        return df


def create_pipeline(config: PipelineConfig = None) -> Pipeline:
    """Create the processing pipeline with fixed scaler."""
    config = config or PipelineConfig()
    return Pipeline([
        ("quality_check", DataQualityChecker(min_periods=config.min_periods)),
        ("technical", TechnicalIndicators(config)),
        ("missing_data", MissingDataHandler(max_missing_ratio=config.max_missing_ratio)),
        ("outliers", OutlierHandler(threshold=config.outlier_threshold)),
        ("feature_selection", FeatureSelector(correlation_threshold=config.correlation_threshold)),
        ("scaling", DataScaler(scale_by_group=True)),  # ← Fixed to not scale prices
        ("final_nan", FinalNaNHandler()),
    ])


def process_stock_data(tickers: Union[str, List[str]], period: str = "2y", config: PipelineConfig = None) -> Dict[str, Any]:
    """Main processing function."""
    tickers = [tickers] if isinstance(tickers, str) else tickers
    fetcher = DataFetcher()
    
    raw_data = fetcher.fetch_multiple(tickers, period) if len(tickers) > 1 else fetcher.fetch_stock_data(tickers[0], period)
    
    if raw_data is None or raw_data.empty:
        return {"success": False, "data": None, "error": "No data retrieved", "summary": None}

    try:
        pipeline = create_pipeline(config)
        processed_data = pipeline.fit_transform(raw_data)
        
        # Validation
        if processed_data['Close'].min() <= 0:
            return {"success": False, "data": None, "error": "Invalid prices after processing", "summary": None}
        
        summary = {
            "tickers": tickers,
            "period": period,
            "raw_shape": raw_data.shape,
            "processed_shape": processed_data.shape,
            "features": [c for c in processed_data.columns if c not in ["Date", "Ticker"]],
            "date_range": (processed_data["Date"].min(), processed_data["Date"].max()),
            "missing_data": processed_data.isnull().sum().sum(),
            "price_range": (processed_data["Close"].min(), processed_data["Close"].max()),
        }

        return {"success": True, "data": processed_data, "error": None, "summary": summary, "pipeline": pipeline}
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return {"success": False, "data": None, "error": str(e), "summary": None}