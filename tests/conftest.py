"""
Shared pytest fixtures for the trading system test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the stock_dashboard package importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent / "stock_dashboard"))


# ---------------------------------------------------------------------------
# Raw OHLCV fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_ohlcv() -> pd.DataFrame:
    """Realistic synthetic OHLCV DataFrame with a Ticker column."""
    rng = np.random.default_rng(0)
    n = 300
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    close = 100.0 * (1 + rng.normal(0, 0.01, size=n)).cumprod()
    high = close * (1 + rng.uniform(0, 0.01, size=n))
    low = close * (1 - rng.uniform(0, 0.01, size=n))
    open_ = close * (1 + rng.normal(0, 0.005, size=n))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
        "Ticker": "TEST",
        "Return": pd.Series(close).pct_change().values,
        "Log_Return": np.log(pd.Series(close) / pd.Series(close).shift(1)).values,
    })
    df.loc[0, ["Return", "Log_Return"]] = np.nan
    return df.dropna(subset=["Return"]).reset_index(drop=True)


@pytest.fixture
def raw_ohlcv_no_ticker(raw_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Same data without the Ticker column (single-ticker mode)."""
    return raw_ohlcv.drop(columns=["Ticker"])


# ---------------------------------------------------------------------------
# Processed DataFrame fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def processed_df(raw_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """DataFrame after running through the full data pipeline."""
    from data_pipeline import PipelineConfig, process_stock_data

    # We can't easily mock yfinance here — instead build a minimal processed
    # DataFrame manually using the pipeline transformers directly.
    from data_pipeline import (
        DataQualityChecker,
        FinalNaNHandler,
        MissingDataHandler,
        TechnicalIndicators,
        PipelineConfig,
    )

    config = PipelineConfig(
        sma_periods=[10, 20],
        volatility_windows=[10],
        return_lags=[1, 2],
        min_periods=50,
    )

    df = raw_ohlcv.copy()
    df = DataQualityChecker(min_periods=50).fit_transform(df)
    df = TechnicalIndicators(config).fit_transform(df)
    df = MissingDataHandler().fit_transform(df)
    df = FinalNaNHandler().fit_transform(df)
    return df


# ---------------------------------------------------------------------------
# ML-ready DataFrame fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def ml_df(processed_df: pd.DataFrame) -> pd.DataFrame:
    """processed_df with a Target column appended."""
    df = processed_df.copy()
    future_ret = df["Close"].pct_change(5).shift(-5)
    df["Target"] = (future_ret > 0.01).astype(int)
    df["Future_Return"] = future_ret
    return df.dropna(subset=["Target"]).reset_index(drop=True)
