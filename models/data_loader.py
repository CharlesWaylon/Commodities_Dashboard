"""
Data loader for the predictive models package.

Fetches historical OHLCV data from Yahoo Finance and returns clean,
date-indexed DataFrames ready for feature engineering.

Note: this module does NOT import streamlit, so it can be run outside
the dashboard (from a script, notebook, or scheduler) without error.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional

from models.config import (
    MODELING_COMMODITIES,
    HISTORY_PERIOD,
    HISTORY_INTERVAL,
)


def load_price_matrix(
    commodities: Optional[dict] = None,
    period: str = HISTORY_PERIOD,
    interval: str = HISTORY_INTERVAL,
) -> pd.DataFrame:
    """
    Pull daily closing prices for a set of commodities.

    Parameters
    ----------
    commodities : dict, optional
        {display_name: ticker} mapping. Defaults to MODELING_COMMODITIES.
    period : str
        yfinance period string, e.g. '2y', '1y', '6mo'.
    interval : str
        Bar size — '1d' for daily, '1wk' for weekly, etc.

    Returns
    -------
    pd.DataFrame
        Shape (n_days, n_commodities). Columns are display names.
        NaN rows are forward-filled then dropped if still missing.
    """
    if commodities is None:
        commodities = MODELING_COMMODITIES

    tickers = list(commodities.values())
    display_names = list(commodities.keys())

    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
    )

    if raw.empty:
        raise RuntimeError("yfinance returned no data. Check tickers or network.")

    # yfinance returns MultiIndex columns when > 1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Rename ticker symbols → readable display names
    ticker_to_name = {v: k for k, v in commodities.items()}
    close = close.rename(columns=ticker_to_name)

    # Keep only columns we asked for (handles any failed tickers gracefully)
    close = close[[c for c in display_names if c in close.columns]]

    # Forward-fill up to 3 bars (handles market holidays), then drop remaining NaNs
    close = close.ffill(limit=3).dropna()
    close.index = pd.to_datetime(close.index)
    close.index.name = "Date"

    return close


def load_single(
    ticker: str,
    period: str = HISTORY_PERIOD,
    interval: str = HISTORY_INTERVAL,
) -> pd.DataFrame:
    """
    Pull full OHLCV history for a single ticker.

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume. DatetimeIndex.
    """
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)

    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def train_test_split_by_date(
    df: pd.DataFrame,
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a date-indexed DataFrame into train and test sets chronologically.

    We never shuffle time-series data — leaking future information into
    training is a common mistake that inflates model performance artificially.

    Returns
    -------
    (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]
