# market_regime/features.py
"""
Feature engineering: transforms raw Price + Volume data into model-ready features.

Features produced
-----------------
Returns       : log daily return
Volatility    : 21-day rolling annualised volatility
Trend_Dist    : % distance from 200-day SMA  (positive = above SMA)
Volume_Surge  : today's volume / 21-day rolling mean volume

Public API
----------
build_features(df) -> pd.DataFrame
    Adds feature columns to a copy of the input DataFrame and drops NaN rows.
"""

import numpy as np
import pandas as pd

from .config import (
    ANNUALISATION_FACTOR,
    SMA_WINDOW,
    VOLATILITY_WINDOW,
    VOLUME_MA_WINDOW,
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all model features and return a clean DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'Price' and 'Volume'.

    Returns
    -------
    pd.DataFrame
        Original columns plus 'Returns', 'Volatility', 'SMA_200',
        'Trend_Dist', 'Vol_SMA_21', 'Volume_Surge'. Rows with NaN
        (first ~200 trading days) are dropped.
    """
    df = df.copy()

    # --- 1. Log returns & rolling volatility ---
    df["Returns"] = np.log(df["Price"] / df["Price"].shift(1))
    df["Volatility"] = (
        df["Returns"].rolling(window=VOLATILITY_WINDOW).std()
        * np.sqrt(ANNUALISATION_FACTOR)
    )

    # --- 2. Trend: distance from long-term SMA ---
    df["SMA_200"] = df["Price"].rolling(window=SMA_WINDOW).mean()
    df["Trend_Dist"] = (df["Price"] / df["SMA_200"]) - 1

    # --- 3. Volume conviction ---
    df["Vol_SMA_21"] = df["Volume"].rolling(window=VOLUME_MA_WINDOW).mean()
    df["Volume_Surge"] = df["Volume"] / df["Vol_SMA_21"]

    # Drop rows where any feature is NaN (mainly the first SMA_WINDOW rows)
    df = df.dropna()

    print(
        f"  Feature engineering complete. "
        f"{len(df):,} rows kept after dropping warm-up NaNs."
    )
    return df


FEATURE_COLS: list[str] = ["Returns", "Volatility", "Trend_Dist", "Volume_Surge"]
"""Ordered list of column names fed into the GMM."""
