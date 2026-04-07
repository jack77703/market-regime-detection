# market_regime/data_loader.py
"""
Handles all data acquisition from Yahoo Finance via yfinance.

Public API
----------
load_spy_data(ticker, start, end) -> pd.DataFrame
    Returns a DataFrame with columns ['Price', 'Volume'] indexed by Date.
"""

import pandas as pd
import yfinance as yf

from .config import END_DATE, START_DATE, TICKER


def load_spy_data(
    ticker: str = TICKER,
    start: str = START_DATE,
    end: str | None = END_DATE,
) -> pd.DataFrame:
    """Download OHLCV data and return a clean Price + Volume DataFrame.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (default: "SPY").
    start : str
        Start date in "YYYY-MM-DD" format.
    end : str | None
        End date in "YYYY-MM-DD" format, or None for today.

    Returns
    -------
    pd.DataFrame
        Columns: ['Price', 'Volume'], DatetimeIndex named 'Date'.

    Raises
    ------
    ValueError
        If the downloaded DataFrame is empty (bad ticker / date range).
    """
    print(f"Downloading {ticker} data from {start} to {end or 'today'}...")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(
            f"No data returned for ticker='{ticker}' "
            f"between {start} and {end}. Check the ticker and date range."
        )

    # Handle potential MultiIndex columns produced by recent yfinance versions
    price: pd.Series = pd.DataFrame(raw["Close"]).iloc[:, 0]
    volume: pd.Series = pd.DataFrame(raw["Volume"]).iloc[:, 0]

    df = pd.DataFrame({"Price": price, "Volume": volume})
    df.index.name = "Date"

    print(f"  Downloaded {len(df):,} trading days.")
    return df
