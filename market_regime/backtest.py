# market_regime/backtest.py
"""
Regime-switching backtest engine and performance metrics.

Strategy
--------
- Long SPY when previous day's regime is Bull or Choppy.
- Move to Cash (position = 0) when previous day's regime is Bear.
- Uses a 1-day lag on the signal to avoid look-ahead bias.

Public API
----------
run_backtest(df, regime_ids)  -> pd.DataFrame   adds signal/position/return cols
performance_metrics(df)       -> pd.DataFrame   annualised metrics table
"""

import numpy as np
import pandas as pd

from .config import ANNUALISATION_FACTOR, RISK_FREE_RATE


def run_backtest(
    df: pd.DataFrame,
    regime_ids: dict,
) -> pd.DataFrame:
    """Apply regime-switching rules and compute cumulative returns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Returns' and 'Regime' columns.
    regime_ids : dict
        Mapping as returned by ``model.fit_regimes`` — needs at least 'bear'.

    Returns
    -------
    pd.DataFrame
        Input DataFrame plus:
        - Signal              : lagged regime ID
        - Position            : 1 (long) or 0 (cash)
        - Strategy_Returns    : position-weighted log returns
        - Cum_Market_Return   : cumulative market performance
        - Cum_Strategy_Return : cumulative strategy performance
    """
    df = df.copy()
    bear_id = regime_ids["bear"]

    # Shift regime by 1 day — trade on yesterday's signal
    df["Signal"] = df["Regime"].shift(1)

    # Long everything except Bear regime days
    df["Position"] = np.where(df["Signal"] == bear_id, 0, 1)
    df["Strategy_Returns"] = df["Position"] * df["Returns"]

    # Cumulative growth (log returns → exp for total growth)
    df["Cum_Market_Return"] = np.exp(df["Returns"].cumsum())
    df["Cum_Strategy_Return"] = np.exp(df["Strategy_Returns"].cumsum())

    in_market_pct = df["Position"].mean() * 100
    print(f"  Backtest complete. In-market {in_market_pct:.1f}% of days.")
    return df


def _max_drawdown(cum_returns: pd.Series) -> float:
    """Maximum percentage drawdown from peak."""
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    return float(drawdown.min())


def performance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and print a side-by-side performance comparison table.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``run_backtest`` — must contain 'Returns',
        'Strategy_Returns', 'Cum_Market_Return', 'Cum_Strategy_Return'.

    Returns
    -------
    pd.DataFrame
        Rows = metric names; columns = ['Buy & Hold SPY', 'Regime Strategy'].
    """
    days = len(df)

    market_ann_ret  = df["Cum_Market_Return"].iloc[-1] ** (ANNUALISATION_FACTOR / days) - 1
    strat_ann_ret   = df["Cum_Strategy_Return"].iloc[-1] ** (ANNUALISATION_FACTOR / days) - 1

    market_ann_vol  = df["Returns"].std() * np.sqrt(ANNUALISATION_FACTOR)
    strat_ann_vol   = df["Strategy_Returns"].std() * np.sqrt(ANNUALISATION_FACTOR)

    market_max_dd   = _max_drawdown(df["Cum_Market_Return"])
    strat_max_dd    = _max_drawdown(df["Cum_Strategy_Return"])

    market_sharpe   = (market_ann_ret - RISK_FREE_RATE) / market_ann_vol
    strat_sharpe    = (strat_ann_ret  - RISK_FREE_RATE) / strat_ann_vol

    metrics = pd.DataFrame(
        {
            "Buy & Hold SPY": [
                f"{market_ann_ret:.2%}",
                f"{market_ann_vol:.2%}",
                f"{market_max_dd:.2%}",
                f"{market_sharpe:.2f}",
            ],
            "Regime Strategy": [
                f"{strat_ann_ret:.2%}",
                f"{strat_ann_vol:.2%}",
                f"{strat_max_dd:.2%}",
                f"{strat_sharpe:.2f}",
            ],
        },
        index=[
            "Annualised Return",
            "Annualised Volatility",
            "Maximum Drawdown",
            "Sharpe Ratio",
        ],
    )

    print("\n--- Backtest Performance Metrics ---")
    print(metrics.to_string())
    return metrics
