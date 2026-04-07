# market_regime/analysis.py
"""
Statistical analysis of fitted regimes.

Public API
----------
regime_summary(df)          -> pd.DataFrame   regime statistics table
evaluate_n_regimes(df, scaler) -> pd.DataFrame  AIC / BIC for k = 1..7
optimal_n_regimes(eval_df)  -> dict            {'aic': int, 'bic': int}
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import (
    AIC_BIC_MAX_REGIMES,
    AIC_BIC_MIN_REGIMES,
    ANNUALISATION_FACTOR,
    COVARIANCE_TYPE,
    N_INIT,
    RANDOM_STATE,
)
from .features import FEATURE_COLS


def regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-regime descriptive statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Regime', 'Returns', 'Volatility', 'Trend_Dist',
        'Volume_Surge' columns (output of ``model.fit_regimes``).

    Returns
    -------
    pd.DataFrame
        Rows = regime IDs; columns = Mean_Ann_Return, Mean_Ann_Vol,
        Mean_Trend_Dist, Mean_Vol_Surge, Days.
    """
    stats = df.groupby("Regime").agg(
        Mean_Daily_Return=("Returns", "mean"),
        Days=("Returns", "count"),
        Mean_Ann_Vol=("Volatility", "mean"),
        Mean_Trend_Dist=("Trend_Dist", "mean"),
        Mean_Vol_Surge=("Volume_Surge", "mean"),
    )
    stats["Mean_Ann_Return"] = stats["Mean_Daily_Return"] * ANNUALISATION_FACTOR

    result = stats[
        ["Mean_Ann_Return", "Mean_Ann_Vol", "Mean_Trend_Dist", "Mean_Vol_Surge", "Days"]
    ]

    print("\n--- Regime Statistics ---")
    print(result.to_string())
    return result


def evaluate_n_regimes(
    df: pd.DataFrame,
    scaler: StandardScaler,
    min_k: int = AIC_BIC_MIN_REGIMES,
    max_k: int = AIC_BIC_MAX_REGIMES,
) -> pd.DataFrame:
    """Fit GMMs for k = min_k..max_k and collect AIC and BIC scores.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the feature columns (``FEATURE_COLS``).
    scaler : StandardScaler
        Already-fitted scaler from ``model.fit_regimes`` — ensures the
        evaluation uses identical scaling.
    min_k, max_k : int
        Range of components to evaluate (inclusive).

    Returns
    -------
    pd.DataFrame
        Columns: 'n_regimes', 'AIC', 'BIC'.
    """
    scaled = scaler.transform(df[FEATURE_COLS])

    print(f"Evaluating AIC/BIC for {min_k}–{max_k} regimes ...")
    records = []
    for n in range(min_k, max_k + 1):
        gmm = GaussianMixture(
            n_components=n,
            covariance_type=COVARIANCE_TYPE,
            random_state=RANDOM_STATE,
            n_init=N_INIT,
        )
        gmm.fit(scaled)
        records.append({"n_regimes": n, "AIC": gmm.aic(scaled), "BIC": gmm.bic(scaled)})
        print(f"  k={n}  AIC={records[-1]['AIC']:,.1f}  BIC={records[-1]['BIC']:,.1f}")

    return pd.DataFrame(records).set_index("n_regimes")


def optimal_n_regimes(eval_df: pd.DataFrame) -> dict:
    """Return the k that minimises AIC and BIC respectively.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Output of ``evaluate_n_regimes``.

    Returns
    -------
    dict with keys 'aic' and 'bic' (int values).
    """
    result = {
        "aic": int(eval_df["AIC"].idxmin()),
        "bic": int(eval_df["BIC"].idxmin()),
    }
    print(
        f"\nOptimal regimes → AIC suggests {result['aic']}, "
        f"BIC suggests {result['bic']}"
    )
    return result
