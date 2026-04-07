# market_regime/model.py
"""
GMM-based market regime detection.

The model fits a Gaussian Mixture Model on the scaled feature matrix and then
maps the raw numeric cluster IDs to interpretable labels (bull / choppy / bear)
by sorting clusters on annualised volatility.

Public API
----------
fit_regimes(df) -> tuple[pd.DataFrame, GaussianMixture, StandardScaler]
    Fits the GMM, appends 'Regime' and 'Regime_Label' columns to df,
    and returns the augmented DataFrame plus fitted model objects.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import COVARIANCE_TYPE, N_INIT, N_REGIMES, RANDOM_STATE, REGIME_LABELS
from .features import FEATURE_COLS


def fit_regimes(
    df: pd.DataFrame,
    n_regimes: int = N_REGIMES,
) -> tuple[pd.DataFrame, GaussianMixture, StandardScaler]:
    """Fit GMM and label each day with its market regime.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns listed in ``features.FEATURE_COLS``.
    n_regimes : int
        Number of latent regimes (default from config).

    Returns
    -------
    df : pd.DataFrame
        Input DataFrame with two new columns:
        - ``Regime``       : raw integer cluster ID (0-based)
        - ``Regime_Label`` : human-readable string ("Bull", "Choppy", "Bear", …)
    gmm : GaussianMixture
        The fitted sklearn GMM object.
    scaler : StandardScaler
        The fitted scaler (needed for AIC/BIC evaluation on the same scale).
    """
    features = df[FEATURE_COLS].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    print(f"  Fitting GMM with {n_regimes} regimes ...")
    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type=COVARIANCE_TYPE,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )
    df = df.copy()
    df["Regime"] = gmm.fit_predict(scaled)

    # --- Map numeric IDs to semantic labels ---
    # Sort regimes by mean annualised volatility: lowest → bull, highest → bear
    regime_vol = (
        df.groupby("Regime")["Volatility"].mean().sort_values()
    )

    label_keys = list(REGIME_LABELS.keys())   # ["bull", "choppy", "bear"]
    id_to_label: dict[int, str] = {
        regime_id: label_keys[rank]
        for rank, regime_id in enumerate(regime_vol.index)
    }
    df["Regime_Label"] = df["Regime"].map(id_to_label)

    # Expose the numeric IDs as module-level convenience attributes
    bull_id = regime_vol.index[0]
    choppy_id = regime_vol.index[1] if n_regimes >= 2 else None
    bear_id = regime_vol.index[-1]

    print(
        f"  Regime mapping: "
        + ", ".join(
            f"cluster {k} → {v}" for k, v in id_to_label.items()
        )
    )
    print(
        f"  Bull ID={bull_id}, Choppy ID={choppy_id}, Bear ID={bear_id}"
    )

    return df, gmm, scaler, {
        "bull": bull_id,
        "choppy": choppy_id,
        "bear": bear_id,
        "id_to_label": id_to_label,
    }
