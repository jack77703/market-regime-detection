# market_regime/config.py
"""
Central configuration for the Market Regime Detection project.
Edit values here to change behaviour across all modules without
touching the core logic.
"""

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
TICKER: str = "SPY"
START_DATE: str = "2005-01-01"
END_DATE: str = "2026-04-01"   # Use None to fetch up to today

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
VOLATILITY_WINDOW: int = 21       # rolling window for annualised volatility
SMA_WINDOW: int = 200             # long-term trend SMA
VOLUME_MA_WINDOW: int = 21        # rolling window for volume surge normalisation
ANNUALISATION_FACTOR: int = 252   # trading days per year

# ---------------------------------------------------------------------------
# GMM Model
# ---------------------------------------------------------------------------
N_REGIMES: int = 3          # number of hidden regimes
COVARIANCE_TYPE: str = "full"
RANDOM_STATE: int = 42
N_INIT: int = 10            # number of initialisations (higher = more stable)

# AIC/BIC evaluation range
AIC_BIC_MIN_REGIMES: int = 1
AIC_BIC_MAX_REGIMES: int = 7

# Regime labels (assigned by sorted volatility: low → bull, mid → choppy, high → bear)
REGIME_LABELS: dict = {
    "bull": "Bull (Low Vol)",
    "choppy": "Choppy (Medium Vol)",
    "bear": "Bear (High Vol)",
}

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
RISK_FREE_RATE: float = 0.0   # annualised; 0.0 = ignore for Sharpe calculation
