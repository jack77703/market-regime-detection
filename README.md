# Advanced Market Regime Detection
### S&P 500 · K-Means & Spectral Clustering · Hidden Markov Models · Walk-Forward Backtest

A four-phase ML project that detects Bull, Sideways, and Bear market regimes in the S&P 500 (2003–2026) using unsupervised clustering and Hidden Markov Models, then validates the signals through a rigorous walk-forward backtest.

---

## Project Structure

```
market_regime_project/
├── Phase1_Data_Engineering_Baseline.ipynb          # Phase 1: data pipeline & feature engineering
├── Phase 3 HMM/
│   ├── Notebooks/
│   │   ├── Phase3_HMM _code.ipynb                  # Phase 3: original HMM pipeline
│   │   └── Phase3_HMM _code（with 4 align methods).ipynb   # Phase 3: 4 alignment method comparison
│   └── Results (csv)/
│       ├── HMM_aligndef1_3years.csv                # Volatility-sorted, 3yr window
│       ├── HMM_aligndef1_5years.csv                # Volatility-sorted, 5yr window
│       ├── HMM_aligndef2_3years.csv                # Pseudo-Sharpe (ret/vol), 3yr window
│       ├── HMM_aligndef2_5years.csv                # Pseudo-Sharpe (ret/vol), 5yr window
│       ├── HMM_aligndef3_3years.csv                # Threshold (pos/neg return), 3yr window
│       ├── HMM_aligndef3_5years.csv                # Threshold (pos/neg return), 5yr window
│       ├── HMM_aligndef4_3years.csv                # Mean-return-sorted, 3yr window
│       └── HMM_aligndef4_5years.csv                # Mean-return-sorted, 5yr window
├── Phase4_Financial_Application_Backtest.ipynb     # Phase 4: backtest & financial analysis
├── sp500_master_dataset.csv                        # S&P 500 daily data 2000–2026
└── requirements.txt
```

---

## Methodology

### Phase 1 — Data Engineering & Baseline
- Loads daily S&P 500 OHLCV data (2000–2026)
- Engineers two core features: **21-day rolling log return** and **21-day rolling volatility**
- Establishes baseline regime statistics

### Phase 3 — HMM Rolling Prediction
A two-stage pipeline applied in a **walk-forward rolling window** (no look-ahead bias):

1. **Clustering** — K-Means or Spectral Clustering groups each trading day into 3 raw clusters based on rolling return and volatility
2. **Label Alignment** — one of 4 alignment methods maps raw cluster integers to consistent Bull/Sideways/Bear labels across windows
3. **HMM Smoothing** — `hmmlearn.CategoricalHMM` uses transition probabilities to convert noisy daily cluster labels into persistent regime states
4. **Prediction** — the final hidden state predicts the next 21-day regime window

**4 Alignment Methods:**

| Method | Logic |
|---|---|
| `aligndef1` | Sort clusters by rolling volatility (low vol → Bull) |
| `aligndef2` | Sort by pseudo-Sharpe ratio (return / volatility) |
| `aligndef3` | Threshold-based (positive return → Bull, negative → Bear) |
| `aligndef4` | Sort by mean return (highest → Bull) |

**2 Window Sizes:** 3-year (756 days) and 5-year (1260 days) training windows → **8 CSV outputs total**

### Phase 4 — Financial Application & Backtest
- Backtests all 8 configurations × 2 tracks (KMeans + Spectral) = **16 strategies**
- Signal: **Long** when regime = Bull Market, **Cash** otherwise
- Benchmark: Buy & Hold S&P 500
- Additional experiment: Binary (1/0/0) vs Graduated (1/0.5/0) vs Long/Short (1/0/−1) position sizing

---

## Key Results

**Best configuration: `aligndef3 / 3-year window / KMeans → HMM`**

| Strategy | Total Return | Sharpe | Max Drawdown |
|---|---|---|---|
| Best (def3_3yr_K) | +372% | **0.609** | **−23.7%** |
| Buy & Hold | +679% | 0.495 | −56.8% |

**Position sizing experiment (on best config):**

| Position Map | Sharpe | Max Drawdown |
|---|---|---|
| Binary (1 / 0 / 0) | **0.609** | **−23.7%** |
| Graduated (1 / 0.5 / 0) | 0.505 | −27.3% |
| Long/Short (1 / 0 / −1) | 0.246 | −44.5% |

The Long/Short strategy's collapse confirms that **Bear regimes capture volatility, not directional movement** — Cash is the correct response to a Bear signal, not Short.

---

## Setup

```bash
git clone https://github.com/jack77703/market-regime-detection.git
cd market-regime-detection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Open notebooks in order:
1. `Phase1_Data_Engineering_Baseline.ipynb`
2. `Phase 3 HMM/Notebooks/Phase3_HMM _code（with 4 align methods).ipynb`
3. `Phase4_Financial_Application_Backtest.ipynb`

---

## Requirements

- Python ≥ 3.10
- pandas, numpy, matplotlib
- scikit-learn (KMeans, SpectralClustering)
- hmmlearn (CategoricalHMM)
- See `requirements.txt` for full versions
