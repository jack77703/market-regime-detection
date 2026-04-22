# Advanced Market Regime Detection
### S&P 500 · K-Means & Spectral Clustering · Hidden Markov Models · Walk-Forward Backtest

A four-phase ML project that detects Bull, Sideways, and Bear market regimes in the S&P 500 (2003–2026) using unsupervised clustering and Hidden Markov Models, then validates the signals through a rigorous walk-forward backtest.

---

## Project Structure

```
market_regime_project/
├── Phase1_Data_Engineering_Baseline.ipynb
├── Phase 2 Member2_Spectral_Clustering.ipynb
    ├── Phase 2 sp500_master_with_both_K-means_Spectral_clusters
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

## Phase 2 Spectral Clustering

## File Structure

```
├── Member2_Spectral_Clustering.ipynb           # Main notebook (this file)
├── sp500_master_dataset .csv                # Input — produced by Member 1
└── sp500_master_with_both_K-means_Spectral_clusters.csv   # Output — passed to Member 3
```

The notebook is divided into three parts:

### Part 1 — Helper Functions

Defines two reusable functions that are also shared with Member 3's HMM pipeline.

**`run_spectral_clustering(X_scaled)`**
- Fits `sklearn.cluster.SpectralClustering` on a scaled 2D feature matrix
- Uses `affinity='nearest_neighbors'` with `n_neighbors=35`
- Returns raw integer labels (0, 1, or 2) — not yet economically meaningful
- Must always be followed by `align_regime_labels()`

**`align_regime_labels(df_slice, raw_label_col)`**
- Resolves the label-flipping problem that occurs in all unsupervised methods
- Uses a return-rank rule: lowest return → Bear (2), middle → Sideways (0), highest → Bull (1)
- Guarantees a stable output convention on every run and every rolling window

**Output label convention (guaranteed):**

| Value | Regime |
|-------|--------|
| `0` | Sideways / Correction |
| `1` | Bull Market (Low Volatility, Positive Returns) |
| `2` | Bear Market (High Volatility) |

### Part 2 — Full-Sample Analysis (2000–2026)

Applies both functions to the complete 26-year dataset and produces the following outputs:

- **Step 1** — Loads the Master Dataset CSV from Member 1
- **Step 2** — Extracts the scaled feature matrix (`Rolling_Return_21_scaled`, `Rolling_Volatility_21_scaled`)
- **Step 3** — Fits Spectral Clustering on all ~6,500 trading days
- **Step 4** — Aligns labels for both Track B (Spectral) and Track A (K-Means) to the same convention
- **Step 5** — Per-regime statistics table (mean return, std return, mean volatility, std volatility, count)
- **Step 6** — Four visualisations:
  - Side-by-side feature space: K-Means vs Spectral Clustering
  - Spectral Clustering standalone feature space
  - S&P 500 time series coloured by Spectral regime
  - S&P 500 time series coloured by K-Means regime

### Part 3 — Export for Member 3

- Saves the enriched DataFrame as `sp500_master_with_both_K-means_Spectral_clusters.csv`
- This file contains all original columns plus: `Spectral_Raw`, `Spectral_Synced`, `Spectral_Regime`, `KMeans_Synced`, `KMeans_Regime`
- Includes a rolling-window loop template showing Member 3 exactly how to call both functions inside their HMM iteration

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
