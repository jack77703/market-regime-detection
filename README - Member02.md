# Member 2 — Spectral Clustering Pipeline
## Advanced Market Regime Detection | S&P 500 (2000–2026)

---

## Overview

The goal is to identify Bull, Bear, and Sideways market regimes in S&P 500 daily data using Spectral Clustering — a network-based unsupervised learning method that outperforms the K-Means baseline (Track A) by capturing the complex, non-spherical geometry of financial market data.

This notebook is **Track B** in the group's A/B comparison pipeline.

---

## Project Pipeline Context

| Member | Track | Task |
|--------|-------|------|
| Member 1 | Track A | Data acquisition, feature engineering, K-Means baseline |
| **Member 2** | **Track B** | **Spectral Clustering — this notebook** |
| Member 3 | Both | Hidden Markov Model + rolling-window out-of-sample loop |
| Member 4 | Both | Backtesting, Sharpe Ratio comparison, final verdict |

The same Master Dataset flows through all four members, ensuring a fair one-to-one A/B comparison.

---

## File Structure

```
├── Member2_Spectral_Clustering.ipynb           # Main notebook (this file)
├── sp500_master_dataset (1).csv                # Input — produced by Member 1
└── sp500_master_with_both_K-means_Spectral_clusters.csv   # Output — passed to Member 3
```

---

## Dependencies

The notebook runs on **Python 3.8+** and requires the following libraries:

```
numpy
pandas
matplotlib
scikit-learn
```

Install all dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## How to Run

### Option 1 — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `Member2_Spectral_Clustering.ipynb` via **File → Upload Notebook**
3. Upload `sp500_master_dataset (1).csv` using the Colab file upload cell already present in **Step 1** of the notebook
4. Run all cells in order via **Runtime → Run All**
5. The output CSV (`sp500_master_with_both_K-means_Spectral_clusters.csv`) will be saved in the Colab working directory — download it for Member 3

> **Note:** Spectral Clustering on ~6,500 days takes approximately 30–90 seconds. This is expected behaviour due to the eigendecomposition step.

### Option 2 — Local Jupyter Notebook

1. Clone or download this repository
2. Place `sp500_master_dataset (1).csv` in the same folder as the notebook
3. Launch Jupyter:
   ```bash
   jupyter notebook Member2_Spectral_Clustering.ipynb
   ```
4. Comment out the `files.upload()` line in **Step 1** (that line is Colab-specific) and ensure `CSV_PATH` points to the correct local filename
5. Run all cells in order

---

## Notebook Structure

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

---

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

---

### Part 3 — Export for Member 3

- Saves the enriched DataFrame as `sp500_master_with_both_K-means_Spectral_clusters.csv`
- This file contains all original columns plus: `Spectral_Raw`, `Spectral_Synced`, `Spectral_Regime`, `KMeans_Synced`, `KMeans_Regime`
- Includes a rolling-window loop template showing Member 3 exactly how to call both functions inside their HMM iteration

---

## Input File

| File | Source | Description |
|------|--------|-------------|
| `sp500_master_dataset (1).csv` | Member 1 | Daily S&P 500 OHLCV data (2000–2026) with engineered features and K-Means labels |

**Required columns in the input file:**

- `Date` — index column (datetime)
- `Close` or `Adj Close` — S&P 500 closing price
- `Rolling_Return_21` — 21-day rolling mean log return (raw)
- `Rolling_Volatility_21` — 21-day rolling standard deviation of log return (raw)
- `Rolling_Return_21_scaled` — StandardScaler-normalised version
- `Rolling_Volatility_21_scaled` — StandardScaler-normalised version
- `KMeans_Label` — raw K-Means cluster labels from Member 1 (optional but needed for A/B comparison plots)

---

## Output File

| File | Destination | Description |
|------|-------------|-------------|
| `sp500_master_with_both_K-means_Spectral_clusters.csv` | Member 3 | All original columns + new cluster label columns |

**New columns added by this notebook:**

| Column | Description |
|--------|-------------|
| `Spectral_Raw` | Raw integer labels from SpectralClustering (arbitrary, pre-alignment) |
| `Spectral_Synced` | Aligned labels: 0=Sideways, 1=Bull, 2=Bear |
| `Spectral_Regime` | Human-readable regime name string |
| `KMeans_Synced` | Aligned K-Means labels using the same convention |
| `KMeans_Regime` | Human-readable K-Means regime name string |

---

## Key Design Decisions

- **`affinity='nearest_neighbors'` over `'rbf'`** — The RBF kernel builds a fully dense similarity matrix where the large Sideways cloud dominates and drowns out the smaller Bear cluster signal. The k-NN graph is sparse and only connects genuinely similar days, preserving local cluster structure.
- **`n_neighbors=35`** — Validated on the full 26-year dataset. Higher values produce smoother boundaries at the cost of speed.
- **Label alignment by return rank** — Purely data-driven and economically grounded. Does not rely on hardcoded mappings that would break across different rolling windows.
- **Raw features for statistics** — All per-regime statistics are computed on unscaled values so the numbers carry direct economic meaning.

---

## For Member 3 — Quick Reference

```python
# Step 1: Get raw spectral labels for a training window
df_train['Spectral_Raw'] = run_spectral_clustering(X_train)

# Step 2: Align to stable convention (0=Sideways, 1=Bull, 2=Bear)
df_train['Spectral_Synced'] = align_regime_labels(
    df_slice      = df_train,
    raw_label_col = 'Spectral_Raw',
    ret_col       = 'Rolling_Return_21'
)

# Step 3: Feed into HMM as observation sequence
obs_sequence = df_train['Spectral_Synced'].values
```

The same `align_regime_labels()` function works identically on K-Means labels (`KMeans_Label`) — ensuring both Track A and Track B use the same label convention for a valid HMM comparison.

---

## Authors

**Member 2** — Advanced Spatial Clustering (Track B)
MS Financial Engineering — Advanced Market Regime Detection Project
