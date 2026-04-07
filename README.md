# Market Regime Detection

A modular Python project that uses a **Gaussian Mixture Model (GMM)** to identify three hidden market regimes in SPY — *Bull*, *Choppy*, and *Bear* — and back-tests a regime-switching strategy against Buy & Hold.

---

## Project Structure

```
market_regime_project/
├── market_regime/          # Core package
│   ├── __init__.py
│   ├── config.py           # Central config (ticker, dates, hyperparams)
│   ├── data_loader.py      # Download SPY data via yfinance
│   ├── features.py         # Feature engineering (returns, vol, trend, volume)
│   ├── model.py            # GMM fitting + regime labelling
│   ├── analysis.py         # Regime stats + AIC/BIC model selection
│   ├── visualization.py    # All Matplotlib plots
│   └── backtest.py         # Regime-switching strategy + performance metrics
├── main.py                 # End-to-end pipeline entry point
├── requirements.txt
└── .github/
    └── workflows/
        └── ci.yml          # GitHub Actions: lint + import checks
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/market_regime_project.git
cd market_regime_project

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python main.py
```

Running `main.py` will:
1. Download SPY price + volume data from Yahoo Finance
2. Engineer four features: log returns, rolling volatility, SMA-200 trend distance, volume surge
3. Fit a 3-component GMM and assign regime labels
4. Print regime statistics and AIC/BIC scores
5. Back-test the regime-switching strategy
6. Save three charts: `regime_chart.png`, `aic_bic_chart.png`, `backtest_chart.png`

---

## Module Overview

| Module | Responsibility | Key function |
|---|---|---|
| `config.py` | All magic numbers in one place | — |
| `data_loader.py` | Download & clean raw OHLCV data | `load_spy_data()` |
| `features.py` | Compute model-ready features | `build_features()` |
| `model.py` | Fit GMM, map clusters to labels | `fit_regimes()` |
| `analysis.py` | Regime stats, AIC/BIC sweep | `regime_summary()`, `evaluate_n_regimes()` |
| `visualization.py` | Matplotlib chart helpers | `plot_regimes()`, `plot_aic_bic()`, `plot_backtest()` |
| `backtest.py` | Strategy simulation + Sharpe/drawdown | `run_backtest()`, `performance_metrics()` |

---

## Configuration

All tunable parameters live in `market_regime/config.py`:

```python
TICKER       = "SPY"          # change to any yfinance symbol
START_DATE   = "2005-01-01"
N_REGIMES    = 3              # number of hidden states
VOLATILITY_WINDOW = 21
SMA_WINDOW   = 200
RISK_FREE_RATE = 0.0          # for Sharpe calculation
```

---

## Contributing

1. Branch off `main` for your feature: `git checkout -b feature/my-improvement`
2. Keep changes scoped to the relevant module(s)
3. Open a Pull Request — CI will run lint checks automatically

---

## Requirements

- Python ≥ 3.10
- See `requirements.txt` for package versions
