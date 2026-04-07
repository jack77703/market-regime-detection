#!/usr/bin/env python
# main.py
"""
End-to-end entry point for the Market Regime Detection project.

Run from the repo root:
    python main.py

Outputs
-------
- Console: regime statistics + backtest metrics
- Figures : regime_chart.png, aic_bic_chart.png, backtest_chart.png
"""

import matplotlib
matplotlib.use("Agg")   # headless / CI-safe backend

import matplotlib.pyplot as plt

from market_regime.data_loader import load_spy_data
from market_regime.features import build_features
from market_regime.model import fit_regimes
from market_regime.analysis import regime_summary, evaluate_n_regimes, optimal_n_regimes
from market_regime.visualization import plot_regimes, plot_aic_bic, plot_backtest
from market_regime.backtest import run_backtest, performance_metrics


def main() -> None:
    print("=" * 60)
    print("  Market Regime Detection Pipeline")
    print("=" * 60)

    # 1. Load data
    df = load_spy_data()

    # 2. Feature engineering
    df = build_features(df)

    # 3. Fit GMM and label regimes
    df, gmm, scaler, regime_ids = fit_regimes(df)

    # 4. Regime statistics
    stats = regime_summary(df)

    # 5. AIC / BIC model selection
    eval_df = evaluate_n_regimes(df, scaler)
    optimal = optimal_n_regimes(eval_df)

    # 6. Visualise regimes
    fig_regimes = plot_regimes(df, regime_ids)
    fig_regimes.savefig("regime_chart.png", dpi=150)
    print("  Saved → regime_chart.png")

    # 7. Visualise AIC/BIC
    fig_aic = plot_aic_bic(eval_df)
    fig_aic.savefig("aic_bic_chart.png", dpi=150)
    print("  Saved → aic_bic_chart.png")

    # 8. Backtest
    df = run_backtest(df, regime_ids)
    metrics = performance_metrics(df)

    # 9. Backtest chart
    fig_bt = plot_backtest(df, regime_ids)
    fig_bt.savefig("backtest_chart.png", dpi=150)
    print("  Saved → backtest_chart.png")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
