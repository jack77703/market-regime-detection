# market_regime/visualization.py
"""
All plotting functions for the Market Regime project.

Public API
----------
plot_regimes(df, regime_ids)              -> matplotlib Figure
plot_aic_bic(eval_df)                     -> matplotlib Figure
plot_backtest(df, regime_ids)             -> matplotlib Figure
"""

import matplotlib.pyplot as plt
import matplotlib.figure
import pandas as pd

from .config import REGIME_LABELS


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
REGIME_COLOURS = {
    "bull":   ("green",  REGIME_LABELS["bull"]),
    "choppy": ("gold",   REGIME_LABELS["choppy"]),
    "bear":   ("red",    REGIME_LABELS["bear"]),
}


def plot_regimes(
    df: pd.DataFrame,
    regime_ids: dict,
) -> matplotlib.figure.Figure:
    """Plot SPY price coloured by detected regime.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Price' and 'Regime' columns.
    regime_ids : dict
        Mapping ``{'bull': int, 'choppy': int, 'bear': int}`` as returned
        by ``model.fit_regimes``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(df.index, df["Price"], color="black", label="SPY Price", linewidth=1)

    for label, (colour, display_name) in REGIME_COLOURS.items():
        regime_id = regime_ids.get(label)
        if regime_id is None:
            continue
        dates = df[df["Regime"] == regime_id].index
        ax.scatter(
            dates,
            df.loc[dates, "Price"],
            color=colour,
            label=display_name,
            s=10,
            alpha=0.6,
        )

    ax.set_yscale("log")
    ax.set_title(
        "SPY Market Regimes: 3-State GMM (Returns, Volatility, Trend, Volume)",
        fontsize=14,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("SPY Log Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_aic_bic(eval_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Plot AIC and BIC scores vs. number of regimes.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Output of ``analysis.evaluate_n_regimes`` — index is n_regimes,
        columns are 'AIC' and 'BIC'.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        eval_df.index,
        eval_df["AIC"],
        label="AIC (Akaike)",
        marker="o",
        color="blue",
        linewidth=2,
    )
    ax.plot(
        eval_df.index,
        eval_df["BIC"],
        label="BIC (Bayesian)",
        marker="s",
        color="orange",
        linewidth=2,
    )

    ax.set_title("Optimal Number of Market Regimes — AIC / BIC", fontsize=14)
    ax.set_xlabel("Number of Regimes (k)", fontsize=12)
    ax.set_ylabel("Information Criterion Score (lower = better)", fontsize=12)
    ax.set_xticks(eval_df.index)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_backtest(
    df: pd.DataFrame,
    regime_ids: dict,
) -> matplotlib.figure.Figure:
    """Plot cumulative equity curves with bear-regime shading.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Cum_Market_Return', 'Cum_Strategy_Return', 'Regime'.
    regime_ids : dict
        Mapping as returned by ``model.fit_regimes``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(
        df.index,
        df["Cum_Market_Return"],
        color="black",
        label="Buy & Hold SPY",
        alpha=0.7,
    )
    ax.plot(
        df.index,
        df["Cum_Strategy_Return"],
        color="blue",
        label="Regime-Switching Strategy",
        linewidth=2,
    )

    # Shade consecutive bear-regime days
    bear_id = regime_ids.get("bear")
    if bear_id is not None:
        bear_dates = df[df["Regime"] == bear_id].index
        for i in range(len(bear_dates) - 1):
            if (bear_dates[i + 1] - bear_dates[i]).days <= 3:
                ax.axvspan(bear_dates[i], bear_dates[i + 1], color="red", alpha=0.1, lw=0)

    ax.set_title("Backtest: Regime-Switching Strategy vs. Buy & Hold", fontsize=14)
    ax.set_ylabel("Cumulative Growth (1.0 = Starting Capital)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
