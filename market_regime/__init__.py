# market_regime/__init__.py
"""
market_regime
=============
A modular Python package for GMM-based market regime detection and backtesting.

Modules
-------
config          : Central configuration (ticker, dates, hyperparameters)
data_loader     : Download price/volume data via yfinance
features        : Feature engineering (returns, volatility, trend, volume)
model           : GMM fitting + regime labelling
analysis        : Regime statistics, AIC/BIC model selection
visualization   : Matplotlib plotting helpers
backtest        : Regime-switching strategy and performance metrics
"""
