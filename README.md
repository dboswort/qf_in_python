# Quantitative Finance in Python
Python implementations of core quantitative-finance models (mean-variance optimisation, CAPM, futures and bond pricing) created as part of the exercises of the Udemy course "Quantitative Finance & Algorithmic Trading in Python" (see [here](https://www.udemy.com/course/quantitative-finance-algorithmic-trading-in-python/?couponCode=MT260902G1B)).

## Overview

This repo contains self-contained Python scripts that implement:

- Classical portfolio theory (Markowitz mean-variance optimisation, CAPM)
- Derivatives pricing (Black-Scholes, Monte Carlo option pricing)
- Fixed-income and interest-rate models (bond pricing, Vasicek model)
- Risk measurement (Value-at-Risk)
- Stochastic-process simulation (Wiener process)
- Core building blocks (present value, normal-distribution exercises)

## Repository structure

Key modules:

- `markowitz_model.py` – Mean-variance portfolio optimisation, efficient frontier, find tangency portfolio.
- `capm.py` – Beta estimation, security market line, expected returns under CAPM.
- `black_scholes.py` – European option pricing and Greeks under the Black-Scholes model.
- `monte_carlo_option_pricing.py` – Monte Carlo simulation for option pricing (path simulation, payoff averaging).
- `value_at_risk.py` – Historical and/or parametric VaR calculations for portfolios.
- `vasicek_model.py` – Zero-coupon bond pricing under the Vasicek interest-rate model.
- `bonds.py` – Basic bond pricing, yield, duration, and convexity calculations.
- `pricing_futures_contract.py` – Fair futures price via cost-of-carry.
- `present_value.py` – Present-value and related TVM utilities.
- `wiener_process.py` – Simulation of standard Brownian motion / Wiener paths.
- `get_price_returns.py` – Utilities for retrieving price time series and calculating log returns.

Supporting files:

- `requirements.txt` – Python dependencies.
- `.gitignore` – Standard Python / notebook ignores.
- `README.md` – This file.

## Requirements

Python 3.9+ is recommended. Install dependencies via:

```bash
pip install -r requirements.txt
