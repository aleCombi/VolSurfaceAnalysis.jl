# VolSurfaceAnalysis

[![Build Status](https://github.com/aleCombi/VolSurfaceAnalysis.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/aleCombi/VolSurfaceAnalysis.jl/actions/workflows/CI.yml?query=branch%3Amaster)

A Julia package for options volatility surface analysis and strategy backtesting. Supports **Polygon.io** (equities: SPY, QQQ, etc.) and **Deribit** (crypto: BTC, ETH).

## Features

- Volatility surface construction from options chains (ITM-preferred, volume tiebreaker)
- Black-76 pricing model with full Greeks (delta, gamma, theta, vega) and IV inversion
- Event-driven backtest engine for scheduled option strategies
- Iron condor strategy with pluggable strike selectors
- ROI-optimized wing selection for condors (maximize credit/max_loss ratio)

## Data Sources

### Polygon.io (Equities)

Options minute bars (OHLC) and spot prices. Since Polygon provides trade data rather than quotes, bid/ask are synthetically approximated from the high/low of each bar. IV is inferred from mark price via Black-76.

### Deribit (Crypto)

Options chain snapshots with native bid/ask quotes and exchange-computed mark IV. Data collected via the companion [DeribitVols](../DeribitVols/) project.

### Local Data Store

Both sources are stored as parquet files in a local data store (see `LocalDataStore` in the code):

```
root/
  massive_flatfiles/
    minute_aggs/date={yyyy-mm-dd}/underlying={SYM}/data.parquet   # Polygon options
    spot_1min/date={yyyy-mm-dd}/symbol={SYM}/data.parquet         # Polygon spot
  deribit_local/
    history/vols_{yyyymmdd}.parquet                                # Deribit daily
    delivery_prices/delivery_prices.parquet                        # Settlement prices
```

## Data Pipeline

```
Polygon.io API ──┐                                   ┌── Volatility surfaces
                 ├──▶ options-collector (Python) ──▶ Local Parquet Store ──▶ VolSurfaceAnalysis.jl
Deribit API ─────┘    (stores as parquet)             └── Strategy backtesting
```

## Quick Start

```bash
# Run tests
julia --project=. -e "using Pkg; Pkg.test()"
```

## Core Types

| Type | Description |
|------|-------------|
| `OptionRecord` | Unified options chain record (from Polygon or Deribit) |
| `VolPoint` | Point on vol surface: `(log_moneyness, tau, vol)` |
| `VolatilitySurface` | Complete surface at a timestamp with `VolPoint` grid |
| `TermStructure` | 1D slice of surface at fixed moneyness |
| `Trade` | Option trade specification (strike, expiry, type, direction) |
| `Position` | Open trade with entry price, spot, and timestamp |

## Backtest Engine

Strategies implement the `ScheduledStrategy` interface:

```julia
struct MyStrategy <: ScheduledStrategy
    schedule::Vector{DateTime}
    # ...
end

entry_schedule(s::MyStrategy) = s.schedule
entry_positions(s::MyStrategy, surface::VolatilitySurface) = [...]
```

Run a backtest and evaluate:

```julia
result = backtest_strategy(strategy, surfaces, settlement_spots)
metrics = performance_metrics(result.positions, result.pnl;
    margin_by_key=condor_max_loss_by_key(result.positions))

metrics.total_roi           # Total ROI on margin
metrics.annualized_roi_cagr # CAGR
metrics.sharpe              # Annualized Sharpe
metrics.sortino             # Annualized Sortino
metrics.win_rate            # Win rate
```

Built-in strategy: `IronCondorStrategy` with pluggable strike selectors (`sigma_selector`, `delta_selector`, `delta_condor_selector`).

## Project Structure

```
src/
  data/               # Data loading (option_records, deribit, polygon, local_store, duckdb)
  backtest/           # Engine, positions, metrics, plots
  strategies/         # Iron condor, strike selection
scripts/              # Runnable backtest scripts
test/                 # Unit tests
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/aleCombi/VolSurfaceAnalysis.jl")
```

## References

- [Deribit API Documentation](https://docs.deribit.com/)
- [Polygon.io Options API](https://polygon.io/docs/options)
- [DeribitVols Data Collector](../DeribitVols/) -- companion project for Deribit data collection
