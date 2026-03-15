# CLAUDE.md -- VolSurfaceAnalysis

## Project Overview

Julia package for options volatility surface analysis and strategy backtesting. Supports Polygon.io (equities: SPY, QQQ, SPXW, etc.) and Deribit (crypto: BTC, ETH). The core pipeline: load options data from parquet files, build volatility surfaces, run option-selling strategies through a backtest engine, and evaluate with ROI/Sharpe/Sortino metrics.

## IMPORTANT: Keep This File Updated

After every task that requires user feedback -- especially tasks that change architecture, add features, modify the data pipeline, or restructure code -- update this file to reflect the current state. This ensures future Claude sessions start with accurate context.

## Quick Commands

```bash
# Run tests
julia --project=. -e "using Pkg; Pkg.test()"
```

## Architecture

### Module Structure (`src/VolSurfaceAnalysis.jl`)

- **Core types & pricing**
  - `data/option_records.jl` -- `OptionRecord`, `SpotPrice`, `Underlying`, `OptionType` (Call/Put)
  - `models.jl` -- Black-76: `black76_price`, `black76_delta/gamma/theta/vega`, `price_to_iv`
  - `surface.jl` -- `VolatilitySurface`, `VolPoint`, `TermStructure`, `build_surface`
  - `trades.jl` -- `Trade`, `payoff`, `price`, `find_vol`

- **Data layer**
  - `data/deribit.jl` -- `DeribitQuote`, parquet readers
  - `data/polygon.jl` -- `PolygonBar`, parquet readers (synthetic bid/ask from OHLC)
  - `data/local_store.jl` -- `LocalDataStore`, `DEFAULT_STORE`, path derivation
  - `data/duckdb.jl` -- DuckDB query helpers for parquet
  - `data/helpers.jl` -- shared data-loading helpers: `build_entry_timestamps`, `load_minute_spots`, `load_surfaces_and_spots`

- **Backtest engine**
  - `backtest/positions.jl` -- `Position`, `open_position`, `settle`, `entry_cost`
  - `backtest/data_source.jl` -- `BacktestDataSource` protocol, `DictDataSource`, `ParquetDataSource`
  - `backtest/engine.jl` -- `ScheduledStrategy`, `backtest_strategy` → `BacktestResult`, `each_entry` (shared timestamp iteration)
  - `backtest/metrics.jl` -- `performance_metrics`, `condor_trade_table`, `condor_max_loss_by_key`
  - `backtest/plots.jl` -- plot helpers

- **Strategies**
  - `strategies.jl` -- includes strike_selection + iron_condor
  - `strategies/strike_selection.jl` -- `StrikeSelectionContext` struct, strike selection primitives + selector factories (`sigma_selector`, `delta_selector`, `delta_condor_selector`, `constrained_delta_selector`)
  - `strategies/iron_condor.jl` -- `IronCondorStrategy`

- **ML module**
  - `ml/features.jl` -- `Feature` and `CandidateFeature` abstract types, callable feature structs (ATMImpliedVol, DeltaSkew, RiskReversal, Butterfly, TermSlope, ATMSpread, DeltaSpread, TotalVolume, PutCallVolumeRatio, HourOfDay, DayOfWeek, ShortPutDelta, ShortCallDelta, EntryCredit, MaxLoss, CreditToMaxLoss), default feature sets
  - `ml/model.jl` -- `create_scoring_model` (Flux MLP), `score_candidates`
  - `ml/training.jl` -- `generate_training_data` (uses `each_entry`), `train_scoring_model!`, `MLCondorSelector` (trained selector callable), `roi_utility`, `pnl_utility`

### Key Patterns

- **Strategy interface**: Implement `ScheduledStrategy` with `entry_schedule()` and `entry_positions()`. The engine calls `backtest_strategy(strategy, source::BacktestDataSource)` returning `BacktestResult`. A convenience `backtest_strategy(strategy, surfaces, spots)` wraps dicts in `DictDataSource`.
- **Data source protocol**: `BacktestDataSource` is the abstract type with methods: `available_timestamps(source)`, `get_surface(source, ts)`, `get_settlement_spot(source, ts)`. `DictDataSource` wraps pre-loaded dicts. `ParquetDataSource` loads lazily from parquet with caching.
- **Strike selectors**: Callable `f(ctx::StrikeSelectionContext) -> (sp_K, sc_K, lp_K, lc_K) | nothing`. `StrikeSelectionContext` has fields `surface`, `expiry`, `history`. Helpers `_ctx_tau(ctx)` and `_ctx_recs(ctx)` derive tau and records. Selector factories: `sigma_selector`, `delta_selector`, `delta_condor_selector`, `constrained_delta_selector`. `MLCondorSelector` is the trained ML selector.
- **each_entry**: Shared iteration function in `engine.jl` that resolves timestamps → surfaces → expiries → settlement. Used by both `backtest_strategy` and `generate_training_data`.
- **IronCondorStrategy**: Takes `(schedule, expiry_interval, strike_selector; quantity, debug)`. Rate/div_yield live in the selector, not the strategy.
- **Pricing convention**: Prices in `OptionRecord` are fractions of spot (Deribit convention). Multiply by `surface.spot` for USD. The backtest uses bid for sells, ask for buys.
- **ROI evaluation**: `performance_metrics(positions, pnls; margin_by_key=condor_max_loss_by_key(positions))` computes ROI using per-condor max loss as margin.

### Data Layout

Local data store root: see `DEFAULT_STORE` in `data/local_store.jl`.

```
root/
  massive_flatfiles/
    minute_aggs/date={yyyy-mm-dd}/underlying={SYM}/data.parquet   # Polygon options
    spot_1min/date={yyyy-mm-dd}/symbol={SYM}/data.parquet         # Polygon spot
  deribit_local/
    history/vols_{yyyymmdd}.parquet                                # Deribit daily
    delivery_prices/delivery_prices.parquet                        # Settlement
```

### Scripts

Scripts use `scripts/Project.toml` via `Pkg.activate(@__DIR__)`.

## Known Technical Debt

- **`src/backtest/metrics.jl`**: `performance_metrics()` has two near-identical branches for margin computation

## Conventions

- Julia 1.10+
- `Float64` for domain values
- `missing` for absent optional data (bid/ask not available)
- `nothing` for "computation failed / skip this entry"
- `DAYS_PER_YEAR = 365.25` for time-to-expiry
- Deribit expiry normalized to 08:00 UTC; Polygon expiry to 16:00 ET (DST-aware)
- **Commit messages**: Do not mention Claude, AI, or any assistant tooling in commit messages or PR descriptions
