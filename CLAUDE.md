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
  - `strategies.jl` -- includes strike_selection + iron_condor + short_strangle
  - `strategies/strike_selection.jl` -- `StrikeSelectionContext` struct, strike selection primitives + selector factories (`sigma_selector`, `delta_selector`, `delta_condor_selector`, `constrained_delta_selector`)
  - `strategies/iron_condor.jl` -- `IronCondorStrategy`
  - `strategies/short_strangle.jl` -- `ShortStrangleStrategy` (naked short put + short call, no wings) + `delta_strangle_selector(put_delta, call_delta)`. Selectors return 2-tuple `(sp_K, sc_K)`.

- **ML module**
  - `ml/features.jl` -- `Feature` and `CandidateFeature` abstract types, callable feature structs (ATMImpliedVol, DeltaSkew, RiskReversal, Butterfly, TermSlope, ATMSpread, DeltaSpread, TotalVolume, PutCallVolumeRatio, HourOfDay, DayOfWeek, ShortPutDelta, ShortCallDelta, EntryCredit, MaxLoss, CreditToMaxLoss), default feature sets
  - `ml/model.jl` -- `create_scoring_model` (Flux MLP), `score_candidates`
  - `ml/training.jl` -- `generate_training_data` (uses `each_entry`), `train_scoring_model!`, `roi_utility`, `pnl_utility`
  - `ml/selectors.jl` -- `ScoredCandidateSelector` (enumerate + score candidates), `DirectDeltaSelector` (predict optimal delta), `MLSizer` (ML-modulated trade sizing)
  - `ml/glmnet.jl` -- `GLMNetModel`, `train_ridge!`, `train_glmnet_classifier!` (ridge/lasso/elastic net via GLMNet.jl, drop-in replacement for Flux models)

- **Visualization**
  - `viz.jl` -- `CondorSpec`, `plot_smile_with_condors(surface, expiry, condor_specs; rate, div_yield, atm_window, title)`. Two-panel figure: put/call smile annotated with Black-76 deltas (top) + per-condor strike-axis structure diagram (bottom). Uses `delta_condor_selector` to pick legs.

### Key Patterns

- **Strategy interface**: Implement `ScheduledStrategy` with `entry_schedule()` and `entry_positions()`. The engine calls `backtest_strategy(strategy, source::BacktestDataSource)` returning `BacktestResult`. A convenience `backtest_strategy(strategy, surfaces, spots)` wraps dicts in `DictDataSource`.
- **Data source protocol**: `BacktestDataSource` is the abstract type with methods: `available_timestamps(source)`, `get_surface(source, ts)`, `get_settlement_spot(source, ts)`. `DictDataSource` wraps pre-loaded dicts. `ParquetDataSource` loads lazily from parquet with caching.
- **Strike selectors**: Callable `f(ctx::StrikeSelectionContext) -> (sp_K, sc_K, lp_K, lc_K) | nothing`. `StrikeSelectionContext` has fields `surface`, `expiry`, `history`. Helpers `_ctx_tau(ctx)` and `_ctx_recs(ctx)` derive tau and records. Selector factories: `sigma_selector`, `delta_selector`, `delta_condor_selector`, `constrained_delta_selector`. `ScoredCandidateSelector` is the candidate-scoring ML selector. `DirectDeltaSelector` predicts optimal delta.
- **each_entry**: Shared iteration function in `engine.jl` that resolves timestamps → surfaces → expiries → settlement. Used by both `backtest_strategy` and `generate_training_data`.
- **IronCondorStrategy**: Takes `(schedule, expiry_interval, strike_selector; sizer=FixedSize(1.0), debug)`. The `sizer` is a callable `f(ctx) -> quantity`. Use `MLSizer(model, means, stds; policy=binary_sizing())` for ML-modulated sizing. Rate/div_yield live in the selector, not the strategy.
- **Pricing convention**: Prices in `OptionRecord` are fractions of spot (Deribit convention). Multiply by `surface.spot` for USD. The backtest uses bid for sells, ask for buys.
- **ROI evaluation**: `performance_metrics(result::BacktestResult)` auto-computes condor max loss margin. Also available as `performance_metrics(positions, pnls; margin_by_key=...)` for custom margin.
- **ML model interface**: Any callable `model(X::Matrix{Float32}) → (1, N)` works with `MLSizer`. Two backends: Flux `Chain` (neural nets) and `GLMNetModel` (ridge/lasso/elastic net). Training: `train_model!`/`train_classifier!` for Flux, `train_ridge!`/`train_glmnet_classifier!` for GLMNet. All return `(model, means, stds, history)`.

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

- **`scripts/sizing_filter.jl`** -- Configurable experiment runner. Replaces 9 prior scripts by parameterizing symbols, feature sets, and ML variants. Config block at top; supports `Sizing` (binary/linear/sigmoid), `Classifier` (loss prediction + skip), and `DeltaRegression` variants. Shared infrastructure in `scripts/lib/experiment.jl`.
- **`scripts/cross_symbol_filter.jl`** -- Cross-symbol training experiment. Trains on SPY+QQQ+IWM+SPXW, tests on SPY. Compares regressor vs classifier, detailed tail risk analysis (CVaR, loss severity buckets, filter forensics).
- **`scripts/strike_selector.jl`** -- Separate: `ScoredCandidateSelector` pipeline (candidate enumeration + scoring), fundamentally different training data generation.

## Polygon IV inversion (rate / div_yield)

`to_option_record(::PolygonBar, spot; rate, div_yield)` inverts mark_iv with the proper forward `F = spot * exp((rate - div_yield) * T)` and discount `D = exp(-rT)`. Defaults to `0.0/0.0` for backward compat. `read_polygon_option_records` and `load_surfaces_and_spots` accept `rate` and `div_yield` kwargs; pass your scenario's rate (e.g., `0.045`) and dividend yield (e.g., `0.013` for SPY) to remove the put-call IV bias from F=spot inversion. Deribit ingestion is unaffected.

## Known Technical Debt

- **`src/backtest/metrics.jl`**: `performance_metrics()` has two near-identical branches for margin computation
- **`src/surface.jl` `_iv_from_price`** (used by `bid_iv`/`ask_iv`): inverts with `F=spot, r=0` regardless of `rate`/`div_yield`. After the Polygon `to_option_record` rate/q fix, `record.mark_iv` uses the proper forward but `bid_vol`/`ask_vol` on `VolPoint` still use the legacy convention.

## Conventions

- Julia 1.10+
- `Float64` for domain values
- `missing` for absent optional data (bid/ask not available)
- `nothing` for "computation failed / skip this entry"
- `DAYS_PER_YEAR = 365.25` for time-to-expiry
- Deribit expiry normalized to 08:00 UTC; Polygon expiry to 16:00 ET (DST-aware)
- **Commit messages**: Do not mention Claude, AI, or any assistant tooling in commit messages or PR descriptions
