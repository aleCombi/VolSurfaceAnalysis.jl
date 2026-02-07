# CLAUDE.md -- VolSurfaceAnalysis

## Project Overview

Julia package for options volatility surface analysis, strategy backtesting, and ML-based strike selection. Supports Polygon.io (equities: SPY, QQQ, etc.) and Deribit (crypto: BTC, ETH). The core pipeline: load options data from parquet files, build volatility surfaces, run option-selling strategies through a backtest engine, and evaluate with ROI/Sharpe/Sortino metrics. An ML module learns to select optimal strikes using neural networks trained on surface features and path signatures.

## IMPORTANT: Keep This File Updated

After every task that requires user feedback -- especially tasks that change architecture, add features, modify the data pipeline, or restructure code -- update this file to reflect the current state. This ensures future Claude sessions start with accurate context.

## Quick Commands

```bash
# Run tests
julia --project=. -e "using Pkg; Pkg.test()"

# Run iron condor backtest (canonical strategy script)
julia --project=scripts scripts/backtest_polygon_iron_condor.jl

# Run short strangle backtest
julia --project=scripts scripts/backtest_polygon_short_strangle.jl

# Run ML training pipeline
julia --project=scripts scripts/ml_strike_selector.jl

# Run ML evaluation vs baseline
julia --project=scripts scripts/evaluate_condor_prediction_vs_baseline.jl
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

- **Backtest engine**
  - `backtest/portfolio.jl` -- `Position`, `open_position`, `settle`, `entry_cost`
  - `backtest/engine.jl` -- `Strategy`, `ScheduledStrategy`, `backtest_strategy`
  - `backtest/metrics.jl` -- `performance_metrics`, `condor_trade_table`, `condor_max_loss_by_key`
  - `backtest/plots.jl` -- plot helpers

- **Strategies**
  - `strategies.jl` -- includes helpers + strategy files
  - `strategies/helpers.jl` -- strike selection: `_sigma_*`, `_delta_*`, `_condor_wings_by_objective`
  - `strategies/iron_condor.jl` -- `IronCondorStrategy`
  - `strategies/short_strangle.jl` -- `ShortStrangleStrategy`

- **ML module**
  - `ml/features.jl` -- `extract_features`, `SurfaceFeatures`, path signatures via ChenSignatures
  - `ml/model.jl` -- `create_strike_model`, `create_scoring_model` (Flux MLP)
  - `ml/training.jl` -- training data generation, training loops, evaluation
  - `ml/strike_selector.jl` -- `MLStrikeSelector`, `MLCondorStrikeSelector`, `MLCondorScoreSelector`

### Key Patterns

- **Strategy interface**: Implement `ScheduledStrategy` with `entry_schedule()` and `entry_positions()`. The engine calls `backtest_strategy(strategy, surfaces, spots)` returning `(positions, pnls)`.
- **Strike selectors**: Callable `f(ctx) -> strikes_tuple | nothing`. The `ctx` named tuple: `(surface, expiry, tau, recs, put_strikes, call_strikes)`.
- **Pricing convention**: Prices in `OptionRecord` are fractions of spot (Deribit convention). Multiply by `surface.spot` for USD. The backtest uses bid for sells, ask for buys.
- **ROI evaluation**: `performance_metrics(positions, pnls; margin_by_key=condor_max_loss_by_key(positions))` computes ROI using per-condor max loss as margin.

### Backtest Engine (Backbone)

The engine is the backbone of the library. All P&L computation should flow through it.

**Current design (what works):**
- `Strategy` -> `ScheduledStrategy`: clean two-level hierarchy with `entry_schedule()` + `entry_positions()`
- Two `backtest_strategy` variants: one settles via surfaces at expiry, one via spot dict
- Pure functional positions: `Position` is immutable, settlement is a function
- Bid/ask realism: `open_position` prices at bid (shorts) and ask (longs)
- Composable selectors: `strike_selector` callable plugs ML, delta-based, or ROI-optimized logic into any strategy

**Current limitations:**
1. No intermediate decisions -- no callback between entry and expiry (no "close at 50% profit")
2. No `exit_positions()` -- exit is always expiry, baked into `backtest_strategy`
3. No portfolio-level awareness -- strategy can't see existing open positions
4. ML eval script bypasses the engine -- reimplements its own loop for oracle comparisons
5. Training labels bypass the engine -- `simulate_condor_pnl()` computes payoff directly, not via `Position`/`settle`

**Intended direction:**
- Phase 1: `simulate_strategy_pnl(selector, surfaces, spots)` thin wrapper -- lets ML training/eval reuse the engine. Oracle selectors become just another callable.
- Phase 2: `MonitoredStrategy` subtype -- called at intermediate timestamps for early exit, rolling, hedging
- Phase 3: Portfolio-level awareness -- strategies see open positions, capital allocation, exposure limits

**What NOT to do:** No transaction costs until live comparison needed. No tick-by-tick simulation. Don't break immutable Position design.

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

| Script | Uses backtest engine? | Purpose |
|--------|----------------------|---------|
| `backtest_polygon_iron_condor.jl` | Yes | Canonical condor backtest with ROI output |
| `backtest_polygon_short_strangle.jl` | Yes | Strangle backtest |
| `ml_strike_selector.jl` | Yes (for validation) | ML training + backtest evaluation |
| `evaluate_condor_prediction_vs_baseline.jl` | **No** (own loop) | Detailed ML vs baseline comparison |

Scripts use `scripts/Project.toml` via `Pkg.activate(@__DIR__)`.

## Known Technical Debt

### Verbosity / Duplication (~25-30% of src/ could be reduced)

1. **`src/ml/training.jl`** (1,494 lines) -- biggest offender:
   - `simulate_strangle_pnl()` vs `simulate_condor_pnl()` share ~90% of code (differ only by 2 vs 4 legs)
   - `generate_training_data()` vs `generate_condor_training_data()` are near-identical loops
   - `train_model!()` vs `train_scoring_model!()` share ~90% of training loop logic
   - Repeated null-checking chains appear 6+ times

2. **`src/strategies/helpers.jl`** (782 lines):
   - `_sigma_strangle_strikes()` vs `_sigma_condor_strikes()` differ only by 2 extra strikes
   - `_delta_strangle_strikes()` vs `_delta_strangle_strikes_asymmetric()` differ only by same-vs-different deltas
   - `_best_delta_strike()` loop pattern repeated 8+ times

3. **`src/ml/model.jl`**: `create_strike_model()` vs `create_scoring_model()` have identical layer construction

4. **`src/backtest/metrics.jl`**: `performance_metrics()` has two near-identical branches for margin computation

### ML Integration Gap

- `evaluate_condor_prediction_vs_baseline.jl` reimplements backtest logic inline (strike resolution, PnL computation, feature extraction) instead of using `backtest_strategy()`
- `training.jl` has its own `condor_metrics_from_strikes()` instead of using `Position`/`settle`
- **Long-term goal**: route all evaluation through the backtest engine

## Conventions

- Julia 1.10+
- `Float64` for domain values, `Float32` for ML tensors
- `missing` for absent optional data (bid/ask not available)
- `nothing` for "computation failed / skip this entry"
- `DAYS_PER_YEAR = 365.25` for time-to-expiry
- Deribit expiry normalized to 08:00 UTC; Polygon expiry to 16:00 ET (DST-aware)
