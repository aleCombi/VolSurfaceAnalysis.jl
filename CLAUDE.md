# CLAUDE.md -- VolSurfaceAnalysis

## Project Overview

Julia package for options volatility surface analysis, strategy backtesting, and ML-based strike selection. Supports Polygon.io (equities: SPY, QQQ, SPXW, etc.) and Deribit (crypto: BTC, ETH). The core pipeline: load options data from parquet files, build volatility surfaces, run option-selling strategies through a backtest engine, and evaluate with ROI/Sharpe/Sortino metrics. An ML module learns to select optimal strikes using neural networks trained on surface features and path signatures.

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
  - `data/helpers.jl` -- shared data-loading helpers: `build_entry_timestamps`, `load_minute_spots`, `load_surfaces_and_spots`, `build_spot_history_dict`, `build_prev_surfaces_dict`

- **Backtest engine**
  - `backtest/portfolio.jl` -- `Position`, `open_position`, `settle`, `entry_cost`
  - `backtest/data_source.jl` -- `BacktestDataSource` protocol, `DictDataSource`, `ParquetDataSource`
  - `backtest/engine.jl` -- `Strategy`, `ScheduledStrategy`, `backtest_strategy`
  - `backtest/metrics.jl` -- `performance_metrics`, `condor_trade_table`, `condor_max_loss_by_key`
  - `backtest/plots.jl` -- plot helpers

- **Strategies**
  - `strategies.jl` -- includes helpers + strategy files
  - `strategies/helpers.jl` -- strike selection: `_sigma_*`, `_delta_*`, `_condor_wings_by_objective`
  - `strategies/iron_condor.jl` -- `IronCondorStrategy`
  - `strategies/short_strangle.jl` -- `ShortStrangleStrategy`

- **ML module**
  - `ml/features.jl` -- `extract_features`, `SurfaceFeatures`, path signatures via ChenSignatures; `prev_surface` kwarg for surface dynamics; `LOGSIG_DEAD_INDICES`, `pruned_logsig_dim()` for dead feature pruning
  - `ml/model.jl` -- `create_strike_model`, `create_scoring_model` (Flux MLP)
  - `ml/training.jl` -- training data generation, training loops, evaluation
  - `ml/strike_selector.jl` -- `MLStrikeSelector`, `MLCondorStrikeSelector`, `MLCondorScoreSelector`

### Key Patterns

- **Strategy interface**: Implement `ScheduledStrategy` with `entry_schedule()` and `entry_positions()`. The engine calls `backtest_strategy(strategy, source::BacktestDataSource)` returning `(positions, pnls)`. A backward-compatible `backtest_strategy(strategy, surfaces, spots)` convenience method wraps dicts in `DictDataSource`.
- **Data source protocol**: `BacktestDataSource` is the abstract type with three methods: `available_timestamps(source)`, `get_surface(source, ts)`, `get_settlement_spot(source, ts)`. `DictDataSource` wraps pre-loaded dicts. `ParquetDataSource` loads lazily from parquet with caching (2x faster than Dict on SPY because it eliminates the dry-run pattern).
- **Strike selectors**: Callable `f(ctx) -> strikes_tuple | nothing`. The `ctx` named tuple: `(surface, expiry, tau, recs, put_strikes, call_strikes)`.
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

| Script | Uses backtest engine? | Purpose |
|--------|----------------------|---------|
| `backtest_polygon_iron_condor.jl` | Yes | Canonical condor backtest with ROI output |
| `backtest_polygon_short_strangle.jl` | Yes | Strangle backtest |
| `ml_strike_selector.jl` | Yes (for validation) | ML training + backtest evaluation |
| `evaluate_condor_prediction_vs_baseline.jl` | **No** (own loop) | Detailed ML vs baseline comparison |
| `compare_datasources.jl` | Yes | Correctness + speed comparison of DictDataSource vs ParquetDataSource |

Scripts use `scripts/Project.toml` via `Pkg.activate(@__DIR__)`.

## Known Technical Debt

### Remaining Duplication

1. **`src/ml/training.jl`** (~1,400 lines):
   - `simulate_strangle_pnl()` vs `simulate_condor_pnl()` share only ~14% code (wing logic fundamentally different)
   - `generate_*` and `train_*` pairs already share core abstractions (`_generate_delta_training_data_core`, `_train_loop!`)
   - Rate/div_yield defaults now use `DEFAULT_RATE`/`DEFAULT_DIV_YIELD` constants

2. **`src/strategies/helpers.jl`**: `_best_delta_strike()` loop pattern repeated 8+ times

3. **`src/backtest/metrics.jl`**: `performance_metrics()` has two near-identical branches for margin computation

### ML Integration Gap

- `evaluate_condor_prediction_vs_baseline.jl` reimplements backtest logic inline (strike resolution, PnL computation, feature extraction) instead of using `backtest_strategy()`
- `training.jl` has its own `condor_metrics_from_strikes()` instead of using `Position`/`settle`
- **Long-term goal**: route all evaluation through the backtest engine

## SPXW Evaluation Results (Feb-Aug 2025, 123 trading days)

### Setup
- Underlying: SPXW options, spot proxied via SPY × 10
- Strategy: 1DTE iron condors, entry at 10:00 ET, ROI-optimized wing selection
- Model: delta-prediction MLP, 59 input features (36 base + 23 pruned logsig)
- Baseline: fixed 16-delta symmetric condor

### Spread Lambda Sweep (friction sensitivity)

| Lambda | Pred Avg ROI | Base Avg ROI | ML Edge | Pred Beats Base |
|--------|-------------|-------------|---------|-----------------|
| 0.0 (canonical) | 6.1% | 2.4% | +3.6% | 65.9% |
| 0.7 | 8.0% | 3.4% | +4.6% | 63.4% |
| 1.0 (no spread) | 8.7% | 4.0% | +4.7% | 64.2% |

- `spread_lambda` controls synthetic bid/ask: 0.0 = bid=low/ask=high (widest, canonical default for all scripts), 1.0 = midpoint (no spread)
- **Key finding**: ML edge is friction-robust. The model's ~2x ROI advantage over baseline holds across all lambda values. The alpha comes from genuine strike selection, not spread-gaming.
- Baseline profitability at all lambda levels reflects the SPX variance risk premium (short-dated options are structurally overpriced due to hedging demand).

### Feature Evolution
- Original model: 45 features (15 base + 30 logsig), 7 near-constant features
- Current model: 59 features (36 base + 23 pruned logsig), 0 constant/near-constant at lambda=0.7
- New feature groups: richer smile (10), spread by moneyness (6), surface dynamics (3), volume (3)
- Dead logsig indices pruned via `LOGSIG_DEAD_INDICES`
- `prev_surface` support added for day-over-day surface change features (`delta_atm_iv_1d`, `delta_skew_1d`, `delta_term_slope_1d`)

### Caveats / Open Questions
- Validation set used for both early stopping and final evaluation -- true OOS holdout (Aug 2025-Feb 2026) preserved for future one-shot test
- SPY × 10 proxy for SPX settlement introduces small tracking error
- No commissions, slippage, or pin risk modeled beyond synthetic spread
- Fixed 16-delta baseline is naive; an adaptive-delta rule might close some of the gap

## Multi-Symbol Lambda Sweep (2026-02-08)

Full experiment write-up: [`experiments/multi_symbol_lambda_sweep/experiment.md`](experiments/multi_symbol_lambda_sweep/experiment.md)

18 independent runs: 6 underlyings (SPXW, SPY, QQQ, IWM, GLD, TLT) x 3 lambda values (0.0, 0.5, 0.7). Per-symbol TOML configs in [`scripts/configs/`](scripts/configs/).

**Key findings:**
- ML edge generalizes to equity options: SPXW (+3.2%), SPY (+1.9%), IWM (+1.2%) weighted avg ROI over baseline
- QQQ marginal (+0.6%), GLD/TLT unprofitable -- no structural variance risk premium to exploit
- Variance risk premium hierarchy: IWM > SPXW > SPY > QQQ > TLT > GLD
- Edge is friction-robust across all lambda values

## Conventions

- Julia 1.10+
- `Float64` for domain values, `Float32` for ML tensors
- `missing` for absent optional data (bid/ask not available)
- `nothing` for "computation failed / skip this entry"
- `DAYS_PER_YEAR = 365.25` for time-to-expiry
- Deribit expiry normalized to 08:00 UTC; Polygon expiry to 16:00 ET (DST-aware)
- **Commit messages**: Do not mention Claude, AI, or any assistant tooling in commit messages or PR descriptions
