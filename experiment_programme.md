# Experiment Programme

## Purpose

This document sets the direction for turning the current collection of large
`scripts/*.jl` files into a smaller number of standardized, reproducible
experiment entrypoints.

The immediate goal is not to design a grand framework. The immediate goal is to
move reusable orchestration out of `scripts/` so that an experiment script does
only two things:

1. create configuration
2. call a runner

The workflow we want is:

1. explore in `scratch/`
2. promote reusable logic into `src/` when the idea stabilizes
3. distill a surviving idea into a thin experiment script

An experiment is therefore not the same thing as a strategy.

- A `Strategy` decides what to trade.
- An `Experiment` specifies how that strategy is run:
  symbol, timeframe, schedule, pricing assumptions, outputs, and reporting.

We do not need a new `Experiment` struct yet. We only need a disciplined shape.

## Current Problem

Most scripts currently mix together several concerns:

- configuration
- data loading
- schedule building
- strategy construction
- backtest execution
- metrics and summaries
- CSV export
- plotting
- run-directory management
- walk-forward window logic

This has three costs:

- scripts are too large to compare or trust quickly
- reproducibility depends on hidden script-local choices
- improvements have to be copied into many files

The result is that `scripts/` acts as both scratch space and productionized
experiment space. Those need to be separated more clearly.

## Near-Term Goal

Make experiment scripts thin and standardized without introducing new heavy
abstractions.

In the near term, this means:

- shared logic moves out of `scripts/`
- experiment scripts become mostly config plus a single call
- the reusable layer can use plain functions and `NamedTuple` config
- we avoid adding speculative types until the repeated patterns are proven

## Non-Goals

This programme is intentionally narrow. It is not trying to do any of the
following yet:

- design a generic research DSL
- unify every experiment shape behind one abstract interface
- replace `scratch/`
- redesign the strategy interface
- solve all walk-forward / ML / optimization abstractions in one pass
- freeze a public API for experiments

## Design Principles

### 1. Scripts Are Entry Points

An experiment script should read like this:

```julia
using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates

cfg = (
    name = "spy_baseline",
    symbol = "SPY",
    start_date = Date(2016, 3, 28),
    end_date = Date(2024, 1, 31),
    entry_time = Time(12, 0),
    expiry_interval = Day(1),
    spread_lambda = 0.7,
    rate = 0.045,
    div_yield = 0.013,
    put_delta = 0.20,
    call_delta = 0.20,
    max_loss = 8.0,
)

run_baseline_condor_experiment(cfg)
```

That is the target shape. The script is declarative. The mechanics live
elsewhere.

### 2. `scratch/` Is Allowed To Be Messy

Exploration should stay fast.

- `scratch/` can duplicate logic
- `scratch/` can print ad hoc diagnostics
- `scratch/` can be ugly

But once an idea graduates into `scripts/`, it should no longer carry bespoke
execution plumbing.

### 3. Shared Logic Lives In `src/`, Not In Many Scripts

If code is needed by more than one experiment, it should not stay copied across
scripts. It should move into package code, even if it is not exported yet.

This keeps experiment behavior inspectable and testable in one place.

### 4. Reproducibility Beats Cleverness

An experiment runner should write down enough context that a run can be
understood later:

- experiment name
- run timestamp
- config snapshot
- output artifacts
- summary metrics
- trade-level results when relevant

Git commit capture and environment capture are desirable, but can be added after
the first extraction pass.

### 5. Narrow Runners Are Better Than Premature Generalization

We do not need one universal `run_experiment`.

It is acceptable, and likely better, to have a small number of focused runners,
for example:

- `run_static_strategy_experiment`
- `run_walkforward_delta_experiment`
- `run_ml_filter_experiment`

If those later converge into a single shape, we can unify them then.

## Proposed Refactor Boundary

The first extraction pass should move the following categories out of scripts.

### A. Run Management

Shared responsibilities:

- build run directory names
- create output directories
- write config snapshots
- write summary tables

Candidate helpers:

- `make_run_dir(name)`
- `write_config(path, cfg)`
- `write_metrics(path, metrics)`
- `write_trade_table(path, df)`

### B. Source / Schedule Builders

Shared responsibilities:

- resolve available dates
- build entry timestamps
- load spots
- build `ParquetDataSource`
- standardize symbol / spot-symbol / multiplier handling

Candidate helpers:

- `build_polygon_source(cfg)`
- `build_entry_schedule(cfg, dates)`
- `build_source_and_schedule(cfg)`

### C. Standard Result Reporting

Shared responsibilities:

- compute `performance_metrics`
- print a standard console summary
- emit trade tables
- emit standard plots

Candidate helpers:

- `summarize_result(result; cfg, label)`
- `save_standard_backtest_outputs(run_dir, result; cfg, label)`

### D. Repeated Walk-Forward Mechanics

A second pass can extract repeated logic for:

- rolling train/test windows
- parameter-grid evaluation
- honest out-of-sample aggregation
- per-fold diagnostics

This should be done after the static experiment path is cleaned up.

## Where The Code Should Go

Near term: put experiment machinery under `src/`, but do not treat it as stable
public API yet.

Suggested layout:

```text
src/
  experiments/
    common.jl
    polygon.jl
    reporting.jl
    static_backtests.jl
    walkforward.jl
```

Initial export policy:

- export only what is clearly stable
- otherwise keep helpers internal to `VolSurfaceAnalysis`

This is preferable to growing `scripts/lib/` into a second codebase.

## Config Shape

We do not need structs yet. Use plain `NamedTuple` config with predictable
field names.

Guidelines:

- prefer flat config over deeply nested config
- use the same field names across experiments where possible
- strategy-specific fields can be appended naturally
- avoid passing dozens of positional arguments

Example:

```julia
cfg = (
    name = "spy_baseline",
    symbol = "SPY",
    spot_symbol = "SPY",
    spot_multiplier = 1.0,
    start_date = Date(2016, 3, 28),
    end_date = Date(2024, 1, 31),
    entry_times = [Time(12, 0)],
    expiry_interval = Day(1),
    spread_lambda = 0.7,
    rate = 0.045,
    div_yield = 0.013,
    min_volume = 0,
    put_delta = 0.20,
    call_delta = 0.20,
    max_loss = 8.0,
    max_spread_rel = 0.50,
)
```

If this shape remains stable across several experiments, that is the moment to
consider a real config type.

## Script Standard

After the refactor, a production experiment script should:

- define config
- optionally define a tiny strategy-builder closure
- call one runner
- contain no custom parquet plumbing
- contain no custom run-dir plumbing
- contain no custom CSV-writing boilerplate
- contain no custom standard-metrics boilerplate

Allowed script-local code:

- experiment-specific config
- experiment-specific grid definitions
- genuinely experiment-specific analysis that is not yet reusable

Not allowed script-local code:

- duplicated source-building code
- duplicated metrics/reporting code
- duplicated artifact-writing code

## Phased Plan

### Phase 1: Static Experiments

Target the simplest and most repeated pattern first:

- load a symbol and schedule
- build one strategy
- run one backtest
- save standard outputs

Reference candidates:

- `scripts/condor_grid.jl`
- `scripts/strangle_grid.jl`

Deliverables:

- common source builder
- common run/output writer
- common static backtest runner
- one migrated reference script

### Phase 2: Signal / Filter Experiments

Sizing/filter/log-signature scripts are not first-class entrypoints right now.
When that line of work resumes, extract the common pieces for:

- training data generation
- model fit
- baseline vs filtered run comparison
- standard summary tables

Reference candidates should start in `scratch/` and only return to `scripts/`
once they are thin wrappers around package-side helpers.

### Phase 3: Walk-Forward Selection Experiments

Only after Phases 1 and 2.

Target repeated windowing logic in:

- rolling delta selection
- rolling wing selection
- rolling regularized ranking

The aim is not to erase experiment-specific logic. The aim is to remove the
boilerplate around window construction, output layout, and standard summaries.

## Definition Of Done For An Experiment

An experiment is considered properly distilled when:

- there is a script in `scripts/`
- the script is thin and declarative
- the script depends on shared package-side helpers
- the script writes a standard run directory
- the main assumptions are visible in one config block
- rerunning the script does not require editing helper code

## Decision Rules

When deciding whether something belongs in `scratch/`, `src/`, or `scripts/`:

- If it is exploratory and unstable, keep it in `scratch/`.
- If it is reusable mechanics, move it to `src/`.
- If it is a repeatable research result, make it a thin script in `scripts/`.

When deciding whether to add a new abstraction:

- first prefer a plain function
- then prefer a config `NamedTuple`
- only add a struct when the shape is repeated and clearly stable

## First Concrete Migration

The first migration should be intentionally boring:

1. extract run-directory and output-writing helpers
2. extract Polygon source/schedule construction
3. extract one static backtest runner
4. rewrite `condor_grid.jl` and `strangle_grid.jl` to use that path

If that result feels cleaner and easier to trust, the programme is working.

If it still feels forced, stop and simplify again before adding more layers.
