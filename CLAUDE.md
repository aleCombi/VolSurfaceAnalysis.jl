# CLAUDE.md -- VolSurfaceAnalysis (rebuild)

## Status

This is the `rebuilt` branch -- a deliberate clean slate. The full codebase
lives on `master` and is the reference we mine from. Everything here is being
built up one small, deliberate piece at a time.

## Rebuild order

1. Data loading -- read raw parquet files into typed records.
2. Strategy backtesting -- one strategy, end-to-end PnL.
3. Reporting -- minimal performance summary.
4. Experiment orchestration -- thin scripts that wire the above.

Each piece lands as its own commit with a short rationale. We grow an
abstraction only when a concrete second use forces it.

## Conventions

- Julia 1.10+
- `Float64` for domain values
- `missing` for absent optional data, `nothing` for "computation failed / skip"
- Commit messages: never mention Claude, AI, or assistant tooling
