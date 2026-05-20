# VolSurfaceAnalysis

A Julia research codebase for options strategy backtesting: each
experiment is declared in a TOML file, run end-to-end through one
engine, and prints its result. The long-term goal is a knowledge base
of rerunnable experiments; see [docs/vision.md](docs/vision.md).

This branch (`rebuilt`) is a deliberate clean slate -- see
[docs/status.md](docs/status.md) for what has landed.

## Run

```bash
julia --project=. -e "using Pkg; Pkg.test()"
julia --project=. scripts/run_experiment.jl configs/noop_smoke.toml
```

## Docs

- [docs/status.md](docs/status.md) -- current state
- [docs/vision.md](docs/vision.md) -- where it's going
- [docs/design.md](docs/design.md) -- coding rules
- [docs/modules/](docs/modules) -- per-module notes
