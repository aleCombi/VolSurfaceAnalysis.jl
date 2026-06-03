# VolSurfaceAnalysis

A Julia research codebase for options strategy backtesting: each
experiment is declared in a TOML file, run end-to-end through one
engine, and prints its result. The long-term goal is a knowledge base
of rerunnable experiments; see [docs/vision.md](docs/vision.md).

`master` is a deliberate clean-slate rebuild; the prior full codebase is
preserved on the `legacy` branch -- see [docs/status.md](docs/status.md) for
what has landed.

## Run

```bash
julia --project=. -e "using Pkg; Pkg.test()"
julia --project=. scripts/run_experiment.jl configs/noop_smoke.toml          # print result
julia --project=. scripts/run_experiment.jl configs/noop_smoke.toml --save   # persist to the KB
```

## Docs

- [docs/status.md](docs/status.md) -- current state
- [docs/vision.md](docs/vision.md) -- where it's going
- [docs/design.md](docs/design.md) -- coding rules
- [docs/modules/](docs/modules) -- per-module notes
