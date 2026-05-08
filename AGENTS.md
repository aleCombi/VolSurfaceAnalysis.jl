# AGENTS.md -- VolSurfaceAnalysis (rebuild)

## Status

This is the `rebuilt` branch -- a deliberate clean slate. The full codebase
lives on `master` and is the reference we mine from. Everything here is being
built up one small, deliberate piece at a time.

Rebuild order:

1. **Data** -- done.
2. **Modelling** (vol surface) -- next.
3. **Backtesting**.
4. **Metric computation**.
5. **Experiment orchestration**.

Visualization is added incrementally alongside each stage, not as a phase
of its own.

## General rules

1. Follow the rules in [docs/design.md](docs/design.md).
2. When a user prompt implies a new rule or a change to an existing one
   (in `docs/design.md` or any module doc), propose the edit explicitly
   and surface it before applying it -- don't quietly absorb it.
