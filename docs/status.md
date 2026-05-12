# Status

This is the `rebuilt` branch -- a deliberate clean slate. The full codebase
lives on `master` and is the reference we mine from. Everything here is being
built up one small, deliberate piece at a time.

Rebuild order:

1. **Data** -- done.
2. **Modelling** (vol surface) -- done. `Curve`s, `surfaces` module,
   and `ModelDataSource` composition are in place.
3. **Backtesting** -- next.
4. **Metric computation**.
5. **Experiment orchestration**.

Visualization is added incrementally alongside each stage, not as a phase
of its own.
