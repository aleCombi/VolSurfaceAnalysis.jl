# Status

This is the `rebuilt` branch -- a deliberate clean slate. The full codebase
lives on `master` and is the reference we mine from. Everything here is being
built up one small, deliberate piece at a time.

Rebuild order:

1. **Data** -- done.
2. **Modelling** (vol surface) -- done. `Curve`s, `surfaces` module,
   and `ModelDataSource` composition are in place.
3. **Positions** -- done. `Trade` / `Position` records and the pure
   `payoff` / `open_position` / `entry_cost` / `pnl` primitives. No
   backtest or trading engine yet -- positions stand alone as the layer
   both will consume.
4. **Backtesting** -- next.
5. **Metric computation**.
6. **Experiment orchestration**.

Visualization is added incrementally alongside each stage, not as a phase
of its own.
