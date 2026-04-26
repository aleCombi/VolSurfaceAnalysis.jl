# I want the rolling short-strangle selector experiment.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates

experiment = (;
    symbol="SPY",
    start_date=Date(2014, 6, 2),
    end_date=Date(2026, 3, 27),
    entry_time=Time(14, 0),
    expiry_interval=Day(1),
    max_tau_days=2.0,
    spread_lambda=0.7,
    rate=0.045,
    div_yield=0.013,
    put_deltas=collect(0.05:0.05:0.40),
    call_deltas=collect(0.05:0.05:0.40),
    train_days=90,
    test_days=30,
    step_days=30,
    z_values=[0.0, 0.1, 0.3, 1.0, 3.0],
    cvar_alpha=0.05,
    baseline_combo=(0.20, 0.05),
)

run_strangle_rolling_experiment(; output_root=@__DIR__, experiment...)
