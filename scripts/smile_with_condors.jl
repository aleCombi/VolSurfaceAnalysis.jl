# I want one random SPY smile snapshot with fixed-delta condors overlaid.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates

experiment = (;
    symbol="SPY",
    start_date=Date(2024, 1, 1),
    end_date=Date(2025, 6, 30),
    entry_time=Time(13, 0),
    window_minutes=5,
    target_tenor_days=1.0,
    min_tenor_days=0.5,
    atm_window=0.03,
    spread_lambda=0.7,
    rate=0.045,
    div_yield=0.013,
    seed=42,
    condor_specs=[
        CondorSpec(0.30, 0.10, :firebrick, "30d / 10d"),
        CondorSpec(0.16, 0.05, :darkorchid, "16d / 05d"),
        CondorSpec(0.10, 0.03, :royalblue, "10d / 03d"),
    ],
)

run_smile_with_condors_experiment(; output_root=@__DIR__, experiment...)
