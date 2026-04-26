# I want the rolling iron-condor selector experiment.
# Choose one mode: "delta", "wing", "joint", "2stage", or "cross_tenor".

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates

experiment = (;
    mode="delta",
    symbol="SPY",
    start_date=Date(2017, 1, 1),
    end_date=Date(2024, 1, 31),
    entry_time=Time(14, 0),
    expiry_interval=Hour(2),
    max_tau_days=0.5,
    spread_lambda=0.7,
    rate=0.045,
    div_yield=0.013,
    train_days=90,
    test_days=30,
    step_days=30,
    put_deltas=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    call_deltas=[0.05, 0.10, 0.15, 0.20],
    wing_widths=[1.0, 2.0, 3.0, 5.0, 8.0, 12.0],
    selection_wing_width=12.0,
    put_delta=0.20,
    call_delta=0.05,
    in_sample_end=Date(2020, 12, 31),
    tenor_train_str="1d",
    tenor_test_str="2h",
    max_tau_train=2.0,
    max_tau_test=0.5,
    entry_train=Time(14, 0),
    entry_test=Time(14, 0),
)

run_condor_rolling_experiment(; output_root=@__DIR__, experiment...)
