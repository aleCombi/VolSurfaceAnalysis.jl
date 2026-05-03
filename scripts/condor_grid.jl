# I want the non-rolling iron-condor grid across SPY/QQQ/IWM/SPXW.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates

experiment = (;
    symbols=[
        ("SPY", "SPY", 1.0),
        ("QQQ", "QQQ", 1.0),
        ("IWM", "IWM", 1.0),
        ("SPXW", "SPY", 10.0),
    ],
    start_date=Date(2024, 1, 1),
    end_date=Date(2025, 12, 31),
    entry_time=Time(10, 0),
    expiry_interval=Day(1),
    spread_lambda=0.7,
    rate=0.045,
    div_yield=0.013,
    max_spread_rel=0.50,
    delta_grid=[0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.25, 0.30],
    max_loss_grid=[2.0, 3.0, 5.0, 7.0, 10.0, 15.0, Inf],
    reference_delta=0.16,
    reference_max_loss=5.0,
)

run_condor_grid_experiment(; output_root=@__DIR__, experiment...)
