# I want the non-rolling SPY short/long strangle grid and derived condor combos.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates

experiment = (;
    symbol="SPY",
    start_date=Date(2016, 3, 28),
    end_date=Date(2024, 1, 31),
    entry_time=Time(12, 0),
    expiry_interval=Day(1),
    spread_lambda=0.7,
    rate=0.045,
    div_yield=0.013,
    put_deltas=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    call_deltas=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    selected=[
        (0.05, 0.05, "5d/5d (widest)"),
        (0.20, 0.05, "20p/5c (asym sweet)"),
        (0.20, 0.20, "20d/20d (symmetric)"),
        (0.30, 0.30, "30d/30d (tightest)"),
        (0.05, 0.30, "5p/30c (inverted)"),
    ],
    condor_combos=[
        (0.10, 0.10, 0.05, 0.05, "10/10 short, 5/5 wings"),
        (0.15, 0.15, 0.05, 0.05, "15/15 short, 5/5 wings"),
        (0.20, 0.20, 0.10, 0.10, "20/20 short, 10/10 wings"),
        (0.20, 0.20, 0.05, 0.05, "20/20 short, 5/5 wings"),
        (0.25, 0.25, 0.10, 0.10, "25/25 short, 10/10 wings"),
        (0.30, 0.30, 0.20, 0.20, "30/30 short, 20/20 wings"),
        (0.20, 0.10, 0.10, 0.05, "20p/10c short, 10p/5c wings"),
        (0.20, 0.05, 0.10, 0.05, "20p/5c short, 10p/5c wings"),
        (0.25, 0.10, 0.10, 0.05, "25p/10c short, 10p/5c wings"),
        (0.25, 0.10, 0.15, 0.05, "25p/10c short, 15p/5c wings"),
        (0.15, 0.10, 0.05, 0.05, "15p/10c short, 5p/5c wings"),
        (0.20, 0.15, 0.10, 0.05, "20p/15c short, 10p/5c wings"),
    ],
    loss_strangles=[
        (0.05, 0.05, "Short 5d/5d"),
        (0.10, 0.05, "Short 10p/5c"),
        (0.20, 0.05, "Short 20p/5c"),
    ],
    loss_condors=[
        (0.10, 0.10, 0.05, 0.05, "Condor 10/10 short, 5/5 wings"),
        (0.20, 0.10, 0.10, 0.05, "Condor 20p/10c short, 10p/5c wings"),
        (0.20, 0.20, 0.10, 0.10, "Condor 20/20 short, 10/10 wings"),
    ],
)

run_strangle_grid_experiment(; output_root=@__DIR__, experiment...)
