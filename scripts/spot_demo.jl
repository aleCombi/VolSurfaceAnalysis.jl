using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using VolSurfaceAnalysis
using Dates
using Random
using Plots

# Random.seed!(42)

start_ts = DateTime(2024, 1, 15, 9, 30)
n = 390
ts = [start_ts + Minute(i) for i in 0:n - 1]

returns = randn(n) .* 0.0005
prices = 480.0 .* exp.(cumsum(returns))

spots = [SpotPrice(Underlying("SPY"), p, t) for (t, p) in zip(ts, prices)]

plot(spots)
