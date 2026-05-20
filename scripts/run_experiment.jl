using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using VolSurfaceAnalysis

length(ARGS) == 1 || error(
    "usage: julia --project=. scripts/run_experiment.jl <config.toml>")

exp = load_experiment(ARGS[1])
res = run_experiment(exp)
show(stdout, MIME"text/plain"(), res)
println()
