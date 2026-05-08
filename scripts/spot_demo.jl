using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using VolSurfaceAnalysis
using Dates
using Random
using Plots

# ---- synthetic series (kept as fallback / sanity check) ----

start_ts = DateTime(2024, 1, 15, 9, 30)
n = 390
ts = [start_ts + Minute(i) for i in 0:n - 1]
returns = randn(n) .* 0.0005
prices = 480.0 .* exp.(cumsum(returns))
synthetic = [SpotPrice(Underlying("SPY"), p, t) for (t, p) in zip(ts, prices)]

display(plot(synthetic; title="SPY spot (synthetic)"))

# ---- real series from ParquetDataSource ----

const ROOT = get(ENV, "VSA_POLYGON_ROOT", "C:/repos/options-collector/data/massive")
const SYMBOL = get(ENV, "VSA_DEMO_SYMBOL", "AAPL")
const DEMO_DATE = Date(get(ENV, "VSA_DEMO_DATE", "2016-03-28"))

if isdir(ROOT)
    ds = ParquetDataSource(SYMBOL, ROOT)
    try
        from = DateTime(DEMO_DATE)
        to = DateTime(DEMO_DATE, Time(23, 59, 59))
        spots = get_spots(ds, from, to)
        if isempty(spots)
            @warn "no spot rows for $SYMBOL on $DEMO_DATE under $ROOT"
        else
            display(plot(spots; title="$SYMBOL spot — $DEMO_DATE"))
        end
    finally
        close(ds)
    end
else
    @warn "skipping parquet demo — $ROOT not found (set VSA_POLYGON_ROOT to override)"
end
