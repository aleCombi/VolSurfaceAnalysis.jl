using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using VolSurfaceAnalysis
using Dates
using Plots
using Plots.Measures

const ROOT = get(ENV, "VSA_POLYGON_ROOT", "C:/repos/options-collector/data/massive")
const SYMBOL = get(ENV, "VSA_DEMO_SYMBOL", "SPY")
const DEMO_DATE = Date(get(ENV, "VSA_DEMO_DATE", "2024-06-03"))
const N_SLICES = parse(Int, get(ENV, "VSA_DEMO_N_SLICES", "5"))

# Flat r and q for SPY -- reasonable mid-2024 USD short rate and SPY div yield.
const RATE = 0.045
const DIV  = 0.013

function pick_midday_ts(ds::ParquetDataSource, date::Date)
    from = DateTime(date)
    to   = DateTime(date, Time(23, 59, 59))
    ts = available_timestamps(ds, from, to)
    isempty(ts) && error("no chain timestamps on $date for $(ticker(ds.underlying))")
    target = DateTime(date, Time(16, 30))  # ~12:30 ET
    _, i = findmin(t -> abs(Dates.value(t - target)), ts)
    return ts[i]
end

with_parquet_source(SYMBOL, ROOT) do raw
    mds = ModelDataSource(raw; rate=FlatCurve(RATE), div=FlatCurve(DIV))

    ts = pick_midday_ts(raw, DEMO_DATE)
    surf = get_surface(mds, ts)
    surf === nothing && error("no surface built at $ts")

    spot = surf.spot
    all_exps = expiries(surf)
    @info "surface built" ts spot n_expiries=length(all_exps) rate=RATE div=DIV

    # Pick the N nearest expiries with tau > 0.
    n = min(N_SLICES, length(all_exps))
    exps = all_exps[1:n]

    p = plot(;
        title="$SYMBOL vol slices @ $(ts) UTC  (S=$(round(spot, digits=2)))",
        xlabel="Strike",
        ylabel="Implied vol",
        legend=:topright,
        bottom_margin=6mm)

    for e in exps
        sl = get_slice(surf, e)
        sl === nothing && continue
        dte = round(sl.tau * 365.25, digits=1)
        plot!(p, sl.strikes, sl.ivs;
              label="$(Date(e))  ($(dte)d)",
              marker=:circle, markersize=2, linewidth=1.5)
    end
    vline!(p, [spot]; color=:black, linestyle=:dash, linewidth=1, label="spot")
    display(p)
end
