using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using VolSurfaceAnalysis
using Dates
using Random
using Plots
using Plots.Measures

const ROOT = get(ENV, "VSA_POLYGON_ROOT", "C:/repos/options-collector/data/massive")
const SYMBOL = get(ENV, "VSA_DEMO_SYMBOL", "AAPL")
const DEMO_DATE = Date(get(ENV, "VSA_DEMO_DATE", "2016-03-28"))

# ---- real series from ParquetDataSource ----

function load_real_spots(symbol, root, date)
    isdir(root) || (@warn "skipping parquet load — $root not found"; return SpotPrice[])
    ds = ParquetDataSource(symbol, root)
    spots = get_spots(ds, DateTime(date), DateTime(date, Time(23, 59, 59)))
    isempty(spots) && @warn "no spot rows for $symbol on $date under $root"
    return spots
end

real_spots = load_real_spots(SYMBOL, ROOT, DEMO_DATE)

# ---- synthetic series anchored to the real series for a fair overlay ----

anchor_price = isempty(real_spots) ? 480.0 : first(real_spots).price
anchor_ts = isempty(real_spots) ? DateTime(2024, 1, 15, 9, 30) : first(real_spots).timestamp
n = isempty(real_spots) ? 390 : length(real_spots)

returns = randn(n) .* 0.0005
prices = anchor_price .* exp.(cumsum(returns))
ts = isempty(real_spots) ?
    [anchor_ts + Minute(i) for i in 0:n - 1] :
    [s.timestamp for s in real_spots]
synthetic = [SpotPrice(Underlying(SYMBOL), p, t) for (t, p) in zip(ts, prices)]

# ---- overlay plot with hourly ticks clipped to the data range ----

all_ts = isempty(real_spots) ? ts : [s.timestamp for s in real_spots]
t_lo, t_hi = extrema(all_ts)
first_hour = ceil(t_lo, Hour)
last_hour = floor(t_hi, Hour)
hour_dts = collect(first_hour:Hour(1):last_hour)
hour_ticks = (Dates.value.(hour_dts), Dates.format.(hour_dts, "HH:00"))

p = plot(synthetic; label="synthetic", linestyle=:dash, color=:gray,
         title="$SYMBOL spot — real vs synthetic ($DEMO_DATE)",
         legend=:topright,
         xlims=(Dates.value(t_lo), Dates.value(t_hi)),
         xticks=hour_ticks,
         xrotation=45,
         bottom_margin=8mm)
isempty(real_spots) || plot!(p, real_spots; label="real", color=:steelblue, linewidth=2)
display(p)
