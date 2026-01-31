# Vol smile at ~1 day tenor with condor strike lines

using Pkg
Pkg.activate(@__DIR__)

using VolSurfaceAnalysis
using Dates
using Plots
using Statistics
gr()

const POLYGON_ROOT = raw"C:\repos\DeribitVols\data\massive_parquet\minute_aggs"
const SPOT_ROOT    = raw"C:\repos\DeribitVols\data\massive_parquet\spot_1min"
const RATE         = 0.045
const DIV_YIELD    = 0.013
const OUTPUT_DIR   = joinpath(@__DIR__, "plots")

date     = Date(2026, 1, 28)
date_str = Dates.format(date, "yyyy-mm-dd")

# ---------------------------------------------------------------
# Load
# ---------------------------------------------------------------
mkpath(OUTPUT_DIR)

spot_prices = read_spot_prices(joinpath(SPOT_ROOT, "date=$date_str", "symbol=SPY", "data.parquet"))
trades      = read_polygon_parquet(joinpath(POLYGON_ROOT, "date=$date_str", "underlying=SPY", "data.parquet"))

# Timestamp near 14:00 UTC
target_time = DateTime(date) + Hour(14)
all_ts      = sort(unique(t.timestamp for t in trades))
entry_ts    = all_ts[argmin([abs(Dates.value(t - target_time)) for t in all_ts])]

spot = get(spot_prices, entry_ts, missing)
if ismissing(spot)
    for offset in -5:5
        global spot
        spot = get(spot_prices, entry_ts + Minute(offset), missing)
        !ismissing(spot) && break
    end
end

# VolRecords at this timestamp
ts_trades = filter(t -> t.timestamp == entry_ts, trades)
recs = VolRecord[]
for t in ts_trades
    r = trade_to_volrecord(t, spot; min_volume=10, compute_iv=false)
    r !== nothing && push!(recs, r)
end

# Expiry closest to +1 day
expiries   = sort(unique(r.expiry for r in recs))
target_exp = entry_ts + Day(1)
expiry     = expiries[argmin([abs(Dates.value(e - target_exp)) for e in expiries])]
tau        = time_to_expiry(expiry, entry_ts)

# Compute IV per strike at this expiry, tracking option type
smile_recs = filter(r -> r.expiry == expiry, recs)
call_moneyness = Float64[]
call_ivs       = Float64[]
put_moneyness  = Float64[]
put_ivs        = Float64[]

for rec in smile_recs
    ismissing(rec.mark_price) && continue
    T  = time_to_expiry(rec.expiry, rec.timestamp)
    T <= 0 && continue
    F  = rec.underlying_price * exp((RATE - DIV_YIELD) * T)
    iv = price_to_iv(rec.mark_price, F, rec.strike, T, rec.option_type; r=RATE)
    if !isnan(iv) && iv > 0
        m = (rec.strike / spot - 1.0) * 100
        if rec.option_type == Call
            push!(call_moneyness, m)
            push!(call_ivs, iv * 100)
        else
            push!(put_moneyness, m)
            push!(put_ivs, iv * 100)
        end
    end
end

# Sort by moneyness for line plotting
call_order = sortperm(call_moneyness)
call_moneyness = call_moneyness[call_order]
call_ivs = call_ivs[call_order]

put_order = sortperm(put_moneyness)
put_moneyness = put_moneyness[put_order]
put_ivs = put_ivs[put_order]

println("$(length(call_ivs)) calls, $(length(put_ivs)) puts, spot=\$$(round(spot; digits=2)), tau=$(round(tau*365.25; digits=2))d")

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
p = plot(size=(860, 480), margin=5Plots.mm)

# Plot calls (steelblue) and puts (coral) separately
plot!(p, call_moneyness, call_ivs,
      color=:steelblue, linewidth=2,
      markershape=:circle, markersize=4, markerstrokewidth=0, markercolor=:steelblue,
      label="Calls")
plot!(p, put_moneyness, put_ivs,
      color=:coral, linewidth=2,
      markershape=:circle, markersize=4, markerstrokewidth=0, markercolor=:coral,
      label="Puts")

# Calculate ATM IV (average of calls and puts near 0% moneyness)
all_moneyness = vcat(call_moneyness, put_moneyness)
all_ivs = vcat(call_ivs, put_ivs)
atm_mask = abs.(all_moneyness) .< 1.0  # within ±1% of ATM
atm_iv = sum(atm_mask) > 0 ? mean(all_ivs[atm_mask]) : 15.0  # default 15% if no ATM data

# Iron condor strikes proportional to ATM IV (scaled by sqrt(tau) for 1-day expiry)
# Short strikes at ±0.5σ, Long strikes at ±1.0σ
daily_vol = atm_iv * sqrt(tau)  # daily expected move in %
short_width = 0.5 * daily_vol   # short strikes at 0.5 sigma
long_width  = 1.0 * daily_vol   # long strikes at 1.0 sigma

println("ATM IV: $(round(atm_iv; digits=1))%, Daily vol: $(round(daily_vol; digits=2))%")
println("Condor: Long ±$(round(long_width; digits=2))%, Short ±$(round(short_width; digits=2))%")

# Condor strikes
vline!(p, [-long_width, -short_width], color=[:green, :orange], linewidth=[1.5, 2.0], label="")
vline!(p, [short_width, long_width],   color=[:orange, :green], linewidth=[2.0, 1.5], label="")

# Labels - Iron Condor: Long put @ -1σ, Short put @ -0.5σ, Short call @ +0.5σ, Long call @ +1σ
max_iv = max(maximum(call_ivs; init=0), maximum(put_ivs; init=0))
annotate!(p, -long_width,  max_iv*1.02, text("Long Put\n-1σ",   8, :center, :bottom, :green))
annotate!(p, -short_width, max_iv*1.02, text("Short Put\n-0.5σ", 8, :center, :bottom, :orange))
annotate!(p,  short_width, max_iv*1.02, text("Short Call\n+0.5σ", 8, :center, :bottom, :orange))
annotate!(p,  long_width,  max_iv*1.02, text("Long Call\n+1σ",  8, :center, :bottom, :green))

# Dynamic x-axis limits based on data range (include all points)
data_max = max(maximum(abs.(call_moneyness); init=0), maximum(abs.(put_moneyness); init=0))
xlim = max(data_max * 1.1, long_width * 1.5, 3.0)

plot!(p,
    xlabel    = "Moneyness (%)",
    ylabel    = "Implied Vol (%)",
    title     = "SPY 1D Smile | $(date) | Spot: \$$(round(spot; digits=2)) | ATM IV: $(round(atm_iv; digits=1))%",
    xlims     = (-xlim, xlim),
    grid      = true,
    gridalpha = 0.25)

out = joinpath(OUTPUT_DIR, "smile_1d.png")
savefig(p, out)
println("Saved: $out")
