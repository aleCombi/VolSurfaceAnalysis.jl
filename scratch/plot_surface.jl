# Plot full vol surfaces (moneyness × tenor) with condor strike lines overlaid
# 2×2 grid: normal, pre-crash, crash, post-crash

using Pkg
Pkg.activate(@__DIR__)

using VolSurfaceAnalysis
using Dates
using Plots
gr()

const POLYGON_ROOT = raw"C:\repos\DeribitVols\data\massive_parquet\minute_aggs"
const SPOT_ROOT    = raw"C:\repos\DeribitVols\data\massive_parquet\spot_1min"
const RATE         = 0.045
const DIV_YIELD    = 0.013
const MAX_TENOR_DAYS = 30   # cap tenor axis

DATES = [
    (Date(2024, 3, 4),  "2024-03-04 (normal)"),      # Monday
    (Date(2024, 8, 2),  "2024-08-02 (pre-crash)"),
    (Date(2024, 8, 5),  "2024-08-05 (crash)"),
    (Date(2024, 8, 6),  "2024-08-06 (post-crash)"),
]

"""
    load_surface(date) -> (moneyness, tenor_days, ivs, spot, atm_iv_1d) or nothing

Load full surface for a date. Returns vectors for all (strike, expiry) with valid IV,
tenor capped at MAX_TENOR_DAYS. Also returns 1-day ATM IV for sigma reference.
"""
function load_surface(date::Date)
    date_str     = Dates.format(date, "yyyy-mm-dd")
    spot_path    = joinpath(SPOT_ROOT,    "date=$date_str", "symbol=SPY",     "data.parquet")
    options_path = joinpath(POLYGON_ROOT, "date=$date_str", "underlying=SPY", "data.parquet")
    (!isfile(spot_path) || !isfile(options_path)) && return nothing

    spot_prices = read_spot_prices(spot_path)
    trades      = read_polygon_parquet(options_path)

    # Timestamp near 14:00 UTC
    target_time = DateTime(date) + Hour(14)
    all_ts = sort(unique(t.timestamp for t in trades))
    isempty(all_ts) && return nothing
    diffs    = [abs(Dates.value(t - target_time)) for t in all_ts]
    entry_ts = all_ts[argmin(diffs)]

    spot = get(spot_prices, entry_ts, missing)
    if ismissing(spot)
        for offset in -5:5
            spot = get(spot_prices, entry_ts + Minute(offset), missing)
            !ismissing(spot) && break
        end
    end
    ismissing(spot) && return nothing

    # VolRecords (volume ≥ 10 for surface density)
    ts_trades = filter(t -> t.timestamp == entry_ts, trades)
    recs = VolRecord[]
    for t in ts_trades
        r = trade_to_volrecord(t, spot; min_volume=10, compute_iv=false)
        r !== nothing && push!(recs, r)
    end
    isempty(recs) && return nothing

    # Compute IV for every record, keep one per (strike, expiry) — prefer higher volume
    best = Dict{Tuple{Float64, DateTime}, Tuple{Float64, Float64}}()   # key → (iv, volume)
    for rec in recs
        ismissing(rec.mark_price) && continue
        T = time_to_expiry(rec.expiry, rec.timestamp)
        T <= 0 && continue
        tenor_days = T * 365.25
        tenor_days > MAX_TENOR_DAYS && continue

        F  = rec.underlying_price * exp((RATE - DIV_YIELD) * T)
        iv = price_to_iv(rec.mark_price, F, rec.strike, T, rec.option_type; r=RATE)
        isnan(iv) || iv <= 0 && continue

        key = (rec.strike, rec.expiry)
        vol = coalesce(rec.volume, 0.0)
        if !haskey(best, key) || vol > best[key][2]
            best[key] = (iv * 100, vol)   # store as %
        end
    end
    isempty(best) && return nothing

    # Unpack
    moneyness  = Float64[]
    tenor_days = Float64[]
    ivs        = Float64[]
    for ((strike, expiry), (iv, _)) in best
        T = time_to_expiry(expiry, entry_ts)
        push!(moneyness,  (strike / spot - 1.0) * 100)
        push!(tenor_days, T * 365.25)
        push!(ivs,        iv)
    end

    # 1-day ATM IV: find point closest to (moneyness=0, tenor≈1)
    atm_1d_idx = argmin(abs.(moneyness) .+ abs.(tenor_days .- 1.0) .* 10)
    atm_iv_1d  = ivs[atm_1d_idx] / 100   # decimal

    return (moneyness, tenor_days, ivs, spot, atm_iv_1d)
end

# ---------------------------------------------------------------
# Plot 2×2 grid
# ---------------------------------------------------------------
n = length(DATES)
p = plot(layout=(2, 2), size=(1100, 820), margin=4Plots.mm)

for (i, (date, label)) in enumerate(DATES)
    result = load_surface(date)
    if result === nothing
        println("$label: no data")
        continue
    end
    moneyness, tenor_days, ivs, spot, atm_iv_1d = result
    one_sigma_1d = atm_iv_1d * sqrt(1.0 / 365.25) * 100   # 1σ at 1-day in moneyness %

    println("$label: $(length(ivs)) points, spot=\$$(round(spot; digits=2)), ATM 1D IV=$(round(atm_iv_1d*100; digits=1))%, 1σ=$(round(one_sigma_1d; digits=2))%")

    # Per-subplot color range (2nd–98th percentile)
    sivs = sort(ivs)
    cl = (sivs[max(1, round(Int, 0.02*length(sivs)))],
          sivs[min(end, round(Int, 0.98*length(sivs)))])

    # Surface scatter colored by IV
    scatter!(p[i], moneyness, tenor_days,
             zcolor=ivs, colorbar=true, clims=cl,
             markersize=2.5, markerstrokewidth=0,
             colorbar_title="IV (%)",
             label="")

    # Condor strike lines (fixed %)
    vline!(p[i], [-5.0, -2.0, 2.0, 5.0],
           color=[:green, :orange, :orange, :green],
           linewidth=[1.2, 1.5, 1.5, 1.2],
           linestyle=:solid, label="")

    # Labels on the lines
    annotate!(p[i], -5.0, MAX_TENOR_DAYS * 0.95, text("L", 7, :center, :top, :green))
    annotate!(p[i], -2.0, MAX_TENOR_DAYS * 0.95, text("S", 7, :center, :top, :orange))
    annotate!(p[i],  2.0, MAX_TENOR_DAYS * 0.95, text("S", 7, :center, :top, :orange))
    annotate!(p[i],  5.0, MAX_TENOR_DAYS * 0.95, text("L", 7, :center, :top, :green))

    plot!(p[i],
        xlabel    = "Moneyness (%)",
        ylabel    = "Tenor (days)",
        title     = "$label",
        xlims     = (-8, 8),
        ylims     = (0, MAX_TENOR_DAYS),
        grid      = true,
        gridalpha = 0.2)
end

out = joinpath(@__DIR__, "surfaces.png")
savefig(p, out)
println("\nSaved: $out")
