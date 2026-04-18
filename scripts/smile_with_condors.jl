# scripts/smile_with_condors.jl
#
# Pick a random SPY snapshot, build a 5-min VWAP volatility surface, and call
# `plot_smile_with_condors` to render the put/call smile for a ~1-day expiry
# with a few fixed-delta iron condors overlaid.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Random, Plots

# =============================================================================
# Configuration
# =============================================================================

SYMBOL              = "SPY"
START_DATE          = Date(2024, 1, 1)
END_DATE            = Date(2025, 6, 30)
ENTRY_TIME          = Time(13, 0)
WINDOW_MINUTES      = 5
TARGET_TENOR_DAYS   = 1.0   # closest expiry to this (skipping 0DTE)
MIN_TENOR_DAYS      = 0.5   # skip same-day expiries
ATM_WINDOW          = 0.03  # smile range = ±3% of spot
RATE                = 0.045
DIV_YIELD           = 0.013
SEED                = 42

CONDOR_SPECS = [
    CondorSpec(0.30, 0.10, :firebrick,  "30Δ / 10Δ"),
    CondorSpec(0.16, 0.05, :darkorchid, "16Δ / 05Δ"),
    CondorSpec(0.10, 0.03, :royalblue,  "10Δ / 03Δ"),
]

# =============================================================================
# Output dir
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "smile_with_condors_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

# =============================================================================
# Load 5-min VWAP records
# =============================================================================

SEED === nothing || Random.seed!(SEED)

store = DEFAULT_STORE
all_dates = available_polygon_dates(store, SYMBOL)
filtered = filter(d -> START_DATE <= d <= END_DATE, all_dates)
isempty(filtered) && error("No $SYMBOL dates in range")

attempts = 0
all_recs = OptionRecord[]
ts_start = nothing
spot_at_window = NaN
chosen_date = nothing
while isempty(all_recs) && attempts < 10
    global chosen_date = rand(filtered)
    global attempts += 1
    global ts_start = et_to_utc(chosen_date, ENTRY_TIME)
    ts_end = ts_start + Minute(WINDOW_MINUTES - 1)
    window_ts = [ts_start + Minute(i) for i in 0:(WINDOW_MINUTES - 1)]
    spot_dict = read_polygon_spot_prices_for_timestamps(
        polygon_spot_root(store), window_ts; symbol=SYMBOL
    )
    isempty(spot_dict) && continue
    path = polygon_options_path(store, chosen_date, SYMBOL)
    isfile(path) || continue
    where = "timestamp BETWEEN '$(Dates.format(ts_start, "yyyy-mm-dd HH:MM:SS"))' " *
            "AND '$(Dates.format(ts_end, "yyyy-mm-dd HH:MM:SS"))'"
    global all_recs = read_polygon_option_records(
        path, spot_dict; where=where, min_volume=0, warn=false, spread_lambda=0.7,
        rate=RATE, div_yield=DIV_YIELD,
    )
    global spot_at_window = haskey(spot_dict, ts_start) ? spot_dict[ts_start] : first(values(spot_dict))
end
isempty(all_recs) && error("No records after $attempts attempts")

println("\nDate: $chosen_date  spot≈$(round(spot_at_window, digits=2))  raw bars: $(length(all_recs))")

# =============================================================================
# VWAP-aggregate to one record per (K, expiry, side)
# =============================================================================

function aggregate_records(recs::Vector{OptionRecord}, ts_anchor::DateTime)
    grouped = Dict{Tuple{Float64,DateTime,OptionType},Vector{OptionRecord}}()
    for r in recs
        push!(get!(grouped, (r.strike, r.expiry, r.option_type), OptionRecord[]), r)
    end
    out = OptionRecord[]
    for ((K, expiry, opt), grp) in grouped
        vols = [coalesce(r.volume, 0.0) for r in grp]
        wts = sum(vols) > 0 ? vols ./ sum(vols) : fill(1.0/length(grp), length(grp))
        closes_usd = [coalesce(r.mark_price, NaN) * r.spot for r in grp]
        spots = [r.spot for r in grp]
        vwap_close = sum(wts .* closes_usd)
        vwap_spot  = sum(wts .* spots)
        T = time_to_expiry(expiry, ts_anchor)
        mark_frac = vwap_close / vwap_spot
        F_text = vwap_spot * exp((RATE - DIV_YIELD) * T)
        iv = price_to_iv(vwap_close / F_text, F_text, K, T, opt; r=RATE)
        mark_iv = isnan(iv) ? missing : iv * 100.0
        bid_frac = sum(wts .* [coalesce(r.bid_price, NaN) for r in grp])
        ask_frac = sum(wts .* [coalesce(r.ask_price, NaN) for r in grp])
        push!(out, OptionRecord(
            grp[1].instrument_name, grp[1].underlying, expiry, K, opt,
            isnan(bid_frac) ? missing : bid_frac,
            isnan(ask_frac) ? missing : ask_frac,
            mark_frac, mark_iv,
            missing, sum(vols), vwap_spot, ts_anchor,
        ))
    end
    return out
end

agg = aggregate_records(all_recs, ts_start)
surface = build_surface(agg)

# =============================================================================
# Pick the expiry closest to TARGET_TENOR_DAYS (skipping anything < MIN_TENOR_DAYS)
# =============================================================================

expiries = unique(rec.expiry for rec in surface.records)
chosen_expiry = nothing
best_score = Inf
for e in expiries
    T = time_to_expiry(e, surface.timestamp)
    T * 365.25 >= MIN_TENOR_DAYS || continue
    score = abs(T * 365.25 - TARGET_TENOR_DAYS)
    if score < best_score
        global best_score = score
        global chosen_expiry = e
    end
end
chosen_expiry === nothing && error("No expiry with T ≥ $MIN_TENOR_DAYS days")
T_d = time_to_expiry(chosen_expiry, surface.timestamp) * 365.25
println("Chosen expiry: $chosen_expiry  T=$(round(T_d, digits=2))d")

# =============================================================================
# Plot via the library function
# =============================================================================

plt = plot_smile_with_condors(surface, chosen_expiry, CONDOR_SPECS;
    rate=RATE, div_yield=DIV_YIELD, atm_window=ATM_WINDOW,
    title="$(SYMBOL) smile  $(ts_start) +$(WINDOW_MINUTES)min VWAP  T=$(round(T_d, digits=2))d  spot=$(round(spot_at_window, digits=2))",
)
plot!(plt; size=(1300, 950))

out = joinpath(run_dir, "smile_with_condors.png")
savefig(plt, out)
println("\nSaved figure: $out")
