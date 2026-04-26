# Smile snapshot helpers.

using Dates
using Plots
using Random
using Statistics

function aggregate_option_records_vwap(
    records::Vector{OptionRecord},
    ts_anchor::DateTime;
    rate::Real,
    div_yield::Real,
)::Vector{OptionRecord}
    grouped = Dict{Tuple{Float64,DateTime,OptionType},Vector{OptionRecord}}()
    for rec in records
        push!(get!(grouped, (rec.strike, rec.expiry, rec.option_type), OptionRecord[]), rec)
    end

    out = OptionRecord[]
    for ((strike, expiry, opt), group) in grouped
        vols = [coalesce(rec.volume, 0.0) for rec in group]
        weights = sum(vols) > 0 ? vols ./ sum(vols) : fill(1.0 / length(group), length(group))

        closes_usd = [coalesce(rec.mark_price, NaN) * rec.spot for rec in group]
        spots = [rec.spot for rec in group]
        vwap_close = sum(weights .* closes_usd)
        vwap_spot = sum(weights .* spots)

        tau = time_to_expiry(expiry, ts_anchor)
        forward = vwap_spot * exp((rate - div_yield) * tau)
        iv = price_to_iv(vwap_close / forward, forward, strike, tau, opt; r=rate)
        mark_iv = isnan(iv) ? missing : iv * 100.0
        mark_frac = vwap_close / vwap_spot
        bid_frac = sum(weights .* [coalesce(rec.bid_price, NaN) for rec in group])
        ask_frac = sum(weights .* [coalesce(rec.ask_price, NaN) for rec in group])

        push!(out, OptionRecord(
            group[1].instrument_name,
            group[1].underlying,
            expiry,
            strike,
            opt,
            isnan(bid_frac) ? missing : bid_frac,
            isnan(ask_frac) ? missing : ask_frac,
            mark_frac,
            mark_iv,
            missing,
            sum(vols),
            vwap_spot,
            ts_anchor,
        ))
    end

    return out
end

function save_smile_with_condors_snapshot(
    surface::VolatilitySurface,
    expiry::DateTime,
    condor_specs,
    path::AbstractString;
    rate::Real,
    div_yield::Real,
    atm_window::Real,
    title::AbstractString,
    size=(1300, 950),
)
    plt = plot_smile_with_condors(surface, expiry, condor_specs;
        rate=rate,
        div_yield=div_yield,
        atm_window=atm_window,
        title=title,
    )
    plot!(plt; size=size)
    mkpath(dirname(path))
    savefig(plt, path)
    return path
end

function run_smile_with_condors_experiment(;
    output_root::AbstractString,
    symbol::AbstractString="SPY",
    start_date::Date=Date(2024, 1, 1),
    end_date::Date=Date(2025, 6, 30),
    entry_time::Time=Time(13, 0),
    window_minutes::Integer=5,
    target_tenor_days::Real=1.0,
    min_tenor_days::Real=0.5,
    atm_window::Real=0.03,
    rate::Real=0.045,
    div_yield::Real=0.013,
    seed=42,
    spread_lambda::Real=0.7,
    condor_specs=[
        CondorSpec(0.30, 0.10, :firebrick, "30d / 10d"),
        CondorSpec(0.16, 0.05, :darkorchid, "16d / 05d"),
        CondorSpec(0.10, 0.03, :royalblue, "10d / 03d"),
    ],
    max_attempts::Integer=10,
    store=DEFAULT_STORE,
)
    run_dir = make_run_dir(output_root, "smile_with_condors")
    println("Output: $run_dir")

    seed === nothing || Random.seed!(seed)
    all_dates = available_polygon_dates(store, symbol)
    filtered = filter(d -> start_date <= d <= end_date, all_dates)
    isempty(filtered) && error("No $symbol dates in range")

    attempts = 0
    all_recs = OptionRecord[]
    ts_start = nothing
    spot_at_window = NaN
    chosen_date = nothing

    while isempty(all_recs) && attempts < max_attempts
        chosen_date = rand(filtered)
        attempts += 1
        ts_start = et_to_utc(chosen_date, entry_time)
        ts_end = ts_start + Minute(window_minutes - 1)
        window_ts = [ts_start + Minute(i) for i in 0:(window_minutes - 1)]
        spots = read_polygon_spot_prices_for_timestamps(
            polygon_spot_root(store), window_ts; symbol=symbol)
        isempty(spots) && continue

        path = polygon_options_path(store, chosen_date, symbol)
        isfile(path) || continue
        where = "timestamp BETWEEN '$(Dates.format(ts_start, "yyyy-mm-dd HH:MM:SS"))' " *
                "AND '$(Dates.format(ts_end, "yyyy-mm-dd HH:MM:SS"))'"
        all_recs = read_polygon_option_records(
            path, spots; where=where, min_volume=0, warn=false,
            spread_lambda=spread_lambda, rate=rate, div_yield=div_yield)
        spot_at_window = haskey(spots, ts_start) ? spots[ts_start] : first(values(spots))
    end
    isempty(all_recs) && error("No records after $attempts attempts")

    println("\nDate: $chosen_date  spot=$(round(spot_at_window, digits=2))  raw bars: $(length(all_recs))")

    agg = aggregate_option_records_vwap(all_recs, ts_start; rate=rate, div_yield=div_yield)
    surface = build_surface(agg)

    chosen_expiry = nothing
    best_score = Inf
    for expiry in unique(rec.expiry for rec in surface.records)
        tau_days = time_to_expiry(expiry, surface.timestamp) * 365.25
        tau_days >= min_tenor_days || continue
        score = abs(tau_days - target_tenor_days)
        if score < best_score
            best_score = score
            chosen_expiry = expiry
        end
    end
    chosen_expiry === nothing && error("No expiry with T >= $min_tenor_days days")

    tau_days = time_to_expiry(chosen_expiry, surface.timestamp) * 365.25
    println("Chosen expiry: $chosen_expiry  T=$(round(tau_days, digits=2))d")

    out = joinpath(run_dir, "smile_with_condors.png")
    save_smile_with_condors_snapshot(surface, chosen_expiry, condor_specs, out;
        rate=rate,
        div_yield=div_yield,
        atm_window=atm_window,
        title="$(symbol) smile  $(ts_start) +$(window_minutes)min VWAP  T=$(round(tau_days, digits=2))d  spot=$(round(spot_at_window, digits=2))",
    )
    println("\nSaved figure: $out")

    return (
        run_dir=run_dir,
        path=out,
        date=chosen_date,
        timestamp=ts_start,
        expiry=chosen_expiry,
        spot=spot_at_window,
    )
end
