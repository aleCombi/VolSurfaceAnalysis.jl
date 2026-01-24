# Plot ATM vol term structures (bid/ask/mark) at 1-hour intervals for a single day
#
# Usage:
#   julia --project=. atm_term_structure_day.jl <path> [underlying] [date]
#
# Examples:
#   julia --project=. atm_term_structure_day.jl data/ BTC 2024-12-15

using VolSurfaceAnalysis
using Dates
using Plots

const OUTPUT_DIR = joinpath(@__DIR__, "plots")

function find_parquet_files(path::String)
    files = String[]
    if isfile(path) && endswith(path, ".parquet")
        push!(files, path)
    elseif isdir(path)
        for (root, _, filenames) in walkdir(path)
            for f in filenames
                endswith(f, ".parquet") && push!(files, joinpath(root, f))
            end
        end
    end
    return files
end

function main(args)
    isempty(args) && error("Usage: julia atm_term_structure_day.jl <path> [underlying] [date]")

    data_path = args[1]
    underlying_filter = length(args) >= 2 ? uppercase(args[2]) : "BTC"
    target_date = length(args) >= 3 ? Date(args[3]) : nothing

    mkpath(OUTPUT_DIR)

    files = find_parquet_files(data_path)
    println("Found $(length(files)) parquet file(s)")

    # Read and filter by underlying
    all_records = VolRecord[]
    for f in files
        records = read_vol_records(f)
        append!(all_records, filter(r -> string(r.underlying) == underlying_filter, records))
    end
    println("$underlying_filter records: $(length(all_records))")

    # Group by hour
    by_hour = split_by_timestamp(all_records, Hour(1))
    timestamps = sort(collect(keys(by_hour)))
    println("Total hourly snapshots: $(length(timestamps))")

    # Pick target date (first available if not specified)
    if target_date === nothing
        target_date = Date(first(timestamps))
        println("No date specified, using first available: $target_date")
    end

    # Filter to just the 24 hours of the target date
    day_timestamps = filter(ts -> Date(ts) == target_date, timestamps)
    println("Snapshots for $target_date: $(length(day_timestamps))")

    isempty(day_timestamps) && error("No data found for date $target_date")

    for ts in day_timestamps
        recs = by_hour[ts]
        try
            term_struct = atm_term_structure(recs)
            tenor_days = term_struct.tenors .* 365.25

            # Convert vols to percentages, handling missing values
            bid_pct = [ismissing(v) ? NaN : v * 100 for v in term_struct.bid_vols]
            ask_pct = [ismissing(v) ? NaN : v * 100 for v in term_struct.ask_vols]
            mark_pct = [ismissing(v) ? NaN : v * 100 for v in term_struct.mark_vols]

            p = plot(
                xlabel = "Tenor (days)",
                ylabel = "ATM Vol (%)",
                title = "$(term_struct.underlying) ATM Term Structure — $(Dates.format(ts, "yyyy-mm-dd HH:MM"))",
                legend = :topright,
                size = (1200, 600),
                xscale = :log10,
                xticks = begin
                    ui = sort(unique(round.(Int, tenor_days)))
                    filter!(x -> x > 0, ui) # Remove 0 from log scale
                    if length(ui) > 12
                        indices = range(1, length(ui), length=12)
                        ui = ui[round.(Int, indices)]
                    end
                    (ui, string.(ui))
                end,
                xlims = (0.1, maximum(tenor_days) * 1.1),
                formatter = x -> string(round(Int, x)),
                bottom_margin = 10Plots.mm,
                tickfontsize = 7
            )

            # Plot bid (green), ask (red), mark (blue)
            plot!(p, tenor_days, bid_pct,
                label = "Bid",
                color = :green,
                marker = :circle,
                linewidth = 2,
                markersize = 4
            )
            plot!(p, tenor_days, ask_pct,
                label = "Ask",
                color = :red,
                marker = :circle,
                linewidth = 2,
                markersize = 4
            )
            plot!(p, tenor_days, mark_pct,
                label = "Mark",
                color = :blue,
                marker = :circle,
                linewidth = 2,
                markersize = 4
            )

            filename = "atm_$(lowercase(underlying_filter))_$(Dates.format(ts, "yyyymmdd_HHMM")).png"
            savefig(p, joinpath(OUTPUT_DIR, filename))
            println("  Saved: $filename")
        catch e
            @warn "Failed at $ts" exception=e
        end
    end

    println("\nPlots saved to: $OUTPUT_DIR")
end

main(ARGS)
