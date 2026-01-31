# Test Polygon Data Loading
# Validates that Polygon minute aggregation data can be loaded and converted

using Pkg
Pkg.activate(@__DIR__)

using VolSurfaceAnalysis
using Dates

# Path to sample Polygon data
const POLYGON_PATH = raw"C:\repos\DeribitVols\data\massive_parquet\minute_aggs\date=2024-01-29\underlying=SPY\data.parquet"

function main()
    println("=" ^ 80)
    println("POLYGON DATA LOADING TEST")
    println("=" ^ 80)
    println()

    # Load Polygon trade data
    println("Loading Polygon data from:")
    println("  $POLYGON_PATH")
    println()

    records = load_polygon_data(POLYGON_PATH; min_volume=5)

    println("Loaded $(length(records)) VolRecords with synthetic bid/ask spreads")
    println()

    # Group by timestamp
    by_timestamp = split_by_timestamp(records)
    timestamps = sort(collect(keys(by_timestamp)))

    println("Time range:")
    println("  First: $(first(timestamps))")
    println("  Last:  $(last(timestamps))")
    println("  Unique timestamps: $(length(timestamps))")
    println()

    # Build a sample surface
    if !isempty(timestamps)
        sample_ts = timestamps[div(length(timestamps), 2)]
        sample_records = by_timestamp[sample_ts]

        println("Sample surface at $(sample_ts):")
        println("  Records: $(length(sample_records))")

        if length(sample_records) > 0
            first_rec = sample_records[1]
            println("  Spot: $(round(first_rec.underlying_price, digits=2))")

            # Count calls vs puts
            n_calls = count(r -> r.option_type == Call, sample_records)
            n_puts = count(r -> r.option_type == Put, sample_records)
            println("  Calls: $n_calls | Puts: $n_puts")

            # Build surface
            try
                surface = build_surface(sample_records)
                println("  Surface points: $(length(surface.points))")

                # Show some example points
                println()
                println("Sample VolPoints:")
                for i in 1:min(5, length(surface.points))
                    pt = surface.points[i]
                    bid_str = pt.bid_vol === missing ? "missing" : "$(round(pt.bid_vol * 100, digits=2))%"
                    ask_str = pt.ask_vol === missing ? "missing" : "$(round(pt.ask_vol * 100, digits=2))%"
                    println("    log_m=$(round(pt.log_moneyness, digits=4)) " *
                           "τ=$(round(pt.τ, digits=4)) " *
                           "vol=$(round(pt.vol * 100, digits=2))% " *
                           "bid_vol=$bid_str " *
                           "ask_vol=$ask_str")
                end

                # Check bid/ask vol computation
                println()
                println("Checking bid/ask vol computation (should be missing if low/high IV fails):")
                non_missing_bid = count(!ismissing, [pt.bid_vol for pt in surface.points])
                non_missing_ask = count(!ismissing, [pt.ask_vol for pt in surface.points])
                println("  Non-missing bid_vol: $non_missing_bid / $(length(surface.points))")
                println("  Non-missing ask_vol: $non_missing_ask / $(length(surface.points))")

            catch e
                println("  ERROR building surface: $e")
            end
        end
    end

    println()
    println("=" ^ 80)
    println("TEST COMPLETE")
    println("=" ^ 80)
    println()
    println("⚠️  REMINDER: Bid/ask spreads are SYNTHETIC (conservative OHLC approximation)")
    println("   bid = low, ask = high from minute bar")
    println("   This is pessimistic and may underestimate strategy performance")
end

main()
