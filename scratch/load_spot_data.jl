# Helper functions to load spot data from Polygon spot_1min directory

using Dates
using DataFrames
using Parquet2

const SPOT_ROOT = raw"C:\repos\DeribitVols\data\massive_parquet\spot_1min"

"""
    load_spot_prices(date::Date, symbol::String="SPY") -> Dict{DateTime, Float64}

Load spot prices for a given date and symbol.
Returns a dictionary mapping timestamp → spot price (close).
"""
function load_spot_prices(date::Date, symbol::String="SPY")::Dict{DateTime, Float64}
    date_str = Dates.format(date, "yyyy-mm-dd")
    path = joinpath(SPOT_ROOT, "date=$date_str", "symbol=$symbol", "data.parquet")

    if !isfile(path)
        return Dict{DateTime, Float64}()
    end

    try
        df = DataFrame(Parquet2.Dataset(path))
        spot_dict = Dict{DateTime, Float64}()

        for row in eachrow(df)
            ts = DateTime(row.timestamp)
            spot_dict[ts] = Float64(row.close)
        end

        return spot_dict
    catch e
        @warn "Failed to load spot data for $date_str: $e"
        return Dict{DateTime, Float64}()
    end
end

"""
    load_spot_prices_range(dates::Vector{Date}, symbol::String="SPY") -> Dict{DateTime, Float64}

Load spot prices for multiple dates.
"""
function load_spot_prices_range(dates::Vector{Date}, symbol::String="SPY")::Dict{DateTime, Float64}
    all_spots = Dict{DateTime, Float64}()

    for date in dates
        spots = load_spot_prices(date, symbol)
        merge!(all_spots, spots)
    end

    return all_spots
end
