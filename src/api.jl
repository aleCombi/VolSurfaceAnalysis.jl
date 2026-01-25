# Deribit API Functions
# Direct API calls to Deribit for fetching market data

using HTTP
using JSON3

const DERIBIT_API_BASE = "https://www.deribit.com/api/v2"

"""
    DeliveryPrice

A single delivery price record from Deribit.

# Fields
- `underlying::Underlying`: The underlying asset (BTC or ETH)
- `delivery_price::Float64`: The 30-minute TWAP settlement price
- `timestamp::DateTime`: Delivery timestamp (08:00 UTC on expiry day)
"""
struct DeliveryPrice
    underlying::Underlying
    delivery_price::Float64
    timestamp::DateTime
end

"""
    index_name(underlying::Underlying) -> String

Get the Deribit index name for an underlying.
"""
function index_name(underlying::Underlying)::String
    underlying == BTC && return "btc_usd"
    underlying == ETH && return "eth_usd"
    error("Unknown underlying: $underlying")
end

"""
    fetch_delivery_prices(underlying::Underlying; count::Int=1000) -> Vector{DeliveryPrice}

Fetch historical delivery prices from Deribit API.

The delivery price is the 30-minute TWAP (Time-Weighted Average Price) of the
Deribit Index, calculated from 07:30 to 08:00 UTC on expiration day. This is
the official settlement price for cash-settled options.

# Arguments
- `underlying`: The underlying asset (BTC or ETH)
- `count`: Maximum number of records to fetch (default: 1000)

# Returns
- Vector of `DeliveryPrice` objects, sorted by timestamp descending (most recent first)

# Example
```julia
prices = fetch_delivery_prices(BTC)
prices[1].delivery_price  # Most recent delivery price
prices[1].timestamp       # When the delivery occurred
```
"""
function fetch_delivery_prices(underlying::Underlying; count::Int=1000)::Vector{DeliveryPrice}
    idx = index_name(underlying)
    url = "$DERIBIT_API_BASE/public/get_delivery_prices"

    response = HTTP.get(url, query=["index_name" => idx, "count" => string(count)])

    if response.status != 200
        error("Deribit API error: $(response.status)")
    end

    json = JSON3.read(response.body)

    if !haskey(json, :result) || !haskey(json.result, :data)
        error("Unexpected API response format")
    end

    data = json.result.data
    prices = Vector{DeliveryPrice}(undef, length(data))

    for (i, record) in enumerate(data)
        # API returns: date (YYYY-MM-DD string), delivery_price
        # Settlement is at 08:00 UTC
        date_str = string(record.date)
        ts = DateTime(date_str, "yyyy-mm-dd") + Dates.Hour(8)
        prices[i] = DeliveryPrice(underlying, Float64(record.delivery_price), ts)
    end

    return prices
end

"""
    fetch_delivery_prices_df(underlying::Underlying; count::Int=1000) -> DataFrame

Fetch historical delivery prices as a DataFrame.

# Arguments
- `underlying`: The underlying asset (BTC or ETH)
- `count`: Maximum number of records to fetch (default: 1000)

# Returns
- DataFrame with columns: underlying, delivery_price, timestamp
"""
function fetch_delivery_prices_df(underlying::Underlying; count::Int=1000)::DataFrame
    prices = fetch_delivery_prices(underlying; count=count)

    return DataFrame(
        underlying = [string(p.underlying) for p in prices],
        delivery_price = [p.delivery_price for p in prices],
        timestamp = [p.timestamp for p in prices]
    )
end

"""
    get_delivery_price(underlying::Underlying, expiry::DateTime) -> Union{Float64, Missing}

Get the delivery price for a specific expiry date.

Searches the fetched delivery prices for a matching timestamp. The expiry should
be normalized to 08:00 UTC (Deribit's settlement time).

# Arguments
- `underlying`: The underlying asset (BTC or ETH)
- `expiry`: The expiry datetime (should be 08:00 UTC)

# Returns
- The delivery price if found, `missing` otherwise
"""
function get_delivery_price(underlying::Underlying, expiry::DateTime)::Union{Float64,Missing}
    # Normalize expiry to 08:00 UTC
    expiry_normalized = DateTime(Dates.Date(expiry)) + Dates.Hour(8)

    prices = fetch_delivery_prices(underlying)

    for p in prices
        if p.timestamp == expiry_normalized
            return p.delivery_price
        end
    end

    return missing
end

"""
    save_delivery_prices(path::AbstractString, underlying::Underlying; count::Int=1000)

Fetch delivery prices and save to a parquet file.

# Arguments
- `path`: Output file path (should end in .parquet)
- `underlying`: The underlying asset (BTC or ETH)
- `count`: Maximum number of records to fetch (default: 1000)
"""
function save_delivery_prices(path::AbstractString, underlying::Underlying; count::Int=1000)
    df = fetch_delivery_prices_df(underlying; count=count)
    Parquet2.writefile(path, df)
end
