# Local Data Store
# Centralized path derivation for the options-collector data layout

using Dates

"""
    LocalDataStore

Represents the local data directory layout produced by options-collector.
All path derivation lives here — change the layout in one place.

# Fields
- `root::String`: Base path (points to `options-collector/data/`)

# Layout (relative to root)
```
massive_flatfiles/
  minute_aggs/date={yyyy-mm-dd}/underlying={SYM}/data.parquet   ← Polygon options
  spot_1min/  date={yyyy-mm-dd}/symbol={SYM}/data.parquet       ← Polygon spot
deribit_local/
  history/vols_{yyyymmdd}.parquet                                ← Deribit daily
  recent/vols_current.parquet                                    ← Deribit live
  delivery_prices/delivery_prices.parquet                        ← Deribit settlement
```
"""
struct LocalDataStore
    root::String
end

# ============================================================================
# Symbol normalization helper
# ============================================================================

_sym(symbol::Underlying) = ticker(symbol)
_sym(symbol::AbstractString) = uppercase(String(symbol))

# ============================================================================
# Root accessors (for functions like read_polygon_spot_prices_for_timestamps
# that take a directory root and construct date/symbol paths internally)
# ============================================================================

"""Return the minute_aggs root (parent of `date=…/underlying=…/data.parquet`)."""
polygon_options_root(store::LocalDataStore)::String =
    joinpath(store.root, "massive_flatfiles", "minute_aggs")

"""Return the spot_1min root (parent of `date=…/symbol=…/data.parquet`)."""
polygon_spot_root(store::LocalDataStore)::String =
    joinpath(store.root, "massive_flatfiles", "spot_1min")

# ============================================================================
# Path derivation
# ============================================================================

"""
    polygon_options_path(store, date, symbol) -> String

Full path to the Polygon options minute-bar parquet for a given date and symbol.
"""
function polygon_options_path(
    store::LocalDataStore,
    date::Date,
    symbol::Union{Underlying, AbstractString}
)::String
    date_str = Dates.format(date, "yyyy-mm-dd")
    joinpath(polygon_options_root(store),
             "date=$date_str", "underlying=$(_sym(symbol))", "data.parquet")
end

"""
    polygon_spot_path(store, date, symbol) -> String

Full path to the Polygon spot 1-minute parquet for a given date and symbol.
"""
function polygon_spot_path(
    store::LocalDataStore,
    date::Date,
    symbol::Union{Underlying, AbstractString}
)::String
    date_str = Dates.format(date, "yyyy-mm-dd")
    joinpath(polygon_spot_root(store),
             "date=$date_str", "symbol=$(_sym(symbol))", "data.parquet")
end

"""
    deribit_history_path(store, date) -> String

Full path to the Deribit daily history parquet for a given date.
"""
function deribit_history_path(store::LocalDataStore, date::Date)::String
    date_str = Dates.format(date, "yyyymmdd")
    joinpath(store.root, "deribit_local", "history", "vols_$date_str.parquet")
end

"""
    deribit_delivery_path(store) -> String

Full path to the Deribit delivery prices parquet.
"""
function deribit_delivery_path(store::LocalDataStore)::String
    joinpath(store.root, "deribit_local", "delivery_prices", "delivery_prices.parquet")
end

# ============================================================================
# Date scanning
# ============================================================================

"""
    available_polygon_dates(store, symbol) -> Vector{Date}

Return sorted vector of dates for which Polygon options data exists for the
given symbol.
"""
function available_polygon_dates(
    store::LocalDataStore,
    symbol::Union{Underlying, AbstractString}
)::Vector{Date}
    root = polygon_options_root(store)
    isdir(root) || return Date[]

    sym = _sym(symbol)
    dates = Date[]
    for entry in readdir(root)
        m = match(r"^date=(\d{4}-\d{2}-\d{2})$", entry)
        m === nothing && continue
        path = joinpath(root, entry, "underlying=$sym", "data.parquet")
        isfile(path) && push!(dates, Date(m[1]))
    end
    return sort(dates)
end

"""
    available_deribit_dates(store) -> Vector{Date}

Return sorted vector of dates for which Deribit history snapshots exist.
"""
function available_deribit_dates(store::LocalDataStore)::Vector{Date}
    hist_root = joinpath(store.root, "deribit_local", "history")
    isdir(hist_root) || return Date[]

    dates = Date[]
    for entry in readdir(hist_root)
        m = match(r"^vols_(\d{4})(\d{2})(\d{2})\.parquet$", entry)
        m === nothing && continue
        push!(dates, Date(parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3])))
    end
    return sort(dates)
end

# ============================================================================
# Default store
# ============================================================================

"""Default LocalDataStore pointing to the options-collector data directory."""
const DEFAULT_STORE = LocalDataStore(raw"C:\repos\options-collector\data")
