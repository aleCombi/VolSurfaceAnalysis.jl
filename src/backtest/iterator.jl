# Surface Iterator
# Lazy iteration over volatility surfaces for backtesting

using Dates

"""
    SurfaceIterator

Lazy iterator over volatility surfaces in chronological order.
Loads data on-demand from LocalDataStore to minimize memory usage.

# Fields
- `store::LocalDataStore`: Data source
- `underlying::Underlying`: Asset to iterate (BTC or ETH)
- `timestamps::Vector{DateTime}`: Ordered timestamps to iterate
- `resolution::Period`: Time resolution for grouping snapshots
- `_cache::Dict{DateTime, Vector{VolRecord}}`: Optional LRU cache

# Example
```julia
store = LocalDataStore("data/")
iter = SurfaceIterator(store, BTC; start_date=Date(2024,1,1), end_date=Date(2024,1,7))

for surface in iter
    println("Spot: \$(surface.spot) at \$(surface.timestamp)")
end
```
"""
mutable struct SurfaceIterator
    store::LocalDataStore
    underlying::Underlying
    timestamps::Vector{DateTime}
    resolution::Period
    _records_by_time::Dict{DateTime, Vector{VolRecord}}
    _loaded::Bool
end

"""
    SurfaceIterator(store, underlying; start_date=nothing, end_date=nothing, resolution=Hour(1))

Create a lazy iterator over vol surfaces.

# Arguments
- `store::LocalDataStore`: Data source
- `underlying::Underlying`: BTC or ETH
- `start_date::Date`: Start of iteration range (optional)
- `end_date::Date`: End of iteration range (optional)  
- `resolution::Period`: Time resolution (default: Hour(1))
"""
function SurfaceIterator(store::LocalDataStore, underlying::Underlying;
                         start_date::Union{Date,Nothing}=nothing,
                         end_date::Union{Date,Nothing}=nothing,
                         resolution::Period=Hour(1))
    # Load all records for the underlying (we need timestamps)
    records = if start_date !== nothing && end_date !== nothing
        load_range(store, start_date, end_date; underlying=underlying)
    else
        load_all(store; underlying=underlying)
    end
    
    # Group by rounded timestamp
    records_by_time = Dict{DateTime, Vector{VolRecord}}()
    for r in records
        ts = floor(r.timestamp, resolution)
        if !haskey(records_by_time, ts)
            records_by_time[ts] = VolRecord[]
        end
        push!(records_by_time[ts], r)
    end
    
    timestamps = sort(collect(keys(records_by_time)))
    
    # Apply date filters to timestamps
    if start_date !== nothing
        start_dt = DateTime(start_date)
        timestamps = filter(ts -> ts >= start_dt, timestamps)
    end
    if end_date !== nothing
        end_dt = DateTime(end_date) + Day(1) - Millisecond(1)
        timestamps = filter(ts -> ts <= end_dt, timestamps)
    end
    
    return SurfaceIterator(store, underlying, timestamps, resolution, records_by_time, true)
end

# ============================================================================
# Iteration Protocol
# ============================================================================

Base.length(iter::SurfaceIterator) = length(iter.timestamps)
Base.eltype(::Type{SurfaceIterator}) = VolatilitySurface

function Base.iterate(iter::SurfaceIterator)
    isempty(iter.timestamps) && return nothing
    return _get_surface(iter, 1), 2
end

function Base.iterate(iter::SurfaceIterator, state::Int)
    state > length(iter.timestamps) && return nothing
    return _get_surface(iter, state), state + 1
end

function _get_surface(iter::SurfaceIterator, idx::Int)::VolatilitySurface
    ts = iter.timestamps[idx]
    records = iter._records_by_time[ts]
    return build_surface(records)
end

# ============================================================================
# Random Access
# ============================================================================

"""
    surface_at(iter::SurfaceIterator, timestamp::DateTime) -> Union{VolatilitySurface, Nothing}

Get the surface at a specific timestamp. Returns nothing if not found.
The timestamp is rounded to the iterator's resolution before lookup.
"""
function surface_at(iter::SurfaceIterator, timestamp::DateTime)::Union{VolatilitySurface, Nothing}
    rounded_ts = floor(timestamp, iter.resolution)
    
    if haskey(iter._records_by_time, rounded_ts)
        return build_surface(iter._records_by_time[rounded_ts])
    end
    
    return nothing
end

"""
    surface_at(iter::SurfaceIterator, idx::Int) -> VolatilitySurface

Get the surface at a specific index (1-based).
"""
function surface_at(iter::SurfaceIterator, idx::Int)::VolatilitySurface
    1 <= idx <= length(iter.timestamps) || error("Index $idx out of bounds (1:$(length(iter.timestamps)))")
    return _get_surface(iter, idx)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    first_timestamp(iter::SurfaceIterator) -> DateTime

Get the first timestamp in the iteration range.
"""
first_timestamp(iter::SurfaceIterator) = first(iter.timestamps)

"""
    last_timestamp(iter::SurfaceIterator) -> DateTime

Get the last timestamp in the iteration range.
"""
last_timestamp(iter::SurfaceIterator) = last(iter.timestamps)

"""
    date_range(iter::SurfaceIterator) -> Tuple{Date, Date}

Get the date range covered by this iterator.
"""
function date_range(iter::SurfaceIterator)::Tuple{Date, Date}
    return (Date(first_timestamp(iter)), Date(last_timestamp(iter)))
end

"""
    timestamps(iter::SurfaceIterator) -> Vector{DateTime}

Get all timestamps in the iterator.
"""
timestamps(iter::SurfaceIterator) = iter.timestamps

"""
    filter_timestamps(iter::SurfaceIterator, predicate::Function) -> SurfaceIterator

Create a new iterator with filtered timestamps.

# Example
```julia
# Only daytime hours
daytime_iter = filter_timestamps(iter, ts -> 8 <= hour(ts) <= 20)
```
"""
function filter_timestamps(iter::SurfaceIterator, predicate::Function)::SurfaceIterator
    new_timestamps = filter(predicate, iter.timestamps)
    new_records = Dict(ts => iter._records_by_time[ts] for ts in new_timestamps)
    return SurfaceIterator(iter.store, iter.underlying, new_timestamps, 
                           iter.resolution, new_records, true)
end
