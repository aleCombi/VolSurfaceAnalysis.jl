# ModelDataSource: model-facing data wrapper.
#
# Composes a raw chain DataSource, a spot DataSource (often the same),
# and pre-built rate/div Curve objects. Exposes a small accessor surface
# (get_chain, get_spot, get_rate, get_div, get_surface) for downstream
# code; caches built surfaces per timestamp.

"""
    ModelDataSource

Holds the references needed to answer per-timestamp model queries:

- `chain_source` -- a `DataSource` answering `get_chain(ds, ts)`.
- `spot_source`  -- a `DataSource` answering `get_spot(ds, ts)`.
- `rate`, `div`  -- `Curve`s evaluated at `ts`.

`get_surface(mds, ts)` builds a `VolatilitySurface` from these inputs
and caches the result (including `nothing` outcomes, so missing chains
and missing spots are not retried).
"""
struct ModelDataSource
    chain_source  :: DataSource
    spot_source   :: DataSource
    rate          :: Curve
    div           :: Curve
    surface_cache :: Dict{DateTime,Union{VolatilitySurface,Nothing}}
end

"""
    ModelDataSource(chain_source; rate, div, spot_source=chain_source)

Construct a `ModelDataSource`. By default `spot_source` is the same
`DataSource` as `chain_source` (the common Polygon-style case).
Supply a different `spot_source` for split-vendor setups (e.g. SPX
cash spot with SPY-option chains).
"""
function ModelDataSource(
    chain_source::DataSource;
    rate::Curve,
    div::Curve,
    spot_source::DataSource = chain_source,
)
    ModelDataSource(
        chain_source, spot_source, rate, div,
        Dict{DateTime,Union{VolatilitySurface,Nothing}}(),
    )
end

available_timestamps(mds::ModelDataSource, from::DateTime, to::DateTime) =
    available_timestamps(mds.chain_source, from, to)

get_chain(mds::ModelDataSource, ts::DateTime) =
    get_chain(mds.chain_source, ts)

get_spot(mds::ModelDataSource, ts::DateTime) =
    get_spot(mds.spot_source, ts)

get_rate(mds::ModelDataSource, ts::DateTime)::Float64 = mds.rate(ts)
get_div(mds::ModelDataSource,  ts::DateTime)::Float64 = mds.div(ts)

"""
    get_surface(mds, ts) -> Union{VolatilitySurface, Nothing}

Build (and cache) the vol surface at `ts`. Returns `nothing` and
caches it when the chain is absent, the spot is missing, or every
expiry in the chain has passed.
"""
function get_surface(mds::ModelDataSource, ts::DateTime)::Union{VolatilitySurface,Nothing}
    haskey(mds.surface_cache, ts) && return mds.surface_cache[ts]
    chain = get_chain(mds.chain_source, ts)
    spot  = get_spot(mds.spot_source, ts)
    surf = if chain === nothing || isempty(chain) || ismissing(spot)
        nothing
    else
        build_surface(chain, Float64(spot), mds.rate(ts), mds.div(ts))
    end
    mds.surface_cache[ts] = surf
    return surf
end

"""
    clear_cache!(mds::ModelDataSource)

Empty the surface cache and forward to the underlying sources. Useful
between sweeps when long backtests would otherwise grow the cache
without bound.
"""
function clear_cache!(mds::ModelDataSource)
    empty!(mds.surface_cache)
    clear_cache!(mds.chain_source)
    # spot_source may be a different DataSource; clear it too. If it is
    # the same object, clear_cache! is idempotent.
    mds.spot_source === mds.chain_source || clear_cache!(mds.spot_source)
    return mds
end
