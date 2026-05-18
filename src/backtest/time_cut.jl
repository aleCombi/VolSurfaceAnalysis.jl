# TimeCutModelDataSource: composition wrapper that masks every accessor at a
# `cutoff` timestamp. The backtest engine builds one per tick and hands it to
# `decide`, giving strategies a no-lookahead boundary through the public
# accessor interface.

"""
    TimeCutModelDataSource

Wraps a `ModelDataSource` with a `cutoff::DateTime`. Every accessor
returns `nothing` / `missing` (matching the inner type's failure mode)
when called for `ts > cutoff`; otherwise it forwards to the inner source.

Not a subtype of `ModelDataSource` (Julia structs are final). Strategies
declare the cut wrapper as their data parameter explicitly, which makes
no-lookahead a property of the supported accessor interface rather than a
call-site convention.

# Fields
- `inner::ModelDataSource` -- the unrestricted source.
- `cutoff::DateTime`       -- inclusive upper bound on visible timestamps.
"""
struct TimeCutModelDataSource
    inner::ModelDataSource
    cutoff::DateTime
end

available_timestamps(c::TimeCutModelDataSource, from::DateTime, to::DateTime) =
    available_timestamps(c.inner, from, min(to, c.cutoff))

get_chain(c::TimeCutModelDataSource, ts::DateTime) =
    ts <= c.cutoff ? get_chain(c.inner, ts) : nothing

get_spot(c::TimeCutModelDataSource, ts::DateTime) =
    ts <= c.cutoff ? get_spot(c.inner, ts) : missing

get_surface(c::TimeCutModelDataSource, ts::DateTime) =
    ts <= c.cutoff ? get_surface(c.inner, ts) : nothing

# Rate and div curves are math objects, not historical observations -- a
# curve built at backtest-setup time encodes "what the rate will be at ts"
# as known then, and evaluating it at any `ts` (past or future relative to
# `cutoff`) is a legitimate forward query rather than lookahead. Passthrough.
get_rate(c::TimeCutModelDataSource, ts::DateTime)::Float64 = get_rate(c.inner, ts)
get_div(c::TimeCutModelDataSource,  ts::DateTime)::Float64 = get_div(c.inner, ts)

clear_cache!(c::TimeCutModelDataSource) = (clear_cache!(c.inner); c)
