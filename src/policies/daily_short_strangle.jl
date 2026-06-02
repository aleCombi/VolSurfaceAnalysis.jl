# Daily short strangle policy.
#
# Smallest honest concrete policy: once a day at a fixed wall-clock time,
# open a short strangle on `underlying` whose two legs are picked by target
# |delta| (one put OTM, one call OTM), expiring at the first available
# slice on or after `t + expiry_interval`. Fixed quantity per leg.
#
# Engineering notes:
# - The cheap gate `Time(t) == entry_time` runs before any surface lookup;
#   the engine fires `decide` on every available timestamp.
# - Continuous `invert_delta` returns a target strike inside the slice's
#   observed strike bracket; we then snap to the nearest strike actually
#   in the slice, because `resolve_quote` in the engine requires an exact
#   match against the chain (and `slice.strikes` is a subset of chain
#   strikes by construction in `build_surface`).
# - If either leg's `invert_delta` returns `nothing` (target outside the
#   observed-delta bracket on that wing), we return `Trade[]` rather than
#   trading the other wing alone -- a one-legged strangle is a different
#   structure.

using Dates

"""
    DailyShortStrangle(underlying, entry_time, expiry_interval,
                      put_delta, call_delta, quantity)

Short OTM put + short OTM call, opened once per day at `entry_time`.
The two legs are picked by target absolute delta (`put_delta`,
`call_delta`); the expiry is the first surface slice on or after
`t + expiry_interval`.

# Fields
- `underlying::Underlying`
- `entry_time::Time`           -- wall-clock entry time (gate inside `decide`)
- `expiry_interval::Period`    -- minimum DTE from entry (e.g. `Day(1)`)
- `put_delta::Float64`         -- target `|Δ|` for the short put leg, in `(0, 1)`
- `call_delta::Float64`        -- target `|Δ|` for the short call leg, in `(0, 1)`
- `quantity::Float64`          -- contracts per leg, `> 0`
"""
struct DailyShortStrangle <: Policy
    underlying      :: Underlying
    entry_time      :: Time
    expiry_interval :: Period
    put_delta       :: Float64
    call_delta      :: Float64
    quantity        :: Float64

    function DailyShortStrangle(underlying::Underlying, entry_time::Time,
                                expiry_interval::Period,
                                put_delta::Real, call_delta::Real, quantity::Real)
        pd = Float64(put_delta);  cd = Float64(call_delta);  q = Float64(quantity)
        0.0 < pd < 1.0 || throw(ArgumentError("put_delta must be in (0, 1), got $put_delta"))
        0.0 < cd < 1.0 || throw(ArgumentError("call_delta must be in (0, 1), got $call_delta"))
        q > 0.0        || throw(ArgumentError("quantity must be positive, got $quantity"))
        expiry_interval > Day(0) ||
            throw(ArgumentError("expiry_interval must be positive, got $expiry_interval"))
        new(underlying, entry_time, expiry_interval, pd, cd, q)
    end
end

"""
    DailyShortStrangle(; underlying, entry_time, expiry_interval,
                       put_delta, call_delta, quantity=1.0)

Keyword-argument constructor. `quantity` defaults to one contract per leg.
"""
DailyShortStrangle(; underlying::Underlying, entry_time::Time,
                   expiry_interval::Period,
                   put_delta::Real, call_delta::Real, quantity::Real=1.0) =
    DailyShortStrangle(underlying, entry_time, expiry_interval,
                       put_delta, call_delta, quantity)

# First expiry in `surface` at or after `target`. `nothing` if none.
function _first_expiry_on_or_after(surface::VolatilitySurface,
                                   target::DateTime)::Union{DateTime,Nothing}
    for e in expiries(surface)
        e >= target && return e
    end
    return nothing
end

# Strikes in `chain` for which a quote of `(underlying, expiry, otype)` exists.
# Returned sorted ascending and de-duplicated.
function _quoted_strikes(chain::AbstractVector{OptionQuote}, expiry::DateTime,
                         underlying::Underlying, otype::OptionType)::Vector{Float64}
    out = Float64[]
    for q in chain
        q.option_type == otype     || continue
        q.expiry      == expiry    || continue
        q.underlying  == underlying || continue
        push!(out, q.strike)
    end
    sort!(out)
    return unique!(out)
end

# Nearest entry in `sorted_strikes` to `K`. Empty vector returns `nothing`.
# Sorted ascending; ties to the lower strike (deterministic; symmetric grids
# don't care). Used to snap a continuous `invert_delta` target to a strike
# that actually carries a quote of the required option_type -- the chain is
# authoritative because `slice.strikes` mixes Put-origin and Call-origin
# strikes (whichever side `build_surface._pick_otm` retained) and the engine's
# `resolve_quote` matches on both strike and option_type.
function _snap_to_sorted(sorted_strikes::Vector{Float64},
                         K::Float64)::Union{Float64,Nothing}
    isempty(sorted_strikes) && return nothing
    K <= sorted_strikes[1]   && return sorted_strikes[1]
    K >= sorted_strikes[end] && return sorted_strikes[end]
    i = searchsortedlast(sorted_strikes, K)
    return (K - sorted_strikes[i]) <= (sorted_strikes[i+1] - K) ?
           sorted_strikes[i] : sorted_strikes[i+1]
end

"""
    tick_times(p::DailyShortStrangle, source, from, to) -> Vector{DateTime}

Emit one candidate timestamp per calendar day in `[from, to]`, at the
policy's `entry_time`. Candidates that fall outside the data's chain
coverage produce `Trade[]` inside `decide` (`get_surface` returns
`nothing`), so non-trading days (weekends / holidays) are tolerated
without consulting `available_timestamps` first.
"""
function tick_times(p::DailyShortStrangle, ::ModelDataSource,
                    from::DateTime, to::DateTime)::Vector{DateTime}
    out = DateTime[]
    d = Date(from)
    d_end = Date(to)
    while d <= d_end
        ts = DateTime(d, p.entry_time)
        from <= ts <= to && push!(out, ts)
        d += Day(1)
    end
    out
end

function decide(p::DailyShortStrangle, t::DateTime,
                data::TimeCutModelDataSource,
                ::AbstractVector{Position})::Vector{Trade}
    Time(t) == p.entry_time || return Trade[]                     # cheap gate
    surface = get_surface(data, t)
    surface === nothing && return Trade[]
    expiry = _first_expiry_on_or_after(surface, t + p.expiry_interval)
    expiry === nothing && return Trade[]
    chain = get_chain(data, t)
    chain === nothing && return Trade[]

    K_put_raw  = invert_delta(surface, expiry, Put,  p.put_delta)
    K_call_raw = invert_delta(surface, expiry, Call, p.call_delta)
    (K_put_raw === nothing || K_call_raw === nothing) && return Trade[]

    put_strikes  = _quoted_strikes(chain, expiry, p.underlying, Put)
    call_strikes = _quoted_strikes(chain, expiry, p.underlying, Call)
    K_put  = _snap_to_sorted(put_strikes,  K_put_raw)
    K_call = _snap_to_sorted(call_strikes, K_call_raw)
    (K_put === nothing || K_call === nothing) && return Trade[]

    return Trade[
        Trade(p.underlying, K_put,  expiry, Put;  direction=-1, quantity=p.quantity),
        Trade(p.underlying, K_call, expiry, Call; direction=-1, quantity=p.quantity),
    ]
end
