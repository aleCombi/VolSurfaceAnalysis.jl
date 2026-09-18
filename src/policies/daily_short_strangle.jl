# DailyShortStrangle: once a day at a fixed time, a short strangle picked by
# target |delta|, expiring at the first slice on or after t + interval.

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
- `quantity::Int`              -- contracts per leg, a positive integer
"""
struct DailyShortStrangle <: Policy
    underlying      :: Underlying
    entry_time      :: Time
    expiry_interval :: Period
    put_delta       :: Float64
    call_delta      :: Float64
    quantity        :: Int

    function DailyShortStrangle(underlying::Underlying, entry_time::Time,
                                expiry_interval::Period,
                                put_delta::Real, call_delta::Real, quantity::Integer)
        pd = Float64(put_delta);  cd = Float64(call_delta)
        0.0 < pd < 1.0 || throw(ArgumentError("put_delta must be in (0, 1), got $put_delta"))
        0.0 < cd < 1.0 || throw(ArgumentError("call_delta must be in (0, 1), got $call_delta"))
        quantity > 0   || throw(ArgumentError("quantity must be positive, got $quantity"))
        expiry_interval > Day(0) ||
            throw(ArgumentError("expiry_interval must be positive, got $expiry_interval"))
        new(underlying, entry_time, expiry_interval, pd, cd, Int(quantity))
    end
end

"""
    DailyShortStrangle(; underlying, entry_time, expiry_interval,
                       put_delta, call_delta, quantity=1)

Keyword-argument constructor. `quantity` defaults to one contract per leg.
"""
DailyShortStrangle(; underlying::Underlying, entry_time::Time,
                   expiry_interval::Period,
                   put_delta::Real, call_delta::Real, quantity::Integer=1) =
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

# Nearest entry in `sorted_strikes` to `K`; `nothing` when empty; ties go to
# the lower strike. Snaps to the chain's strikes of the leg's type, not the
# slice's: a slice keeps one side per strike, and a fill matches both.
function _snap_to_sorted(sorted_strikes::Vector{Float64},
                         K::Float64)::Union{Float64,Nothing}
    isempty(sorted_strikes) && return nothing
    K <= sorted_strikes[1]   && return sorted_strikes[1]
    K >= sorted_strikes[end] && return sorted_strikes[end]
    i = searchsortedlast(sorted_strikes, K)
    return (K - sorted_strikes[i]) <= (sorted_strikes[i+1] - K) ?
           sorted_strikes[i] : sorted_strikes[i+1]
end

declared_underlyings(p::DailyShortStrangle) = (p.underlying,)

"""
    tick_times(p::DailyShortStrangle, data, from, to) -> Vector{DateTime}

Emit one candidate timestamp per calendar day in `[from, to]`, at the
policy's `entry_time`. Candidates that fall outside the data's chain
coverage produce `Order[]` inside `decide` (no surface at that
instant), so non-trading days (weekends / holidays) are tolerated
without consulting the data's timestamps first.
"""
function tick_times(p::DailyShortStrangle, ::MarketData,
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

"""
    decide(p::DailyShortStrangle, t, data::TimeCut, book::Book) -> Vector{Order}

One `Order(:daily_short_strangle, [short put, short call])` with `Open`
legs of `p.quantity` contracts each at the entry tick, or `Order[]` when
the gate does not fire, no surface or chain is visible, no expiry lies
on or after `t + expiry_interval`, or either wing cannot be placed. The
book is read for nothing: this policy only opens, and lifecycle settles
its lots.
"""
function decide(p::DailyShortStrangle, t::DateTime,
                data::TimeCut,
                ::Book)::Vector{Order}
    Time(t) == p.entry_time || return Order[]                     # cheap gate
    surface = only_or_missing(at(data, VolatilitySurface, p.underlying, t))
    ismissing(surface) && return Order[]
    expiry = _first_expiry_on_or_after(surface, t + p.expiry_interval)
    expiry === nothing && return Order[]
    chain = at(data, OptionQuote, p.underlying, t)
    isempty(chain) && return Order[]

    K_put_raw  = invert_delta(surface, expiry, Put,  p.put_delta)
    K_call_raw = invert_delta(surface, expiry, Call, p.call_delta)
    # One wing failing skips the entry: a one-legged strangle is another structure.
    (K_put_raw === nothing || K_call_raw === nothing) && return Order[]

    put_strikes  = _quoted_strikes(chain, expiry, p.underlying, Put)
    call_strikes = _quoted_strikes(chain, expiry, p.underlying, Call)
    K_put  = _snap_to_sorted(put_strikes,  K_put_raw)
    K_call = _snap_to_sorted(call_strikes, K_call_raw)
    (K_put === nothing || K_call === nothing) && return Order[]

    return Order[Order(:daily_short_strangle, [
        Leg(ContractKey(p.underlying, K_put,  expiry, Put),  Short, p.quantity, Open),
        Leg(ContractKey(p.underlying, K_call, expiry, Call), Short, p.quantity, Open),
    ])]
end
