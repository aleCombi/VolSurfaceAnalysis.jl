# Building the marked curve: the one place in `metrics` that reads market
# data. Everything else in the module is a pure function over a
# `MarkedCurve` or over per-trade dollars.
#
# The identity the builder rests on. At an instant `t`,
#
#   marked profit = realised profit + unallocated fees + unrealised profit
#
# and the first two terms are exactly the ledger's cash plus the cost
# basis of whatever is still open:
#
#   cash(t) = realised + unallocated - SUM side_sign * unit_price * mult * qty
#   unrealised = SUM side_sign * (mark - unit_price) * mult * qty
#
# so the cost basis cancels and
#
#   marked profit = cash(t) + SUM side_sign * mark * mult * qty
#
# That is why cash alone overstates a short book -- the opening premium is
# a receipt against a liability, and the second term is that liability --
# and why a flat book needs no market data at all: the sum is empty and
# the marked profit is the realised total.

# Whether the map has a provider for a kind at all. `entry` errors when it
# does not, and an experiment configuring no surface is an ordinary
# configuration, not a failure: the quote mid is then the only mark source.
_has_provider(m::MarketData, ::Type{R}) where {R} = any(p -> kind(p) === R, m.entries)
_has_provider(c::TimeCut, ::Type{R}) where {R} = _has_provider(c.inner, R)

# The surface fallback, or `nothing` when it cannot answer. Every way it
# can fail to produce a price for *this* contract is a fallback that did
# not fire, and the caller turns that into one named failure; a fault that
# is not about this contract's price is rethrown.
#
# **The surface must be stamped at `t` exactly.** `asof` walks backward, so
# it can hand back a surface built at an earlier instant -- and a surface
# carries its own `spot` and its slices their own cached `tau`, so pricing
# off a stale one values the contract at *that* instant, not at `t`. Marking
# successive sessions from one stale surface is carrying a price forward,
# which is the thing the curve refuses to do (see `metrics.md`, "What an
# unmarkable point does"): it would leave `n_unmarked` at zero while the
# profit never moved. A surface that could not be built at `t` is a fallback
# that did not fire, and the session breaks instead.
function _surface_mark(cut::TimeCut, c::ContractKey, t::DateTime)::Union{Nothing,Float64}
    _has_provider(cut, VolatilitySurface) || return nothing
    s = try
        only_or_missing(asof(cut, VolatilitySurface, c.underlying, t))
    catch e
        (e isa UnservedSelector || e isa DerivationExhausted) || rethrow()
        return nothing
    end
    ismissing(s) && return nothing
    s.timestamp == t && return _price_at(s, c)
    return nothing                                           # stale: not a mark at `t`
end

# One contract off a surface already known to be stamped at the mark instant.
function _price_at(s::VolatilitySurface, c::ContractKey)::Union{Nothing,Float64}
    get_slice(s, c.expiry) === nothing && return nothing      # no slice at this expiry
    p = price(s, c.expiry, c.strike, c.option_type)
    return (isfinite(p) && p >= 0) ? p : nothing
end

"""
    mark_price(cut::TimeCut, contract::ContractKey, t::DateTime) -> Float64

What one open contract is worth per share at `t`, read through `cut`:
its own quote mid `(bid + ask) / 2`, with the surface price as fallback,
and [`UnpriceableLeg`](@ref) `:no_mark` after that. Reads only; writes
nothing.

The mid, not the last trade: a trade may be hours stale at a quiet
strike, while a two-sided quote is a price someone stands behind at `t`.
A one-sided, non-finite or negative quote is not a mid, so it falls
through to the surface exactly as an absent one does. The surface must be
stamped at `t`: a stale one prices the contract at its own instant, and
reusing it across sessions would be carrying a price forward.

Marking is pricing a leg, so the failure carries the same name and the
same three fields as every other unpriceable leg. `:no_mark` is the mark
counterpart of `:no_quote`: not "the chain is empty" but "no honest price
for this contract exists here at all, by either route".
"""
function mark_price(cut::TimeCut, contract::ContractKey, t::DateTime)::Float64
    for q in at(cut, OptionQuote, contract.underlying, t)
        q.strike      == contract.strike      || continue
        q.expiry      == contract.expiry      || continue
        q.option_type == contract.option_type || continue
        if !ismissing(q.bid) && !ismissing(q.ask)
            mid = (Float64(q.bid) + Float64(q.ask)) / 2
            # Non-negative as well as finite, the same bar the surface route
            # holds its own prices to. A long option is never a liability and
            # a short one never an asset, so a negative mid is corrupt input,
            # not a cheap contract; `OptionQuote` does not validate, so this
            # is where it is caught. It falls through to the surface exactly
            # as an absent quote does.
            (isfinite(mid) && mid >= 0) && return mid
        end
        break                                   # one quote per contract per instant
    end
    p = _surface_mark(cut, contract, t)
    p === nothing && throw(UnpriceableLeg(contract, t, :no_mark))
    return p
end

"""
    marked_curve(L::Ledger, data, u::Underlying, from::DateTime, to::DateTime) -> MarkedCurve

The [`MarkedCurve`](@ref) of `L` over `[from, to]`: at the close of every
session of `u` inside the window ([`session_closes`](@ref)), the ledger's
cash as it stood at that instant plus the open book marked to market
through a `TimeCut` at that instant.

Not a function of the ledger alone, which is the whole reason it is built
here rather than derived on demand: only the unrealised term needs market
data, and only when the book is not flat. `data` is the opened reader map.

Every open lot at a session close is either marked ([`mark_price`](@ref))
or the whole session is unmarkable: one lot without an honest price makes
the portfolio total unanswerable, so the point joins `unmarked_at` under
that lot's reason and no partial sum is reported. A calendar-open date
whose window holds no print joins it as `:unexpected_gap`, stamped at the
session's nominal 16:00 ET close -- a label on a failure, never a value.

Reads through a cut at each point, so no-lookahead is structural here as
in the tick loop, and warns once with the count if any session went
unmarked (design rule 7: the boundary that finds a gap is the boundary
that reports it).
"""
function marked_curve(L::Ledger, data, u::Underlying,
                      from::DateTime, to::DateTime)::MarkedCurve
    grid = session_closes(data, u, from, to)
    points = Tuple{DateTime,Bool}[(t, false) for t in grid.closes]
    append!(points, Tuple{DateTime,Bool}[(t, true) for t in grid.gaps])
    sort!(points; by = first)

    # The effective-time replay, folded forward once across the grid rather
    # than refolded per point: `book_effective` at every session close would
    # re-sort and re-apply the whole journal thousands of times for the same
    # answer. Equal instants fold in sequence order, as `book_effective`
    # defines it, and every reference points backward in effective time, so
    # the fold never meets a lot it has not yet opened.
    due = sort(L.events; by = e -> (effective_at(e), sequence(e)))
    book = Book()
    next = 1

    timestamps, profit = DateTime[], Float64[]
    unmarked_at, unmarked_reason = DateTime[], Symbol[]
    for (t, is_gap) in points
        while next <= length(due) && effective_at(due[next]) <= t
            apply!(book, due[next])
            next += 1
        end
        if is_gap
            push!(unmarked_at, t); push!(unmarked_reason, :unexpected_gap)
            continue
        end
        cut = TimeCut(data, t)
        open_value = 0.0
        reason = nothing
        for lot in open_lots(book)
            try
                open_value += side_sign(lot.side) * mark_price(cut, lot.contract, t) *
                              contract_spec(lot.contract.underlying).multiplier * lot.remaining
            catch e
                e isa UnpriceableLeg || rethrow()
                reason = e.reason
                break
            end
        end
        if reason === nothing
            push!(timestamps, t); push!(profit, cents_to_usd(book.cash) + open_value)
        else
            push!(unmarked_at, t); push!(unmarked_reason, reason)
        end
    end
    isempty(unmarked_at) || @warn(
        "marked curve: sessions left unmarked", underlying = u,
        n_unmarked = length(unmarked_at), reasons = sort(unique(unmarked_reason)))
    return MarkedCurve(timestamps, profit, unmarked_at, unmarked_reason)
end
