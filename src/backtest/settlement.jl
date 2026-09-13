# Lifecycle: which lots fall due at a tick, and what price settles them.
#
# The same split as the venue: the engine computes, the ledger records.
# `settlement_price` is the rule, a symbol dispatched through a table in
# the `_FILL_RULES` style; `settlements` applies it to the open lots and
# returns what the ledger's own `record_expiry!` takes. Neither mutates
# anything. `settlements` is not side-effect-free, though: it is the one
# boundary that catches an unpriceable lot, and it warns (see below).
#
# `UnpriceableLeg` is `engine.jl`'s, beside the other named failures:
# settling is pricing a leg at intrinsic against a reference print, so
# the failure has the same name and the same three fields.
#
# Sessions come from the spot tree, not from the calendar. A date is a
# session when the underlying printed in regular hours, and its close is
# the last of those prints; that is what settles the six early-close
# sessions (official close 13:00 ET) correctly with no early-close table.
# The calendar only contradicts the tree: a printless weekday it calls
# open is a named valuation failure (design rule 7), never evidence that
# the exchange was closed.

using BusinessDays

# The exchange calendar as BusinessDays ships it. `isbday` is false for
# weekends, the ten annual NYSE holidays and the ad-hoc closures.
const _NYSE = BusinessDays.USNYSE()

# Ad-hoc closures the calendar might miss. Checked against BusinessDays
# 0.9.25 on 2026-09-13 and **empty**: `USNYSE` already carries both
# national days of mourning (2018-12-05, George H. W. Bush; 2025-01-09,
# Jimmy Carter) and the Hurricane Sandy closure (2012-10-29/30). Kept as
# the seam for the next one, which the calendar will not have on the day.
const _ADHOC_CLOSURES = Set{Date}()

# US equity regular trading hours, in ET. The close is the bound of the
# reference window, not a claim that a print exists at it.
const _SESSION_OPEN  = Time(9, 30)
const _SESSION_CLOSE = Time(16, 0)

# How far back the walk for a settlement session may go, in calendar
# days. Four consecutive closed days is the longest the NYSE calendar
# produces (Hurricane Sandy after a weekend, a holiday either side of
# one); ten leaves room and still terminates, in the spirit of
# `DerivationExhausted`. Under this calendar `:no_session` is therefore
# the bound's guarantee rather than a case the data can reach -- the
# printless-weekday check fires first -- and it stays because the
# termination guarantee has to have a name.
const _SESSION_WALK_DAYS = 10

# Contract expiries are stamped `et_to_utc(date, Time(16, 0))` and every
# timestamp on disk is UTC, so the listed date is read back in ET.
_et_date(dt::DateTime)::Date = Date(DateTime(astimezone(ZonedDateTime(dt, tz"UTC"), TZ_ET)))

# Whether the calendar says the exchange was closed on `d`.
_closed_on(d::Date)::Bool = !isbday(_NYSE, d) || d in _ADHOC_CLOSURES

function _session_close(cut::TimeCut, contract::ContractKey, t::DateTime)::Float64
    listed = _et_date(contract.expiry)
    for k in 0:(_SESSION_WALK_DAYS - 1)
        d = listed - Day(k)
        prints = between(cut, SpotPrice, contract.underlying,
                         et_to_utc(d, _SESSION_OPEN), et_to_utc(d, _SESSION_CLOSE))
        isempty(prints) || return last(prints).price
        _closed_on(d) || throw(UnpriceableLeg(contract, t, :unexpected_gap))
    end
    throw(UnpriceableLeg(contract, t, :no_session))
end

const _SETTLEMENT_RULES = Dict{Symbol,Function}(:session_close => _session_close)

"""
    settlement_price(rule::Symbol, cut::TimeCut, contract::ContractKey, t::DateTime) -> Float64

The reference price `contract` settles at under `rule`, read through
`cut`. Reads only; writes nothing. Throws [`UnpriceableLeg`](@ref)
(`:unexpected_gap`, `:no_session`) when no honest reference price
exists. Errors, naming the known rules, for an unknown `rule`.

`:session_close`: the last regular-session print of the settlement
session stands in for the official close. From the contract's listed
expiry date in ET, walk back at most `$(_SESSION_WALK_DAYS)` calendar
days; the first date whose underlying printed between 09:30 and 16:00 ET
is the settlement session and the last of those prints is the price. A
printless date the exchange calendar calls open is `:unexpected_gap`, a
data gap and not a closure; exhausting the walk is `:no_session`. Early
closes need no table: on a 13:00 ET close the last print in the window
is the 13:00 one.

Every instant queried is at or before the contract's expiry, which is at
or before `t`, and every read goes through the tick's cut, so
no-lookahead is structural here as everywhere else.
"""
function settlement_price(rule::Symbol, cut::TimeCut, contract::ContractKey, t::DateTime)::Float64
    f = get(_SETTLEMENT_RULES, rule) do
        error("settlement_price: unknown settlement rule :$rule. " *
              "Known: $(sort(collect(keys(_SETTLEMENT_RULES))))")
    end
    return f(cut, contract, t)
end

"""
    settlements(cut::TimeCut, book::Book, prev::DateTime, t::DateTime;
                settlement_rule::Symbol) -> (settled, unsettled)

The lifecycle step as a function of the cut and the book, the twin of
[`fill_legs`](@ref): the lots of `book` falling due in `(prev, t]`, each
paired with the price [`settlement_price`](@ref) gives it. `settled` is
a `Vector{Tuple{Lot,Float64}}`, ready for `record_expiry!`; `unsettled`
holds one [`UnpriceableLeg`](@ref) per lot that could not be priced.
Mutates nothing, and walks `open_lots` in opening-fill order so a replay
reproduces.

A lot is examined exactly once, ever: the interval, not an
`expiry <= t` threshold. An unsettleable lot stays open by design and
its answer is fixed by the contract's expiry rather than by `t`, so a
threshold would re-derive the same failure at every later tick. The
interval is open below, so a contract expiring exactly at `prev` is
never examined; the engine passes the window start at the first tick,
which is safe today because nothing is open before it and a fill after
expiry is refused. Seeding a ledger with open lots -- resuming a run, or
a live loop against an existing book -- is what would make that bound
matter.

This is the one place the named failure is caught, and the one place it
is reported: a `@warn` per unsettled lot, carrying the contract, its
expiry and the reason. Reporting here rather than at the call site is
design rule 7's own reason -- a caller could forget, and a silent gap is
exactly what the rule exists to prevent.
"""
function settlements(cut::TimeCut, book::Book, prev::DateTime, t::DateTime;
                     settlement_rule::Symbol)
    settled   = Tuple{Lot,Float64}[]
    unsettled = UnpriceableLeg[]
    for lot in open_lots(book)
        prev < lot.contract.expiry <= t || continue
        try
            push!(settled, (lot, settlement_price(settlement_rule, cut, lot.contract, t)))
        catch e
            e isa UnpriceableLeg || rethrow()
            @warn("lot left open: no honest settlement price",
                  contract = lot.contract, expiry = lot.contract.expiry, reason = e.reason)
            push!(unsettled, e)
        end
    end
    return (settled = settled, unsettled = unsettled)
end
