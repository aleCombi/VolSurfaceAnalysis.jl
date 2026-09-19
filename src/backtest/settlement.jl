# Lifecycle: which lots fall due at a tick and what price settles them,
# and the session grid the marked curve samples on.

using BusinessDays

# The exchange calendar as BusinessDays ships it. `isbday` is false for
# weekends, the ten annual NYSE holidays and the ad-hoc closures.
const _NYSE = BusinessDays.USNYSE()

# Ad-hoc closures the calendar might miss. Checked against BusinessDays
# 0.9.25 on 2026-09-13 and **empty**: `USNYSE` already carries both
# national days of mourning (2018-12-05, George H. W. Bush; 2025-01-09,
# Jimmy Carter) and the Hurricane Sandy closure (2012-10-29/30).
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

# Every timestamp is UTC, so the listed date of an expiry is its ET date.
# `parse_massive_ticker` stamps `et_to_utc(date, Time(16, 0))`, but a
# `ContractKey` takes any instant and the window bound above does not
# assume that convention.
_et_date(dt::DateTime)::Date = Date(DateTime(astimezone(ZonedDateTime(dt, tz"UTC"), TZ_ET)))

# Whether the calendar says the exchange was closed on `d`.
_closed_on(d::Date)::Bool = !isbday(_NYSE, d) || d in _ADHOC_CLOSURES

function _session_close(cut::TimeCut, contract::ContractKey, t::DateTime)::Float64
    # The domain. Every window below ends at or before the expiry instant,
    # so a cut that does not reach the expiry cannot see the settlement
    # session's close: it would answer with a provisional intraday print,
    # or read a truncated -- possibly empty -- window and walk back past a
    # session that had not finished printing yet. Design rule 7: name it.
    cut.cutoff < contract.expiry &&
        throw(UnpriceableLeg(contract, t, :no_session_close))
    listed = _et_date(contract.expiry)
    for k in 0:(_SESSION_WALK_DAYS - 1)
        d = listed - Day(k)
        # The window never runs past the expiry instant. On the listed date
        # of an intraday expiry that is the expiry itself; on an ordinary
        # 16:00 ET expiry and on every walked-back session it is the close.
        window_start = et_to_utc(d, _SESSION_OPEN)
        window_end   = min(et_to_utc(d, _SESSION_CLOSE), contract.expiry)
        if window_start > window_end
            # The rule's other domain edge, and only the listed date can
            # reach it: every walked-back date closes before the expiry
            # instant, so its window is the whole session. An expiry before
            # 09:30 ET is the AM-settled shape -- the opening print settles
            # it, not a close this rule could read -- so name it rather than
            # hand the reversed range to a provider and then blame the data
            # for an emptiness the bounds created. The calendar is the only
            # witness available here, because the window that would show a
            # print is empty by construction; on a date it calls closed
            # nothing is claimed about the contract at all, the listed date
            # is simply not a session, and the walk back is the answer.
            _closed_on(d) || throw(UnpriceableLeg(contract, t, :pre_open_expiry))
            continue
        end
        # `between` promises an iterable, not a container: consume it once
        # and keep the last record, which is the latest because the protocol
        # promises sorted order. Indexing it would fail on a lazy reader, and
        # probing it for emptiness and then reading it again is a second
        # traversal a single-pass reader need not survive.
        close_price = nothing
        for p in between(cut, SpotPrice, contract.underlying, window_start, window_end)
            close_price = p.price
        end
        close_price === nothing || return close_price
        _closed_on(d) || throw(UnpriceableLeg(contract, t, :unexpected_gap))
    end
    throw(UnpriceableLeg(contract, t, :no_session))
end

"""
    session_closes(m, u::Underlying, from::DateTime, to::DateTime) -> (closes, gaps)

The session grid of `u` inside `[from, to]`, by the same rule
[`settlement_price`](@ref) settles against: a calendar-open ET date is a
session when `u` printed in the reference window (09:30-16:00 ET), and
its close is the last of those prints. `closes` holds one instant per
session, ascending; `gaps` holds the nominal 16:00 ET instant of each
calendar-open date that printed nothing, the [`UnpriceableLeg`](@ref)
`:unexpected_gap` case in grid form -- a date the exchange says was open
and the tree cannot place a close for. Mutates nothing.

A session counts only when its whole reference window lies inside
`[from, to]`; a clipped window is temporal absence. Reads one session
window at a time, never the gaps between them, and carries the exposure
[`settlement_price`](@ref) states.
"""
function session_closes(m, u::Underlying, from::DateTime, to::DateTime)
    closes, gaps = DateTime[], DateTime[]
    # Walk the dates, not the prints: deriving the grid from whatever
    # printed would make a printless open date silently absent instead of a
    # named gap, and would read instants no session covers.
    d, last_d = _et_date(from), _et_date(to)
    while d <= last_d
        # The calendar is asked first here, unlike `_session_close`, which
        # reads the tree first: a date the calendar calls closed is not a
        # session even if the tree printed on it (pinned by a test).
        if !_closed_on(d)
            window_start = et_to_utc(d, _SESSION_OPEN)
            window_end   = et_to_utc(d, _SESSION_CLOSE)
            # The whole window, or nothing: a clipped one is a session this
            # caller did not see end to end.
            if from <= window_start && window_end <= to
                # `between` promises an iterable, not a container: consume it
                # once and keep the last, which is the latest because the
                # protocol promises sorted order.
                last_ts = nothing
                for p in between(m, SpotPrice, u, window_start, window_end)
                    last_ts = p.timestamp
                end
                last_ts === nothing ? push!(gaps, window_end) : push!(closes, last_ts)
            end
        end
        d += Day(1)
    end
    return (closes = closes, gaps = gaps)
end

# The settlement rules by name, the `_FILL_RULES` shape. One entry, and it
# is PM settlement: the contract settles against the close of its
# settlement session, which is what `ContractSpec.settlement` records as
# `PMSettled` for every underlying `_CONTRACT_TABLE` lists.
# AM settlement (`:session_open`, the first print of the listed session) is
# parked in status.md; an AM-settled contract is `UnsupportedSettlement`
# at load and here.
const _SETTLEMENT_RULES = Dict{Symbol,Function}(:session_close => _session_close)

"""
    UnsupportedSettlement

Thrown by [`settlements`](@ref) for a lot whose underlying settles in a
style no rule in `_SETTLEMENT_RULES` serves. Carries the `underlying` and
its `style`.

A configuration error, so it stops the run rather than joining the
`unsettled` list; `load_experiment` refuses such a config up front.
"""
struct UnsupportedSettlement <: Exception
    underlying::Underlying
    style::SettlementStyle
end

Base.showerror(io::IO, e::UnsupportedSettlement) = print(io,
    "UnsupportedSettlement: ", e.underlying, " options are ", e.style,
    " and no settlement rule serves that style (known styles: PMSettled)")

# Settlement style is a contract fact, so the rule is looked up per lot
# rather than passed in. An underlying the contract table does not list is
# `UnknownContract` from `contract_spec`, the same failure `record_expiry!`
# would raise on the lot a moment later.
function _rule_for(u::Underlying)::Symbol
    style = contract_spec(u).settlement
    style === PMSettled || throw(UnsupportedSettlement(u, style))
    return :session_close
end

"""
    settlement_price(rule::Symbol, cut::TimeCut, contract::ContractKey, t::DateTime) -> Float64

The reference price `contract` settles at under `rule`, read through
`cut`. Reads only; writes nothing. Throws [`UnpriceableLeg`](@ref)
(`:no_session_close`, `:pre_open_expiry`, `:unexpected_gap`,
`:no_session`) when no honest reference price exists. Errors, naming the
known rules, for an unknown `rule`.

`:session_close`: PM settlement. The last regular-session print of the
settlement session stands in for the official close. From the contract's
listed expiry date in ET, walk back at most `$(_SESSION_WALK_DAYS)`
calendar days; the first date whose underlying printed in the window is
the settlement session and the last of those prints is the price. The
window runs from 09:30 ET to **the earlier of 16:00 ET and the
contract's own expiry instant**, so an intraday expiry never settles at
a print from after it expired. A printless date the exchange calendar
calls open is `:unexpected_gap`, a data gap and not a closure;
exhausting the walk is `:no_session`.

**Exposure.** Early closes need no table, but only where no
extended-hours print falls inside the window: then the last print in
the window of a 13:00 ET close is the 13:00 one. A 15:59 print on an
early-close day would become the settlement price, and no bound here can
catch it, since nothing in a `SpotPrice` records which session it came
from. The production spot tree does serve extended hours and is measured
not to print inside an early-close window; the `data` module promises
nothing about sessions. The official-close kind parked in status.md
would make the rule structural.

**Domain.** `cut` must reach the contract's expiry; a cut before it is
`:no_session_close`. An expiry earlier than 09:30 ET on a listed date
the calendar calls open is `:pre_open_expiry`, the AM-settled case no
rule here serves; at exactly 09:30 ET the window is one instant, and a
print visible at it settles the contract. When the listed date is closed
the walk back proceeds as for any closed date.
"""
function settlement_price(rule::Symbol, cut::TimeCut, contract::ContractKey, t::DateTime)::Float64
    f = get(_SETTLEMENT_RULES, rule) do
        error("settlement_price: unknown settlement rule :$rule. " *
              "Known: $(sort(collect(keys(_SETTLEMENT_RULES))))")
    end
    return f(cut, contract, t)
end

"""
    settlements(cut::TimeCut, book::Book, prev::DateTime, t::DateTime) -> (settled, unsettled)

The lifecycle step as a function of the cut and the book, the twin of
[`fill_legs`](@ref): the lots of `book` falling due in `(prev, t]`, each
paired with the price [`settlement_price`](@ref) gives it. `settled` is
a `Vector{Tuple{Lot,Float64}}`, ready for `record_expiry!`; `unsettled`
is the same shape for the ones that failed, each lot paired with the
[`UnpriceableLeg`](@ref) it raised. The lot rides along because the
caller has to say *which* lot went unanswered, and two lots of one
contract falling due together are two questions. Mutates nothing, and
walks `open_lots` in opening-fill order so a replay reproduces.

The rule is per lot: `contract_spec(underlying).settlement` picks it,
and a style no rule serves throws [`UnsupportedSettlement`](@ref). The
one place a settlement failure is caught: a `@warn` per unsettled lot,
with the contract, its expiry and the reason. The interval is open
below, so a lot whose expiry is at or before `prev` is never examined;
through the engine none can be, since the venue refuses a fill at or
after expiry, but a lot written through `record_order!` directly at its
expiry instant is legal to the ledger and escapes every interval.
"""
function settlements(cut::TimeCut, book::Book, prev::DateTime, t::DateTime)
    settled   = Tuple{Lot,Float64}[]
    unsettled = Tuple{Lot,UnpriceableLeg}[]
    for lot in open_lots(book)
        prev < lot.contract.expiry <= t || continue
        # Outside the `try`: an unsupported style is not a valuation
        # failure and must not be swallowed by the catch below.
        rule = _rule_for(lot.contract.underlying)
        try
            push!(settled, (lot, settlement_price(rule, cut, lot.contract, t)))
        catch e
            e isa UnpriceableLeg || rethrow()
            @warn("lot left open: no honest settlement price",
                  contract = lot.contract, expiry = lot.contract.expiry, reason = e.reason)
            push!(unsettled, (lot, e))
        end
    end
    return (settled = settled, unsettled = unsettled)
end
