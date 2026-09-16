# Lifecycle: which lots fall due at a tick, and what price settles them.
#
# The same split as the venue: the engine computes, the ledger records.
# `settlement_price` is the rule, a symbol dispatched through a table in
# the `_FILL_RULES` style; `settlements` applies it to the open lots and
# returns what the ledger's own `record_expiry!` takes. Neither mutates
# anything. `settlements` is not side-effect-free, though: it is the one
# boundary that catches an unpriceable lot, and it warns (see below).
#
# Which rule a lot gets is not the caller's choice: settlement style is a
# contract fact, `contract_spec(u).settlement`, so `settlements` routes per
# lot off the table. A run-level symbol could not be right for a book
# holding both an AM- and a PM-settled contract, and it agreed with the
# table today only because every underlying the table lists is PM-settled.
#
# `UnpriceableLeg` is `engine.jl`'s, beside the other named failures:
# settling is pricing a leg at intrinsic against a reference print, so
# the failure has the same name and the same three fields.
#
# Sessions come from the spot tree, not from the calendar. A date is a
# session when the underlying printed in the reference window, and its
# close is the last of those prints. That reads an early close (official
# close 13:00 ET) with no early-close table, and it does so on the
# strength of its input rather than of the bounds: the window's last print
# is the session's only where the tree holds regular-session prints alone
# (`market_data.md`). A provider that also serves extended-hours prints
# breaks it -- a 15:59 print on a 13:00 ET close sits inside the window and
# settles the contract -- and this rule cannot detect that, because nothing
# in a `SpotPrice` says which session it came from. The production tree
# does not guarantee it either; what holds there is measured, not
# promised, and `market_data.md` says so. The calendar only contradicts
# the tree: a printless weekday it calls open is a named valuation failure
# (design rule 7), never evidence that the exchange was closed.
#
# `session_closes` is that same rule enumerated rather than applied to one
# contract: the instants of every session in a window, which is the grid
# the metrics module samples its marked curve on. One rule, so the grid a
# ratio is annualised over and the price a contract settles at cannot
# drift apart.
#
# Two bounds keep the answer honest rather than merely permitted by the
# cut. The reference window ends at the earlier of 16:00 ET and the
# contract's own expiry, so an intraday expiry never settles at a print
# from after it expired -- a price that did not exist at the instant the
# event stamps, which would make `book_effective` report a lot settled at
# a future number. And the cut must reach the expiry at all: a cut that
# cannot see the settlement session's close is `:no_session_close`, not a
# provisional intraday print dressed up as a settlement.
#
# The lower bound has a domain too. `:session_close` is PM settlement, and
# an expiry earlier than 09:30 ET on its own listed date has no session
# close behind it to settle against: it settles against that session's
# *opening* print, the AM-settled case (`:pre_open_expiry`). That is a
# contract this rule does not serve, not a gap in the data -- the window
# is empty by construction, and no print could fill it.

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

A session counts only when its **whole** reference window lies inside
`[from, to]`. A window the evaluation bounds clip is not a short session
but a session this run did not see end to end, and a consumer that
annualises by sessions needs every observation to span one whole one.
That absence is temporal (design rule 7): the session is outside the
window, not unanswerable inside it.

Early closes and unscheduled closures need no table here for the same
reason they need none in `:session_close`, and carry the same
requirement of the input: the window's last print is the session's only
where the tree holds regular-session prints alone (`market_data.md`).

**One read per session window, never one range read across the whole
period.** The reference window is the only place that input contract is
claimed to hold; a range read spanning the gaps between sessions also
reads the extended-hours prints the contract says nothing about, and on
the production SPY tree those include an instant carrying two
disagreeing rows -- a `ConflictingRecords` that aborts a read no session
needed. Reading exactly the windows `:session_close` reads gives this
function exactly that rule's exposure to the data, and nothing more. The
per-day block cache means the cost is still one partition read per date.
"""
function session_closes(m, u::Underlying, from::DateTime, to::DateTime)
    closes, gaps = DateTime[], DateTime[]
    # Walk the dates, not the prints: deriving the grid from whatever
    # printed would make a printless open date silently absent instead of a
    # named gap, and would read instants no session covers.
    d, last_d = _et_date(from), _et_date(to)
    while d <= last_d
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
#
# AM settlement is the gap, and it is future work rather than an oversight.
# The rule it needs is a second entry here, `:session_open`, reading the
# *first* print of the listed session rather than the last -- a different
# window, not a different bound on this one. An expiry before 09:30 ET is
# precisely the contract that would ask for it, which is why
# `:session_close` names that case (`:pre_open_expiry`) instead of guessing
# at a price. It is unwritten on purpose: no AM-settled underlying exists
# in the contract table to test it against, and a settlement rule nothing
# exercises is worse than an absent one. Until one does, an AM-settled
# contract is `UnsupportedSettlement` at both ends -- at load and here.
const _SETTLEMENT_RULES = Dict{Symbol,Function}(:session_close => _session_close)

"""
    UnsupportedSettlement

Thrown by [`settlements`](@ref) for a lot whose underlying settles in a
style no rule in `_SETTLEMENT_RULES` serves. Carries the `underlying` and
its `style`.

This is a configuration error, not a valuation one, which is why it stops
the run instead of joining the `unsettled` list: [`UnpriceableLeg`](@ref)
names one lot whose price is unavailable at this instant and design rule 7
says leave that lot open and say so, while this names a contract class the
codebase cannot settle at all -- every later tick would give the same
answer, and finishing the run would report a position that was never
valued as though it were merely still open. `load_experiment` refuses such
a config up front; reaching here means the book holds a lot the loader
never saw.
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

**What this rule needs of its input.** Early closes need no table, but
only where the data is regular-session prints alone: under that the last
print in the window of a 13:00 ET close is the 13:00 one. A provider
that also serves extended-hours prints breaks this rule silently -- a
15:59 print on an early-close day is inside the 09:30-16:00 window and
becomes the settlement price -- and no bound here can catch it, since
nothing in a `SpotPrice` records which session it came from. The
production spot tree does serve extended hours and is measured not to
print inside the exposed window; `market_data` states the requirement,
what the tree actually provides, and which data kind would make the rule
structural instead.

**Domain.** `cut` must reach the contract's expiry; a cut before it is
`:no_session_close`. The question this answers is what the contract
settled at, and a cut that cannot see the settlement session's close
cannot answer it -- it would return a provisional intraday print, or
read a truncated window and walk back past a session that had not
finished printing. Design rule 7 gives that a name rather than a
plausible-looking number. The engine can never reach it: lifecycle
builds the cut at a tick at or after the expiry. It exists for a direct
caller driving the lifecycle step from its own loop.

The other edge is the contract, not the cut: an expiry earlier than
09:30 ET on a listed date the calendar calls open is
`:pre_open_expiry`. Such a contract settles against that session's
opening print -- the AM-settled case, which no rule here serves yet --
and this rule's window for it is empty however complete the data is, so
design rule 7 names the contract rather than reporting a data gap. An
expiry at exactly 09:30 ET is a one-instant window and settles at the
opening print if one exists. When the listed date is *not* a session the
question does not arise: the walk back proceeds as it does for any
closed date, and the settlement session is the previous one.

Within the domain every instant queried is at or before the contract's
expiry, which is at or before the cut, and every read goes through the
cut, so no-lookahead is structural here as everywhere else.
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

The rule is per lot, not per run: `contract_spec(underlying).settlement`
picks it, so a book holding two styles settles each one correctly. A lot
whose style no rule serves throws [`UnsupportedSettlement`](@ref) and the
run stops -- see that type for why it is not caught and warned like an
unpriceable lot.

A lot is examined exactly once, ever: the interval, not an
`expiry <= t` threshold. An unsettleable lot stays open by design and
its answer is fixed by the contract's expiry rather than by `t`, so a
threshold would re-derive the same failure at every later tick. The
interval misses nothing only because the engine refuses to fill a leg at
or after its contract's expiry (`fill_legs`, `:expired_contract`): every
lot is therefore in the book strictly before its own expiry, and so
inside the interval that examines it. The ledger alone does not give
that -- it accepts a fill effective at the expiry instant -- so the
guarantee is the engine's, not the ledger's.

The interval is open below, so a contract expiring exactly at `prev` is
never examined; the engine passes the window start at the first tick,
which is safe today because nothing is open before it. Seeding a ledger
with open lots -- resuming a run, or a live loop against an existing
book -- is what would make that bound matter.

This is the one place the named failure is caught, and the one place it
is reported: a `@warn` per unsettled lot, carrying the contract, its
expiry and the reason. Reporting here rather than at the call site is
design rule 7's own reason -- a caller could forget, and a silent gap is
exactly what the rule exists to prevent. The warning is not the whole
account, though: `run_backtest` carries every entry out as a
[`RunFailure`](@ref) on its result, because a log line does not survive
the run and a ledger replay cannot recover a thing that did not happen.
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
