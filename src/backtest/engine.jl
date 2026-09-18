# The tick loop, the venue applied to an order, and the cross-record
# check between the fills and the order journal.

using Dates

# ---- named failures --------------------------------------------------

"""
    UnpriceableLeg

A leg that cannot honestly be priced (design rule 7): `contract` at `t`
for `reason`. From the venue: `:expired_contract` (`t` is at or after
the expiry), `:no_quote` (an empty chain, or the contract absent from
it), `:no_executable_side` (the side the fill rule needs is `missing`),
`:no_spot` (the underlying is served but has no spot at `t`). From
settlement, which is pricing a leg at intrinsic: `:unexpected_gap`,
`:no_session`, `:no_session_close` and `:pre_open_expiry`, as
[`settlement_price`](@ref) defines them. Nothing serving the selector
stays `UnservedSelector`, thrown by `at`.
"""
struct UnpriceableLeg <: Exception
    contract::ContractKey
    t::DateTime
    reason::Symbol
end

"""
    JoinViolation

A fill and its order leg disagree, found by [`check_join`](@ref):
`field` names what disagrees (`:order_id`, `:first_leg_id`,
`:observations`, `:contract`, `:side`, `:intent`, `:group`, `:quantity`,
`:quote_at`, `:spot_at`, `:fill_rule`, `:bid`, `:ask`, `:price`), `id`
the offending fill (or order record) and `reason` says how.
"""
struct JoinViolation <: Exception
    field::Symbol
    id::Int
    reason::String
end

Base.showerror(io::IO, e::UnpriceableLeg) =
    print(io, "UnpriceableLeg: ", e.contract, " at ", e.t, " cannot be priced (", e.reason, ")")
Base.showerror(io::IO, e::JoinViolation) =
    print(io, "JoinViolation: ", e.field, " on ", e.id, ": ", e.reason)

# ---- what a finished run could not answer -----------------------------

"""
    RunFailure

One unanswerable question a **completed** run retained: `at` is the
instant it was asked, `stage` which pass asked it (`:settlement` for the
lifecycle, `:mark` for the marked curve), `subject` names what could not
be answered, and `reason` is the [`UnpriceableLeg`](@ref) name.

`subject` tells two questions apart: a lot names its contract and its
opening fill, a session that never printed names its underlying. A
label, never a value.
"""
struct RunFailure
    at      :: DateTime
    stage   :: Symbol
    subject :: String
    reason  :: Symbol
end

# The whole vocabulary of `stage`, in the canonical order failures sort by.
# Two passes ask questions and no third one does, so the column is closed:
# a stored failure naming any other stage belongs to no pass this code runs
# and is a defect, not a row to keep. It lives beside `RunFailure` because
# the producers define it; persistence checks stored rows against it.
const RUN_FAILURE_STAGES = (:mark, :settlement)

# The subject of a failure about one contract, and about one lot of it.
# Two lots of the same contract ask two questions at one instant; the
# opening fill id is what distinguishes their answers.
_failure_subject(c::ContractKey)::String = string(
    ticker(c.underlying), " ", Dates.format(c.expiry, "yyyy-mm-ddTHH:MM:SS"), " ",
    c.strike, c.option_type == Call ? "C" : "P")
_failure_subject(lot::Lot)::String =
    _failure_subject(lot.contract) * " lot@" * string(lot.open_fill_id)

# ---- the venue applied to an order -----------------------------------

"""
    resolve_quote(cut::TimeCut, contract::ContractKey, t::DateTime) -> OptionQuote

The quote in `at(cut, OptionQuote, contract.underlying, t)` matching
`contract` exactly on `(underlying, strike, expiry, option_type)`.
Throws [`UnpriceableLeg`](@ref) `:no_quote` on an empty chain or a
contract absent from it: the policy emitted a leg for a contract it
could not see. Reads quotes, not surfaces: a surface retains only
inverted IVs, while the raw bid/ask the fill needs lives on the chain
quote.
"""
function resolve_quote(cut::TimeCut, contract::ContractKey, t::DateTime)::OptionQuote
    for q in at(cut, OptionQuote, contract.underlying, t)
        q.underlying  == contract.underlying  || continue
        q.strike      == contract.strike      || continue
        q.expiry      == contract.expiry      || continue
        q.option_type == contract.option_type || continue
        return q
    end
    throw(UnpriceableLeg(contract, t, :no_quote))
end

"""
    fill_legs(cut::TimeCut, order::Order, t::DateTime;
              fill_rule::Symbol, cost_model::Symbol, tick_cents::Int = TICK_CENTS)
        -> (prices, fees, observations, fill_rule)

The venue as a pure function: for every leg of `order`, before anything
else happens, check that the contract still trades at `t`, then resolve
its quote ([`resolve_quote`](@ref)), its price through `fill_rule`
([`fill_price`](@ref)) and the spot of the leg's own underlying at `t`;
then the commission of the whole order under `cost_model`
([`commission`](@ref)), negated into `Fee` amounts. Returns the per-leg
keywords `record_order!` takes as a `NamedTuple`; each observation is
the quote's bid, ask and timestamp and the spot's price and timestamp.
Throws [`UnpriceableLeg`](@ref) (`:expired_contract`, `:no_quote`,
`:no_executable_side`, `:no_spot`) for a leg that cannot honestly be
priced, and `UnservedSelector` when nothing serves an underlying. Reads
through the cut and writes nothing.
"""
function fill_legs(cut::TimeCut, order::Order, t::DateTime;
                   fill_rule::Symbol, cost_model::Symbol, tick_cents::Int = TICK_CENTS)
    n = length(order.legs)
    prices       = Vector{Float64}(undef, n)
    observations = Vector{LegObservation}(undef, n)
    for (k, leg) in enumerate(order.legs)
        # Trading stops at the expiry instant, so the contract cannot be
        # filled at or after it, whatever the chain still quotes.
        t < leg.contract.expiry || throw(UnpriceableLeg(leg.contract, t, :expired_contract))
        q = resolve_quote(cut, leg.contract, t)
        p = fill_price(fill_rule, q.bid, q.ask, leg.side, tick_cents)
        ismissing(p) && throw(UnpriceableLeg(leg.contract, t, :no_executable_side))
        # The leg's own underlying: a `spot_for` remap on the surface
        # provider prices the surface, not the fill.
        spot = only_or_missing(at(cut, SpotPrice, leg.contract.underlying, t))
        ismissing(spot) && throw(UnpriceableLeg(leg.contract, t, :no_spot))
        prices[k]       = p
        observations[k] = LegObservation(q.timestamp, q.bid, q.ask, spot.price, spot.timestamp)
    end
    fees = -commission(cost_model, prices, Int[leg.quantity for leg in order.legs])
    return (prices = prices, fees = fees, observations = observations, fill_rule = fill_rule)
end

# ---- the cross-record contract ---------------------------------------

function _check_fill_join(e::Fill, r::OrderRecord, k::Int, tick_cents::Int;
                          cumulative_quantity::Union{Nothing,Int}=nothing)::Nothing
    leg = r.order.legs[k]
    id = event_id(e)
    e.contract == leg.contract || throw(JoinViolation(:contract, id,
        "fill contract $(e.contract) differs from order leg $(e.order_leg_id)'s"))
    e.side == leg.side || throw(JoinViolation(:side, id,
        "fill side $(e.side) differs from order leg $(e.order_leg_id)'s $(leg.side)"))
    e.intent == leg.intent || throw(JoinViolation(:intent, id,
        "fill intent $(e.intent) differs from order leg $(e.order_leg_id)'s $(leg.intent)"))
    e.group == r.group || throw(JoinViolation(:group, id,
        "fill group $(e.group) differs from order $(r.order_id)'s $(r.group)"))
    quantity = something(cumulative_quantity, e.quantity)
    quantity <= leg.quantity || throw(JoinViolation(:quantity, id,
        "fills of order leg $(e.order_leg_id) sum to $quantity of $(leg.quantity) ordered"))
    e.fill_rule == :broker_execution && return nothing
    obs = r.observations[k]
    obs.quote_at <= r.decided_at || throw(JoinViolation(:quote_at, id,
        "quote observed at $(obs.quote_at), after the decision at $(r.decided_at)"))
    obs.spot_at <= r.decided_at || throw(JoinViolation(:spot_at, id,
        "spot observed at $(obs.spot_at), after the decision at $(r.decided_at)"))
    haskey(_FILL_RULES, e.fill_rule) || throw(JoinViolation(:fill_rule, id,
        "unknown fill rule :$(e.fill_rule)"))
    p = fill_price(e.fill_rule, obs.bid, obs.ask, e.side, tick_cents)
    ismissing(p) && throw(JoinViolation(e.side == Long ? :ask : :bid, id,
        "the side :$(e.fill_rule) needs for a $(e.side) leg is missing from the observation"))
    e.price == p || throw(JoinViolation(:price, id,
        "price $(e.price) is not :$(e.fill_rule) applied to the observation ($p)"))
    return nothing
end

"""
    check_join(L::Ledger; tick_cents::Int = TICK_CENTS) -> Nothing

The fill review's cross-record contract between the events and the
order journal, used for persistence write and load. Order
records: ids are `1, 2, ...` in order, each `first_leg_id` is the
previous record's plus its leg count (so leg ids are contiguous and
never shared), and there is one observation per leg. For every `Fill`:
its order leg exists (`DanglingReference` `:order_leg_id` otherwise);
contract, side and intent equal the leg's and group equals the
record's; over the whole ledger the fills of that leg sum to at most the
leg's quantity; and, unless `fill_rule == :broker_execution`, the leg's
observation has `quote_at` and `spot_at` at or before `decided_at`, the
side the rule needs present, and
`price == fill_price(fill_rule, bid, ask, side, tick_cents)` (an unknown
rule is a violation on `:fill_rule`). Under `:broker_execution` this
slice still keeps one observation row per leg, but its values are not
consulted. Every other disagreement is a
[`JoinViolation`](@ref) naming its field. Execution-id uniqueness is
`commit!`'s `DuplicateExecution` and is not repeated here.
"""
function check_join(L::Ledger; tick_cents::Int = TICK_CENTS)::Nothing
    expect_id, expect_leg = 1, 1
    for r in L.orders
        r.order_id == expect_id || throw(JoinViolation(:order_id, r.order_id,
            "order ids are not 1, 2, ... in order; expected $expect_id"))
        r.first_leg_id == expect_leg || throw(JoinViolation(:first_leg_id, r.order_id,
            "leg ids are not contiguous: first_leg_id $(r.first_leg_id), expected $expect_leg"))
        length(r.observations) == length(r.order.legs) || throw(JoinViolation(:observations, r.order_id,
            "$(length(r.observations)) observations for $(length(r.order.legs)) legs"))
        expect_id  += 1
        expect_leg += length(r.order.legs)
    end
    filled = Dict{Int,Int}()                      # order leg id -> quantity filled so far
    for e in L.events
        e isa Fill || continue
        r, k = order_leg(L, e.order_leg_id)
        total = get(filled, e.order_leg_id, 0) + e.quantity
        filled[e.order_leg_id] = total
        _check_fill_join(e, r, k, tick_cents; cumulative_quantity=total)
    end
    return nothing
end

"""
    check_join(L::Ledger, rec::OrderRecord; tick_cents::Int = TICK_CENTS) -> Nothing

Check one newly appended order record and the fills it produced. The scan
is bounded by `rec.known_to` and filters the record's contiguous leg-id
range, so other orders appended from the same decision boundary are ignored.
The per-fill contract is identical to the whole-ledger form, except quantity
is checked per fill; cumulative partial-fill quantity remains a whole-ledger
check.
"""
function check_join(L::Ledger, rec::OrderRecord; tick_cents::Int = TICK_CENTS)::Nothing
    n = length(rec.order.legs)
    length(rec.observations) == n || throw(JoinViolation(:observations, rec.order_id,
        "$(length(rec.observations)) observations for $n legs"))
    first_id, last_id = rec.first_leg_id, rec.first_leg_id + n - 1
    for e in Iterators.reverse(L.events)
        sequence(e) <= rec.known_to && break
        e isa Fill || continue
        first_id <= e.order_leg_id <= last_id || continue
        _check_fill_join(e, rec, e.order_leg_id - first_id + 1, tick_cents)
    end
    return nothing
end

# ---- the loop ---------------------------------------------------------

"""
    run_backtest(agent::Agent, data::MarketData, from::DateTime, to::DateTime,
                 clock::Clock; fill_rule = :cross_spread,
                 cost_model = :ibkr_pro_us_options) -> (ledger, failures)

Walk the ticks of `clock` in `[from, to]`, or the agent's `tick_times`
schedule when it returns one. Per tick: settle the lots that fell due
since the previous tick ([`settlements`](@ref), then the ledger's
[`record_expiry!`](@ref)), build the cut, ask the agent for its policy
and the policy for orders on the ledger's own book, price every leg of
each order ([`fill_legs`](@ref)), book it as one transaction
([`record_order!`](@ref)) with `known_to` captured after the tick's
expiries, and run the per-record [`check_join`](@ref). After the last
tick, settle once more at `to`. `data` is the opened reader map.

Returns the ledger paired with the [`RunFailure`](@ref)s the run
retained, one per lot either settlement pass left open. The venue
choices are keywords with the same defaults `Experiment` takes; the tick
is [`TICK_CENTS`](@ref) and the settlement rule is a contract fact per
lot.
"""
function run_backtest(agent::Agent, data::MarketData, from::DateTime, to::DateTime,
                      clock::Clock; fill_rule::Symbol = :cross_spread,
                      cost_model::Symbol = :ibkr_pro_us_options)
    L = Ledger()
    failures = RunFailure[]
    # Sparse policies (once a day on minute data) override `tick_times` so
    # the engine never enumerates the clock's grid; keep the `if`.
    ticks = tick_times(agent, data, from, to)
    if ticks === nothing
        ticks = timestamps(data, clock, from, to)
    end
    prev = from                                    # lower bound of the lifecycle interval
    for t in ticks
        cut = TimeCut(data, t)
        # 1. Settle what fell due in (prev, t], before the decision.
        due = settlements(cut, L.book, prev, t)
        foreach(due.settled) do (lot, p)
            record_expiry!(L, lot; settlement_price = p,
                           effective_at = lot.contract.expiry, recorded_at = t)
        end
        _retain_unsettled!(failures, due.unsettled)
        prev = t
        # 2. Decide on the book the ledger owns.
        policy = current_policy(agent, t, cut, L.book)
        orders = decide(policy, t, cut, L.book)
        known_to = last_sequence(L)                # what every order of this tick saw
        # 3. Fill: every leg priced before anything is written.
        for order in orders
            rec = record_order!(L, order; fill_legs(cut, order, t; fill_rule, cost_model)...,
                                effective_at = t, recorded_at = t, known_to)
            check_join(L, rec)
        end
    end
    # 4. Window end: settle once more at the evaluation endpoint.
    final = settlements(TimeCut(data, to), L.book, prev, to)
    foreach(final.settled) do (lot, p)
        record_expiry!(L, lot; settlement_price = p,
                       effective_at = lot.contract.expiry, recorded_at = to)
    end
    _retain_unsettled!(failures, final.unsettled)
    return (ledger = L, failures = failures)
end

# A lot left open because no honest settlement price existed, kept as a
# fact of the run. Stamped at the instant the question was asked, which is
# the tick that examined the lot -- a lot is examined exactly once, ever.
function _retain_unsettled!(failures::Vector{RunFailure},
                            unsettled::AbstractVector{<:Tuple{Lot,UnpriceableLeg}})
    for (lot, e) in unsettled
        push!(failures, RunFailure(e.t, :settlement, _failure_subject(lot), e.reason))
    end
    return failures
end

"""
    run_backtest(policy::Policy, data::MarketData, from, to, clock; kw...) -> (ledger, failures)

Convenience overload for the fixed-policy case: wraps `policy` in a
`StaticAgent` and runs the agent-driven loop with the same keywords, and
returns the same `(ledger, failures)` pair.
"""
run_backtest(policy::Policy, data::MarketData, from::DateTime, to::DateTime, clock::Clock; kw...) =
    run_backtest(StaticAgent(policy), data, from, to, clock; kw...)
