# Backtest engine.
#
# Drives an `Agent` (or a bare `Policy`, wrapped in a `StaticAgent`) over
# the ticks of a declared `Clock` in `[from, to]`. Per tick: builds a
# `TimeCut` of the data at `t`, asks the agent for the current `Policy`,
# asks that policy for orders, fills each order against the quote chain
# at `t`, and appends the resulting `Position`s to the ledger. Returns
# the bare ledger -- reporting / PnL aggregation is a downstream concern.

using Dates

"""
    resolve_quote(cut::TimeCut, trade::Trade, t::DateTime) -> OptionQuote

Find the `OptionQuote` in `at(cut, OptionQuote, trade.underlying, t)`
that matches `trade`'s contract (underlying, expiry, strike,
option_type). The match is exact on all four fields; a strike not in
the chain is a programming error (the policy should only emit trades
for contracts it can see).

Reads quotes, not surfaces: a surface retains only inverted IVs, while
the raw bid/ask the fill needs lives on the chain quote.
"""
function resolve_quote(cut::TimeCut, trade::Trade, t::DateTime)::OptionQuote
    chain = at(cut, OptionQuote, trade.underlying, t)
    isempty(chain) &&
        error("resolve_quote: no chain at $t for $(trade)")
    for q in chain
        q.underlying  == trade.underlying  || continue
        q.strike      == trade.strike      || continue
        q.expiry      == trade.expiry      || continue
        q.option_type == trade.option_type || continue
        return q
    end
    error("resolve_quote: no matching quote in chain at $t for $(trade)")
end

"""
    run_backtest(agent::Agent, data::MarketData, from::DateTime, to::DateTime,
                 clock::Clock) -> Vector{Position}

Walk the ticks of `clock` in `[from, to]` (or the agent's `tick_times`
override when it returns one), ask the agent for the current policy,
ask that policy what to do, fill its orders, and return the ledger of
every fill. `data` is the opened reader map (see `with_data`).

Closes are emitted by the policy as counter-trades (opposite direction,
same contract); they accumulate in the ledger alongside the opens they
offset. "Currently-open net positions" is a view over the ledger, not a
separate collection.
"""
function run_backtest(
    agent::Agent,
    data::MarketData,
    from::DateTime,
    to::DateTime,
    clock::Clock,
)::Vector{Position}
    positions = Position[]
    # Sparse policies (e.g. once-a-day strangle entry on minute data) can
    # override `tick_times` to tell the engine the only timestamps that
    # matter, avoiding the cost of walking every clock tick and gating
    # inside `decide`. Default is the declared clock's grid.
    ticks = tick_times(agent, data, from, to)
    if ticks === nothing
        ticks = timestamps(data, clock, from, to)
    end
    for t in ticks
        cut    = TimeCut(data, t)
        policy = current_policy(agent, t, cut, positions)
        orders = decide(policy, t, cut, positions)
        for trd in orders
            qte = resolve_quote(cut, trd, t)
            # The fill spot is the trade's own underlying's spot; a
            # `spot_for` remap on the surface provider does not apply here.
            spot = only_or_missing(at(cut, SpotPrice, trd.underlying, t))
            ismissing(spot) &&
                error("run_backtest: missing spot at $t for fill of $(trd)")
            push!(positions, open_position(trd, qte, spot.price))
        end
    end
    return positions
end

"""
    run_backtest(policy::Policy, data::MarketData, from::DateTime, to::DateTime,
                 clock::Clock) -> Vector{Position}

Convenience overload for the fixed-policy case: wraps `policy` in a
`StaticAgent` and runs the standard agent-driven loop. The primary
caller is training / evaluation code that wants to score a single
candidate policy over a window without constructing an Agent.
"""
run_backtest(policy::Policy, data::MarketData, from::DateTime, to::DateTime, clock::Clock) =
    run_backtest(StaticAgent(policy), data, from, to, clock)
