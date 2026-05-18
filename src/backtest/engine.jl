# Backtest engine.
#
# Drives a `Strategy` across every available timestamp in `[from, to]`.
# Per tick: builds a `TimeCutModelDataSource` cut to `t`, asks the
# strategy for orders, fills each order against the chain at `t`, and
# appends the resulting `Position`s to the ledger. Returns the bare
# ledger -- reporting / PnL aggregation is a downstream concern.

using Dates

"""
    resolve_quote(cut::TimeCutModelDataSource, trade::Trade, t::DateTime)
        -> OptionQuote

Find the `OptionQuote` in the chain at `t` that matches `trade`'s
contract (underlying, expiry, strike, option_type). The match is exact
on all four fields; a strike-not-in-chain is treated as a programming
error (the strategy should only emit trades for contracts it can see).

Goes through `get_chain` rather than `get_surface` because surfaces
retain only inverted IVs -- the raw bid/ask the fill needs lives on the
chain quote, not the slice.
"""
function resolve_quote(cut::TimeCutModelDataSource, trade::Trade, t::DateTime)::OptionQuote
    chain = get_chain(cut, t)
    chain === nothing &&
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
    run_backtest(strategy::Strategy, source::ModelDataSource,
                 from::DateTime, to::DateTime) -> Vector{Position}

Walk every timestamp in `available_timestamps(source, from, to)`, ask the
strategy what to do, fill its orders, and return the ledger of every fill.

Closes are emitted by the strategy as counter-trades (opposite direction,
same contract); they accumulate in the ledger alongside the opens they
offset. "Currently-open net positions" is a view over the ledger, not a
separate collection.
"""
function run_backtest(
    strategy::Strategy,
    source::ModelDataSource,
    from::DateTime,
    to::DateTime,
)::Vector{Position}
    positions = Position[]
    for t in available_timestamps(source, from, to)
        cut = TimeCutModelDataSource(source, t)
        orders = decide(strategy, t, cut, positions)
        for trd in orders
            qte = resolve_quote(cut, trd, t)
            spot_val = get_spot(cut, t)
            ismissing(spot_val) &&
                error("run_backtest: missing spot at $t for fill of $(trd)")
            push!(positions, open_position(trd, qte, Float64(spot_val)))
        end
    end
    return positions
end
