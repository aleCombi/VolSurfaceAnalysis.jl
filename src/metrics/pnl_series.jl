# PnLSeries: the canonical per-round-trip PnL intermediate that every
# metric reads from. Built from a `Vector{Position}` ledger by
# FIFO-matching each closing counter-trade against the earliest open lot
# on the same contract. Any open lot left at the end of the walk is
# marked to a caller-supplied settlement spot, so "still open at end of
# window" remains a first-class outcome rather than being silently
# discarded.
#
# Why round-trip aggregation and not per-Position PnL: the engine
# records closes as counter-trades (own `Position` rows whose
# `realized_pnl(p, spot)` is the *payoff* of the close leg, not its
# contribution to a closed round trip). Summing `realized_pnl` across
# rows double-counts. The intermediate sits one layer above and emits
# one number per round trip plus one number per still-open residual.

"""
    PnLSeries

Per-round-trip realized-PnL series, the canonical intermediate every
metric in this module reads from.

A "round trip" is either:

- an open lot fully matched (FIFO) by one or more closing counter-trades
  on the same contract -- one entry per matched chunk, timestamped at
  the close fill, PnL `= (-_unit_cost(open) - _unit_cost(close)) * qty`; or
- an open lot still outstanding at the end of the ledger -- one entry
  per residual chunk, PnL `= (_unit_payoff(open, settlement_spot) -
  _unit_cost(open)) * qty`, timestamped at `settlement_timestamp` (or
  the leg's expiry if no timestamp was supplied).

# Fields
- `timestamps::Vector{DateTime}` -- one entry per round trip (close
  timestamp, or settlement timestamp for still-open residuals).
- `pnl::Vector{Float64}` -- realized PnL of that round trip, in USD.
- `settlement_spot::Float64` -- spot used to mark any still-open residual.
- `n_opens::Int` -- raw count of opening fills in the ledger.
- `n_closes::Int` -- raw count of closing fills in the ledger.
"""
struct PnLSeries
    timestamps::Vector{DateTime}
    pnl::Vector{Float64}
    settlement_spot::Float64
    n_opens::Int
    n_closes::Int
end

_contract_key(p::Position) =
    (p.trade.underlying, p.trade.strike, p.trade.expiry, p.trade.option_type)

_unit_cost(p::Position) = p.entry_price * p.trade.direction

function _unit_payoff(p::Position, spot::Real)
    s = Float64(spot)
    K = p.trade.strike
    intrinsic = p.trade.option_type == Call ? max(s - K, 0.0) : max(K - s, 0.0)
    return intrinsic * p.trade.direction
end

# Mutable lot in the FIFO queue: the originating opening Position plus
# how much of its quantity is still unmatched.
mutable struct _OpenLot
    pos::Position
    remaining::Float64
end

"""
    pnl_series(positions::AbstractVector{Position}, settlement_spot::Real;
               settlement_timestamp::Union{DateTime,Nothing}=nothing) -> PnLSeries

Aggregate `positions` into a per-round-trip PnL series.

Fills are grouped by contract `(underlying, strike, expiry, option_type)`
and walked in `entry_timestamp` order. Each fill either extends the
open same-side lots on that contract or closes (FIFO) against the
oldest opposite-side lots. Each match emits one entry. Any lot still
outstanding at the end is marked to `settlement_spot` and emitted as
a residual entry.

`settlement_timestamp` is the timestamp recorded on residual entries;
defaults to the leg's `trade.expiry`. The experiment orchestrator
passes its window end so residual marks are honestly stamped at
"mark-to-spot at `exp.to`" rather than at the contract expiry.

Permissive on direction sequence: any fill that doesn't match
opposing lots simply becomes a new lot on its own side. The metrics
layer cannot tell a "first short open" from an "orphan close" -- the
ledger has no such marking -- so neither is special-cased.
"""
function pnl_series(positions::AbstractVector{Position}, settlement_spot::Real;
                    settlement_timestamp::Union{DateTime,Nothing}=nothing)::PnLSeries
    settle = Float64(settlement_spot)

    by_contract = Dict{Tuple{Underlying,Float64,DateTime,OptionType},Vector{Position}}()
    n_opens = 0
    n_closes = 0
    for p in positions
        push!(get!(() -> Position[], by_contract, _contract_key(p)), p)
    end

    timestamps = DateTime[]
    pnl        = Float64[]

    for (_, fills) in by_contract
        sort!(fills, by = p -> p.entry_timestamp)
        lots = _OpenLot[]
        side = 0   # +1 if open lots are long, -1 if short, 0 if empty
        for fill in fills
            if isempty(lots) || fill.trade.direction == side
                # extend the same-side queue
                push!(lots, _OpenLot(fill, fill.trade.quantity))
                side = fill.trade.direction
                n_opens += 1
            else
                # closing counter-trade: FIFO-match against opposing lots
                remaining = fill.trade.quantity
                n_closes += 1
                while remaining > 0 && !isempty(lots)
                    lot = lots[1]
                    matched = min(remaining, lot.remaining)
                    push!(timestamps, fill.entry_timestamp)
                    push!(pnl,
                          (-_unit_cost(lot.pos) - _unit_cost(fill)) * matched)
                    lot.remaining -= matched
                    remaining     -= matched
                    if lot.remaining == 0
                        popfirst!(lots)
                    end
                end
                if remaining > 0
                    # overshoot: the close fully consumed the opposing lots
                    # and the residual flips the net position to its own side
                    push!(lots, _OpenLot(fill, remaining))
                    side = fill.trade.direction
                else
                    isempty(lots) && (side = 0)
                end
            end
        end
        # residual still-open lots: mark to settlement_spot
        for lot in lots
            ts = settlement_timestamp === nothing ? lot.pos.trade.expiry :
                                                    settlement_timestamp
            push!(timestamps, ts)
            push!(pnl,
                  (_unit_payoff(lot.pos, settle) - _unit_cost(lot.pos)) * lot.remaining)
        end
    end

    # Sort the combined series by timestamp so equity_curve is monotonic
    # in time across contracts.
    order = sortperm(timestamps)
    return PnLSeries(timestamps[order], pnl[order], settle, n_opens, n_closes)
end

"""
    equity_curve(series::PnLSeries) -> Vector{Float64}

Cumulative realized PnL in chronological order: `cumsum(series.pnl)`.
Empty input returns an empty vector.
"""
equity_curve(s::PnLSeries)::Vector{Float64} = cumsum(s.pnl)
