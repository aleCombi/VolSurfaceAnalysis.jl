# Position: an immutable snapshot of a filled trade. Once a `Position` exists,
# everything needed to compute realized PnL against any future spot is frozen
# in its fields -- no chain, no surface, no broker connection required.

"""
    Position

Immutable record of a filled trade. Carries the entry-time snapshot
(`entry_price`, `entry_spot`, bid/ask, timestamp) so settlement is a pure
function of the position and the settlement spot.

# Fields
- `trade::Trade`
- `entry_price::Float64`  -- absolute USD per share, signed by direction at fill
- `entry_spot::Float64`   -- underlying price at entry (diagnostic; not used for PnL)
- `entry_bid::Union{Float64,Missing}`
- `entry_ask::Union{Float64,Missing}`
- `entry_timestamp::DateTime`
"""
struct Position
    trade::Trade
    entry_price::Float64
    entry_spot::Float64
    entry_bid::Union{Float64,Missing}
    entry_ask::Union{Float64,Missing}
    entry_timestamp::DateTime
end

"""
    open_position(trade::Trade, qte::OptionQuote, spot::Float64) -> Position

Fill `trade` against `qte`: longs cross the ask, shorts cross the bid. The
opposite side is recorded as-is (may be `missing`). `entry_timestamp` is
taken from `qte.timestamp`.

Throws `ArgumentError` when the quote does not describe the same contract
(underlying / strike / expiry / option type), or when the fill side
(`ask` for longs, `bid` for shorts) is `missing`.
"""
function open_position(trade::Trade, qte::OptionQuote, spot::Float64)::Position
    qte.underlying == trade.underlying ||
        throw(ArgumentError("quote underlying $(qte.underlying) != trade underlying $(trade.underlying)"))
    qte.strike == trade.strike ||
        throw(ArgumentError("quote strike $(qte.strike) != trade strike $(trade.strike)"))
    qte.expiry == trade.expiry ||
        throw(ArgumentError("quote expiry $(qte.expiry) != trade expiry $(trade.expiry)"))
    qte.option_type == trade.option_type ||
        throw(ArgumentError("quote option_type $(qte.option_type) != trade option_type $(trade.option_type)"))

    fill_price = trade.direction > 0 ? qte.ask : qte.bid
    ismissing(fill_price) &&
        throw(ArgumentError("fill side ($(trade.direction > 0 ? :ask : :bid)) is missing on quote"))

    return Position(trade, fill_price, spot, qte.bid, qte.ask, qte.timestamp)
end

"""
    entry_cost(position::Position) -> Float64

Signed cash flow at entry, in USD: positive when premium was paid (long),
negative when premium was received (short).
"""
entry_cost(p::Position)::Float64 =
    p.entry_price * p.trade.direction * p.trade.quantity

"""
    pnl(position::Position, settlement_spot::Float64) -> Float64

Realized PnL at expiry: `payoff(trade, settlement_spot) - entry_cost(position)`.
"""
pnl(p::Position, settlement_spot::Float64)::Float64 =
    payoff(p.trade, settlement_spot) - entry_cost(p)

"""
    pnl(positions::AbstractVector{Position}, settlement_spot::Float64) -> Float64

Sum of [`pnl`](@ref) across legs of a multi-leg structure cash-settled at the
same spot.
"""
pnl(positions::AbstractVector{Position}, settlement_spot::Float64)::Float64 =
    sum(pnl(p, settlement_spot) for p in positions; init=0.0)
