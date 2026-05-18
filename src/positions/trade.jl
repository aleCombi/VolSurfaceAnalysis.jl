# Trade: time-independent specification of an option contract to buy or sell.
# No spot, no entry timestamp, no price -- those belong on `Position` once a
# trade is filled. `payoff` lives here because it depends only on the contract
# and the spot at expiry; it has no concept of what was paid.

"""
    Trade

A single option contract spec: which contract, which side, how many. Pure
description -- a `Trade` can be constructed before any market is consulted.

# Fields
- `underlying::Underlying`
- `strike::Float64`
- `expiry::DateTime`
- `option_type::OptionType`
- `direction::Int`     -- `+1` long, `-1` short
- `quantity::Float64`  -- contracts; must be `> 0`
"""
struct Trade
    underlying::Underlying
    strike::Float64
    expiry::DateTime
    option_type::OptionType
    direction::Int
    quantity::Float64

    function Trade(underlying::Underlying, strike::Real, expiry::DateTime,
                   option_type::OptionType, direction::Int, quantity::Real)
        strike_val = Float64(strike)
        quantity_val = Float64(quantity)
        direction in (-1, 1) ||
            throw(ArgumentError("direction must be -1 or +1, got $direction"))
        quantity_val > 0.0 ||
            throw(ArgumentError("quantity must be positive, got $quantity"))
        strike_val > 0.0 ||
            throw(ArgumentError("strike must be positive, got $strike"))
        new(underlying, strike_val, expiry, option_type, direction, quantity_val)
    end
end

"""
    Trade(underlying, strike, expiry, option_type; direction=1, quantity=1.0)

Kwarg form with defaults: long, one contract.
"""
Trade(underlying::Underlying, strike::Real, expiry::DateTime,
      option_type::OptionType; direction::Int=1, quantity::Real=1.0) =
    Trade(underlying, strike, expiry, option_type, direction, quantity)

"""
    payoff(trade::Trade, spot_at_expiry::Real) -> Float64

Cash payoff at expiry: `intrinsic * direction * quantity`. Knows nothing
about what was paid -- useful for scenario analysis and payoff diagrams.
For realized PnL, see [`realized_pnl`](@ref).
"""
function payoff(trade::Trade, spot_at_expiry::Real)::Float64
    spot = Float64(spot_at_expiry)
    intrinsic = trade.option_type == Call ?
        max(spot - trade.strike, 0.0) :
        max(trade.strike - spot, 0.0)
    return intrinsic * trade.direction * trade.quantity
end
