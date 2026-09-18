# The simulated venue: the fill rule (a quote to a leg price), the cost
# model (an order's commission) and the tick.

# ---- price rules ------------------------------------------------------

# Fill prices land on the tick, rounded away from the trader. The
# tolerance absorbs the binary noise of `price * 100` (1.07 * 100 is
# 107.00000000000001), so a quote already on the tick is unchanged.
const _TICK_NOISE = 1e-6

# One cent at every premium for every underlying the contract table lists
# (the backtest module doc cites the penny program). A parameter of
# `fill_price`, `fill_legs` and `check_join` so the join check can state
# the tick it recomputes against.
const TICK_CENTS = 1

function _cross_spread(bid, ask, side::Side, tick_cents::Int)
    raw = side == Long ? ask : bid              # a buy crosses to the ask, a sale to the bid
    ismissing(raw) && return missing
    cents = Float64(raw) * 100
    ticks = side == Long ? ceil(cents / tick_cents - _TICK_NOISE) :
                           floor(cents / tick_cents + _TICK_NOISE)
    return ticks * tick_cents / 100
end

const _FILL_RULES = Dict{Symbol,Function}(:cross_spread => _cross_spread)

"""
    fill_price(rule::Symbol, bid, ask, side::Side, tick_cents::Int = TICK_CENTS) -> Union{Float64,Missing}

The per-share price a leg of `side` fills at under `rule`, from the raw
`bid` and `ask` (either may be `missing`) and the class's tick in cents,
which defaults to [`TICK_CENTS`](@ref). Takes raw values so `check_join`
can recompute it from an observation. `missing` when the side the rule
needs is missing. Errors, naming the known rules, for an unknown `rule`;
`tick_cents` must be positive.

`:cross_spread`: `Long` takes the ask, `Short` the bid, rounded onto the
tick away from the trader (a buy up, a sale down); a price already on
the tick is unchanged, and every result passes `contract_cents` for a
listed underlying.
"""
function fill_price(rule::Symbol, bid, ask, side::Side, tick_cents::Int = TICK_CENTS)
    tick_cents >= 1 || throw(ArgumentError("fill_price: tick_cents must be positive, got $tick_cents"))
    f = get(_FILL_RULES, rule) do
        error("fill_price: unknown fill rule :$rule. Known: $(sort(collect(keys(_FILL_RULES))))")
    end
    return f(bid, ask, side, tick_cents)
end

# ---- cost models ------------------------------------------------------

_no_commission(prices, quantities)::Vector{Int} = zeros(Int, length(prices))

# IBKR Pro, US options, fixed rate per contract at monthly volume of
# 10,000 contracts or fewer, by premium per share. From
# https://www.interactivebrokers.com/en/pricing/commissions-options.php
# as fetched on 2026-09-12:
#
#   premium < USD 0.05                 USD 0.25 per contract
#   USD 0.05 <= premium < USD 0.10     USD 0.50 per contract
#   premium >= USD 0.10                USD 0.65 per contract
#   minimum per order                  USD 1.00
#
# The page's own examples at this tier (1 contract at USD 2 premium is
# USD 1.00; 2 at USD 5, USD 1.30; 3 at USD 0.075, USD 1.50; 5 at USD
# 0.03, USD 1.25) are the test literals.
function _ibkr_rate_cents(price::Real)::Int
    price < 0.05 && return 25
    price < 0.10 && return 50
    return 65
end

function _ibkr_pro_us_options(prices, quantities)::Vector{Int}
    n = length(prices)
    n == 0 && return Int[]
    unraised = Int[_ibkr_rate_cents(prices[k]) * Int(quantities[k]) for k in 1:n]
    total = sum(unraised)
    charged = max(total, 100)                   # the USD 1.00 minimum per order
    # Shared over the legs in whole cents by cumulative rounding in
    # proportion to each leg's unraised amount (the rule `round_trips`
    # uses for fee shares), so the shares sum to `charged` exactly. The
    # rational keeps the quotient exact; round(Int, x) is to nearest,
    # ties to even.
    shares = zeros(Int, n)
    cumulative, before = 0, 0
    for k in 1:n
        cumulative += unraised[k]
        after = round(Int, charged * cumulative // total)
        shares[k] = after - before
        before = after
    end
    return shares
end

const _COST_MODELS = Dict{Symbol,Function}(:none => _no_commission,
                                           :ibkr_pro_us_options => _ibkr_pro_us_options)

"""
    commission(model::Symbol, prices::AbstractVector{<:Real},
               quantities::AbstractVector{<:Integer}) -> Vector{Int}

The commission of one order under `model`, as non-negative whole cents
per leg; `fill_legs` negates them into `Fee` amounts. `prices` are per
share and `quantities` are contracts, one per leg. Errors, naming the
known models, for an unknown `model`.

`:none` returns zeros. `:ibkr_pro_us_options` is IBKR Pro's fixed-rate
schedule for US options at the lowest monthly-volume tier (10,000
contracts a month or fewer; a ten-year daily strangle trades about 5,000
in total), from
<https://www.interactivebrokers.com/en/pricing/commissions-options.php>
as fetched on 2026-09-12: USD 0.25 per contract below a premium of USD
0.05 per share, USD 0.50 from 0.05 to below 0.10, USD 0.65 at 0.10 and
above, with a minimum of USD 1.00 per order. The order's commission is
the sum over legs of its rate times its quantity, raised to the minimum
when below it, then shared over the legs in whole cents by cumulative
rounding in proportion to each leg's unraised amount. Higher volume
tiers and third-party fees (exchange, ORF, OCC clearing, FINRA CAT, SEC)
are not modelled.
"""
function commission(model::Symbol, prices::AbstractVector{<:Real},
                    quantities::AbstractVector{<:Integer})::Vector{Int}
    length(prices) == length(quantities) || throw(ArgumentError(
        "commission: $(length(prices)) prices for $(length(quantities)) quantities"))
    all(q -> q > 0, quantities) || throw(ArgumentError("commission: quantities must be positive"))
    f = get(_COST_MODELS, model) do
        error("commission: unknown cost model :$model. Known: $(sort(collect(keys(_COST_MODELS))))")
    end
    return f(prices, quantities)
end
