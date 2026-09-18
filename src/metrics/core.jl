# The always-on metrics: computed on every call, never requested.

"""
    total_pnl(trades::AbstractVector{<:Real}) -> Float64

Realised PnL across every closed trade, in USD. Returns `0.0` when
nothing closed. The realised total, never the curve's last level.
"""
total_pnl(trades::AbstractVector{<:Real})::Float64 = sum(trades; init=0.0)

"""
    n_round_trips(trades::AbstractVector{<:Real}) -> Int

Number of closed trades, at the grouping `trades` was built with
(structures by default, legs under `unit = :leg`).
"""
n_round_trips(trades::AbstractVector{<:Real})::Int = length(trades)

"""
    hit_rate(trades::AbstractVector{<:Real}) -> Float64

Fraction of closed trades with strictly positive PnL. Returns `NaN` when
nothing closed -- hit rate is genuinely undefined with no trades, and
`NaN` propagates through downstream math rather than silently reading as
"0% wins". Breakeven trades (PnL exactly zero) are not wins.
"""
function hit_rate(trades::AbstractVector{<:Real})::Float64
    n = length(trades)
    n == 0 && return NaN
    return count(>(0), trades) / n
end

"""
    n_opens(L::Ledger) -> Int

`Open` fills in the ledger.
"""
n_opens(L::Ledger)::Int = count(e -> e isa Fill && e.intent == Open, L.events)

"""
    n_closes(L::Ledger) -> Int

`Close` fills in the ledger. The twin of [`n_opens`](@ref); note that an
expiry is not a closing fill, so a book that expired rather than traded
out reports opens with no closes.
"""
n_closes(L::Ledger)::Int = count(e -> e isa Fill && e.intent == Close, L.events)
