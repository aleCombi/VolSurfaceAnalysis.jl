# PnLSeries: the canonical per-round-trip PnL intermediate that every
# metric reads from. It is built by `pnl_series(::Ledger)` in
# `ledger_series.jl`, from the ledger's round trips, one sample per
# structure closed at one instant. `window_end_spot` and `n_unmarked` are
# placeholders until slice 5 replaces the series with the structure
# series and the equity curve: the ledger neither marks nor skips an
# open lot, so the adapter fills them with `NaN` and `0`.

"""
    PnLSeries

Per-round-trip realized-PnL series, the canonical intermediate every
metric in this module reads from. Built from a ledger by
[`pnl_series(::Ledger)`](@ref): one sample per structure (group) closed
at one instant, in USD, ordered by `(timestamp, pnl)`.

# Fields
- `timestamps::Vector{DateTime}` -- one entry per sample, the closing instant.
- `pnl::Vector{Float64}` -- realized PnL of that sample, in USD.
- `window_end_spot::Float64` -- placeholder until slice 5; `NaN` from the ledger.
- `n_opens::Int` -- count of `Open` fills in the ledger.
- `n_closes::Int` -- count of `Close` fills in the ledger.
- `n_unmarked::Int` -- placeholder until slice 5; `0` from the ledger.
"""
struct PnLSeries
    timestamps::Vector{DateTime}
    pnl::Vector{Float64}
    window_end_spot::Float64
    n_opens::Int
    n_closes::Int
    n_unmarked::Int
end

"""
    equity_curve(series::PnLSeries) -> Vector{Float64}

Cumulative realized PnL in chronological order: `cumsum(series.pnl)`.
Empty input returns an empty vector.
"""
equity_curve(s::PnLSeries)::Vector{Float64} = cumsum(s.pnl)
