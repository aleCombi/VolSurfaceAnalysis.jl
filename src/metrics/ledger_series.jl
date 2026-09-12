# Adapter from the event ledger to today's `PnLSeries`, so the existing
# metrics run unchanged on a ledger until slice 5 moves them onto the
# structure-level series and the equity curve. Metrics depend on the
# ledger, never the reverse.

"""
    pnl_series(L::Ledger; unit::Symbol = :structure) -> PnLSeries

The round trips of `L` as a [`PnLSeries`](@ref). With `unit = :structure`
(the default) one sample per `(group, closed_at)` with pnl summed, so a
multi-leg structure closed at one instant is one sample; with
`unit = :leg` one sample per round trip. Timestamps are `closed_at`;
`n_opens` and `n_closes` count `Open` and `Close` fills. Samples are
ordered by `(timestamp, pnl)` exactly as `pnl_series(positions)` orders
them, so path metrics read the same canonical order.

`window_end_spot` is `NaN` and `n_unmarked` is `0`: the ledger neither
force-settles nor skips an open lot (open lots stay open and are marked
by the equity curve), and both fields leave `PnLSeries` in slice 5.
"""
function pnl_series(L::Ledger; unit::Symbol = :structure)::PnLSeries
    trips = round_trips(L)
    timestamps = DateTime[]
    pnl        = Float64[]
    if unit == :structure
        keys_in_order = Tuple{Int,DateTime}[]
        acc = Dict{Tuple{Int,DateTime},Float64}()
        for r in trips
            key = (r.group, r.closed_at)
            haskey(acc, key) || push!(keys_in_order, key)
            acc[key] = get(acc, key, 0.0) + r.pnl
        end
        for key in keys_in_order
            push!(timestamps, key[2])
            push!(pnl, acc[key])
        end
    elseif unit == :leg
        for r in trips
            push!(timestamps, r.closed_at)
            push!(pnl, r.pnl)
        end
    else
        throw(ArgumentError("unit must be :structure or :leg, got $(repr(unit))"))
    end
    n_opens  = count(e -> e isa Fill && e.intent == Open,  L.events)
    n_closes = count(e -> e isa Fill && e.intent == Close, L.events)
    order = sortperm(eachindex(timestamps); by = i -> (timestamps[i], pnl[i]))
    # window_end_spot and n_unmarked leave PnLSeries in slice 5; until then
    # the ledger has no window-end mark and never skips a lot.
    return PnLSeries(timestamps[order], pnl[order], NaN, n_opens, n_closes, 0)
end
