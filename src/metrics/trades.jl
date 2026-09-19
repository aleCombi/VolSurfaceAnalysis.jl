# trade_pnl: the ledger's round trips as one dollar figure per closed trade.

"""
    trade_pnl(L::Ledger; unit::Symbol = :structure) -> Vector{Float64}

The realised outcome of each closed trade in `L`, in USD. With
`unit = :structure` (the default) one entry per `(group, closed_at)` with
cents summed before the conversion, so a strangle closed at one instant is
one trade; with `unit = :leg` one entry per round trip.

The pairing is the ledger's own (`Match` and `Expiry` under a named
rule), never inferred here. Cents are summed as integers and cross to
dollars once, at [`cents_to_usd`](@ref). Entries are ordered by
`(closed_at, pnl)`, losses first within an instant, so a run's trade
vector is deterministic and reconstructible.

Open lots contribute nothing: a lot that has not been consumed has no
realised outcome. What it is *worth* is the marked curve's question, not
this one.
"""
function trade_pnl(L::Ledger; unit::Symbol = :structure)::Vector{Float64}
    trips = round_trips(L)
    closed_at = DateTime[]
    cents     = Int[]
    if unit == :structure
        keys_in_order = Tuple{Int,DateTime}[]
        acc = Dict{Tuple{Int,DateTime},Int}()
        for r in trips
            key = (r.group, r.closed_at)
            haskey(acc, key) || push!(keys_in_order, key)
            acc[key] = get(acc, key, 0) + r.pnl
        end
        for key in keys_in_order
            push!(closed_at, key[2]); push!(cents, acc[key])
        end
    elseif unit == :leg
        for r in trips
            push!(closed_at, r.closed_at); push!(cents, r.pnl)
        end
    else
        throw(ArgumentError("unit must be :structure or :leg, got $(repr(unit))"))
    end
    usd = Float64[cents_to_usd(c) for c in cents]
    order = sortperm(eachindex(closed_at); by = i -> (closed_at[i], usd[i]))
    return usd[order]
end
