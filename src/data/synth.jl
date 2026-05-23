# This file is part of the data-sourcing adapter layer. It exists because
# the Polygon options parquet store carries only OHLCV per minute bar; there
# is no bid/ask feed. To let downstream code that requires a fillable quote
# (positions, backtest) run against this store, we declare a small
# `QuoteSynthesizer` strategy that turns an `OptionBar` -- a faithful mirror
# of Polygon's parquet row schema -- into an `OptionQuote`. The data source
# CARRIES a synthesizer and consults it at row-load time; it does not
# hardcode one. When a real bid/ask feed lands, a different synthesizer
# (or a different `DataSource` entirely) takes its place without touching
# anything downstream.

"""
    OptionBar

Faithful mirror of one Polygon options OHLCV minute-bar row. Carries the
contract identity (so it can be turned into an `OptionQuote` without an
extra lookup) plus the raw `open`/`high`/`low`/`close`/`volume` fields.

This is an adapter-layer type. Production downstream code should consume
`OptionQuote`s produced via [`synthesize`](@ref); `OptionBar` exists so
the synthesis policy is explicit and testable instead of buried inside
the parquet reader.

# Fields
- `instrument_id::String`
- `underlying::Underlying`
- `expiry::DateTime`
- `strike::Float64`
- `option_type::OptionType`
- `open::Union{Float64,Missing}`
- `high::Union{Float64,Missing}`
- `low::Union{Float64,Missing}`
- `close::Union{Float64,Missing}`
- `volume::Union{Float64,Missing}`
- `timestamp::DateTime`
"""
struct OptionBar
    instrument_id::String
    underlying::Underlying
    expiry::DateTime
    strike::Float64
    option_type::OptionType
    open::Union{Float64,Missing}
    high::Union{Float64,Missing}
    low::Union{Float64,Missing}
    close::Union{Float64,Missing}
    volume::Union{Float64,Missing}
    timestamp::DateTime
end

"""
    QuoteSynthesizer

Abstract strategy type: turns an [`OptionBar`](@ref) into an
[`OptionQuote`](@ref). A data source whose underlying record is a bar
declares the synthesizer it uses at construction time; the synthesizer
encodes the policy by which raw OHLCV becomes a fillable bid/ask.

Concrete subtypes implement `synthesize(s, bar)::OptionQuote`.
"""
abstract type QuoteSynthesizer end

"""
    synthesize(s::QuoteSynthesizer, bar::OptionBar) -> OptionQuote

Project a bar through the synthesizer policy into a quote. Implementations
must return an `OptionQuote` carrying `bar`'s contract identity, timestamp,
and `volume` unchanged.
"""
function synthesize end

"""
    SpreadFromOHLCV(λ)

Synthesize bid/ask from an OHLCV bar by interpolating between the bar's
extreme range and its close:

    bid  = low  + λ · (close − low)
    ask  = high − λ · (high − close)
    mark = close

`λ` tightens the synthesized spread around `close`:

- `λ = 0.0` → `bid = low`, `ask = high` (widest, most conservative fill).
- `λ = 0.7` → canonical default used across this project's experiments.
- `λ = 1.0` → `bid = ask = close` (midpoint, zero spread).

`λ` is required at the type level; there is no default. Strategies and
configs that want the canonical 0.7 spell it out, so the fill policy is
always visible in the experiment record.

Missing-data policy: if `high`, `low`, or `close` is `missing`, the
synthesized `bid` and `ask` are `missing` (and `mark = close` if present).
Downstream `open_position` will then throw on the missing fill side -- a
silent zero-spread fallback would invent a market that did not trade.

Throws `ArgumentError` when `λ` is outside `[0, 1]`.
"""
struct SpreadFromOHLCV <: QuoteSynthesizer
    lambda::Float64
    function SpreadFromOHLCV(λ::Real)
        0.0 <= λ <= 1.0 ||
            throw(ArgumentError("SpreadFromOHLCV lambda must be in [0, 1], got $λ"))
        new(Float64(λ))
    end
end

function synthesize(s::SpreadFromOHLCV, bar::OptionBar)::OptionQuote
    mark = bar.close
    if ismissing(bar.high) || ismissing(bar.low) || ismissing(bar.close)
        bid = missing
        ask = missing
    else
        λ = s.lambda
        bid = bar.low  + λ * (bar.close - bar.low)
        ask = bar.high - λ * (bar.high - bar.close)
    end
    return OptionQuote(
        bar.instrument_id, bar.underlying, bar.expiry, bar.strike, bar.option_type,
        bid, ask, mark, missing, missing, bar.volume, bar.timestamp,
    )
end
