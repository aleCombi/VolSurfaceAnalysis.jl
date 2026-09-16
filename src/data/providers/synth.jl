# `data/providers`: turning a bar into a fillable quote.

"""
    QuoteSynthesizer

Abstract strategy type: turns an [`OptionBar`](@ref) into an
[`OptionQuote`](@ref). Concrete subtypes implement
`synthesize(s, bar)::OptionQuote`.
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
- `λ = 0.7` → the value this project's configs use.
- `λ = 1.0` → `bid = ask = close` (midpoint, zero spread).

`λ` is required at the type level; there is no default.

If `high`, `low` or `close` is `missing` the synthesized `bid` and `ask`
are `missing` too, and `mark = close` if present. Not zero spread: a
fallback there would invent a market that did not trade.

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
