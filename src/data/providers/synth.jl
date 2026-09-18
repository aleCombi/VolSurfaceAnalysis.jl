# `data/providers`: turning a bar into a fillable quote.

"""
    QuoteSynthesizer

Abstract policy type: how an [`OptionBar`](@ref) becomes an
[`OptionQuote`](@ref). Concrete subtypes implement `synthesize`.
"""
abstract type QuoteSynthesizer end

"""
    synthesize(s::QuoteSynthesizer, bar::OptionBar) -> OptionQuote

The quote `s` reads off `bar`. Carries `bar`'s contract identity,
timestamp and `volume` unchanged.
"""
function synthesize end

"""
    SpreadFromOHLCV(λ)

Bid/ask interpolated between the bar's range and its close, with `mark =
close`:

    bid = low  + λ · (close − low)
    ask = high − λ · (high − close)

`λ = 0` is the full range, `λ = 1` a zero spread at the close. When
`high`, `low` or `close` is `missing`, so are `bid` and `ask`. Throws
`ArgumentError` when `λ` is outside `[0, 1]`.
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
