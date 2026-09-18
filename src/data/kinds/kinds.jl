# The kinds: the record types, and the contract every one of them answers
# (`selector`, `selector_type`, `snapshot`). See the `data` module doc.

@enum OptionType Call Put

"""
    Underlying(ticker)

Selector for underlying-keyed kinds. The ticker is uppercase-normalized,
so `Underlying("spy") == Underlying("SPY")`.
"""
struct Underlying
    ticker::String
    Underlying(s::AbstractString) = new(uppercase(String(s)))
end

ticker(u::Underlying) = u.ticker
Base.show(io::IO, u::Underlying) = print(io, u.ticker)
# Content hash: the objectid default changes with every build.
Base.hash(u::Underlying, h::UInt) = hash(u.ticker, hash(:Underlying, h))
Base.:(==)(a::Underlying, b::Underlying) = a.ticker == b.ticker

"""
    Currency(code)

Selector for currency-keyed kinds. The code is uppercase-normalized, as
`Underlying`'s ticker is.
"""
struct Currency
    code::String
    Currency(s::AbstractString) = new(uppercase(String(s)))
end

Base.show(io::IO, c::Currency) = print(io, c.code)
Base.hash(c::Currency, h::UInt) = hash(c.code, hash(:Currency, h))   # content hash, as Underlying
Base.:(==)(a::Currency, b::Currency) = a.code == b.code

"""
    OptionQuote

One contract's quote at `timestamp`. `bid`, `ask`, `mark`, `iv`,
`open_interest` and `volume` are `missing` when the source did not
carry them.
"""
struct OptionQuote
    instrument_id::String
    underlying::Underlying
    expiry::DateTime
    strike::Float64
    option_type::OptionType
    bid::Union{Float64,Missing}
    ask::Union{Float64,Missing}
    mark::Union{Float64,Missing}
    iv::Union{Float64,Missing}
    open_interest::Union{Float64,Missing}
    volume::Union{Float64,Missing}
    timestamp::DateTime
end

"""
    SpotPrice

The underlying's price at `timestamp`. Never carries a missing price:
a row without one is not an observation. Does not record its session:
a source may print outside regular hours, and nothing here says so.
"""
struct SpotPrice
    underlying::Underlying
    price::Float64
    timestamp::DateTime
end

"""
    OptionBar

One contract's OHLCV minute bar, with the contract identity so it can
become an `OptionQuote` through [`synthesize`](@ref) without a lookup.
Any OHLCV field the source did not carry is `missing`.
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

# --- curve kinds ------------------------------------------------------------
# A `Curve` stamped with a visibility time and a selector. The two-argument
# constructors stamp the start of time: "always known".

"""
    RateCurve(currency, curve[, timestamp])

The rate `Curve` for `currency` as of `timestamp`, which defaults to the
start of time.
"""
struct RateCurve
    currency::Currency
    curve::Curve
    timestamp::DateTime
end
RateCurve(currency::Currency, curve::Curve) = RateCurve(currency, curve, typemin(DateTime))

"""
    DivCurve(underlying, curve[, timestamp])

The dividend-yield `Curve` for `underlying` as of `timestamp`, which
defaults to the start of time.
"""
struct DivCurve
    underlying::Underlying
    curve::Curve
    timestamp::DateTime
end
DivCurve(underlying::Underlying, curve::Curve) = DivCurve(underlying, curve, typemin(DateTime))

"""
    selector(r) -> selector value

The value that distinguishes parallel series of `r`'s kind. Every kind
defines exactly one method.
"""
function selector end

"""
    selector_type(::Type{R}) -> Type

The type of `selector(r)` for records of kind `R`; a trait on the kind,
so a selector can be checked without a record in hand.
"""
function selector_type end

"""
    snapshot(::Type{R}) -> Bool

Whether kind `R` holds one record per selector per instant (`true`) or
many (`false`, a grid kind). A trait on the kind. For a snapshot kind,
two records for one selector at one instant are either the same record
twice, which collapses where the rows enter, or a `ConflictingRecords`.
Every reader that can be handed a duplicate applies the rule, fixtures
included, so a test cannot construct a state a real reader refuses.
"""
function snapshot end

selector(r::OptionBar)   = r.underlying
selector(r::OptionQuote) = r.underlying
selector(r::SpotPrice)   = r.underlying

selector_type(::Type{OptionBar})   = Underlying
selector_type(::Type{OptionQuote}) = Underlying
selector_type(::Type{SpotPrice})   = Underlying

snapshot(::Type{OptionBar})   = false
snapshot(::Type{OptionQuote}) = false
snapshot(::Type{SpotPrice})   = true
selector(r::RateCurve) = r.currency
selector(r::DivCurve)  = r.underlying
selector_type(::Type{RateCurve}) = Currency
selector_type(::Type{DivCurve})  = Underlying
snapshot(::Type{RateCurve}) = true
snapshot(::Type{DivCurve})  = true
