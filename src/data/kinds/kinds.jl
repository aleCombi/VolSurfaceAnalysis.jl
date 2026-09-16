# The kinds: what a market datum IS.
#
# A kind is a plain immutable record type. Every kind answers two questions
# the protocol depends on:
#
# - `timestamp::DateTime` is VISIBILITY time: the moment the record became
#   knowable. The time cut filters on it and on nothing else. Any other date
#   a record carries (expiry, ex-date, effective date) is an ordinary field.
#   A record read off a minute bar is stamped at BAR END -- the vendor's
#   bar-open stamp plus the bar interval -- so a decision at `t` reads the
#   completed minute and not one still running (see docs/modules/data.md).
# - `selector(r)` is the field that distinguishes parallel series of the
#   same kind (`Underlying` for market data, `Currency` for rate curves), and
#   `selector_type(R)` is its type. Together they are what `BySelector`,
#   `Clock`, `Constant` and the config loader check.

@enum OptionType Call Put

struct Underlying
    ticker::String
    Underlying(s::AbstractString) = new(uppercase(String(s)))
end

ticker(u::Underlying) = u.ticker
Base.show(io::IO, u::Underlying) = print(io, u.ticker)
# Content hash, explicitly: the default falls back to objectid, which for a
# type in a precompiled package changes with every build, so a Dict keyed
# on Underlying would iterate in a build-dependent order.
Base.hash(u::Underlying, h::UInt) = hash(u.ticker, hash(:Underlying, h))
Base.:(==)(a::Underlying, b::Underlying) = a.ticker == b.ticker

"""
    Currency(code)

Selector for currency-keyed kinds (rate curves). The code is
uppercase-normalized, mirroring `Underlying`, so `Currency("usd") ==
Currency("USD")`.
"""
struct Currency
    code::String
    Currency(s::AbstractString) = new(uppercase(String(s)))
end

Base.show(io::IO, c::Currency) = print(io, c.code)
Base.hash(c::Currency, h::UInt) = hash(c.code, hash(:Currency, h))   # content hash, as Underlying
Base.:(==)(a::Currency, b::Currency) = a.code == b.code

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

struct SpotPrice
    underlying::Underlying
    price::Float64
    timestamp::DateTime
end

"""
    OptionBar

Faithful mirror of one Massive options OHLCV minute-bar row. Carries the
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

# --- curve kinds ------------------------------------------------------------
# A curve as a market-data record: the curve *as of* `timestamp` (visibility
# time), evaluated at a maturity by calling it. Snapshot kinds: they hold
# until superseded, so consumers read them with `only_or_missing(asof(...))`.
# The two-argument constructors stamp the start of time, the `Constant` case.

"""
    RateCurve(currency, curve[, timestamp])

The rate `Curve` for `currency` as of `timestamp` (visibility time;
defaults to the start of time, "always known"). Selector: the currency.
"""
struct RateCurve
    currency::Currency
    curve::Curve
    timestamp::DateTime
end
RateCurve(currency::Currency, curve::Curve) = RateCurve(currency, curve, typemin(DateTime))

"""
    DivCurve(underlying, curve[, timestamp])

The dividend-yield `Curve` for `underlying` as of `timestamp`. Selector:
the underlying.
"""
struct DivCurve
    underlying::Underlying
    curve::Curve
    timestamp::DateTime
end
DivCurve(underlying::Underlying, curve::Curve) = DivCurve(underlying, curve, typemin(DateTime))

"""
    selector(r) -> selector value

The value that distinguishes parallel series of `r`'s kind (the
`Underlying` of a bar, quote or spot; the `Currency` of a rate curve).
Every kind defines exactly one method.
"""
function selector end

"""
    selector_type(::Type{R}) -> Type

The type of `selector(r)` for records of kind `R`. A trait on the kind,
used to check selectors at construction time (`Clock`, `BySelector`) and
in the config loader.
"""
function selector_type end

"""
    snapshot(::Type{R}) -> Bool

Whether kind `R` carries ONE record per selector per instant. A trait on
the kind. `true` for kinds read through `only_or_missing` (a spot, a
curve, a surface): two rows for one selector at one instant are then
either the same row twice, which collapses, or two answers, which is a
`ConflictingRecords`. `false` for grid kinds (bars, quotes), where many
rows per instant is the shape. Readers that can be handed a duplicate
apply the rule where the rows enter -- the parquet spot reader after its
sort, `InMemory` at construction -- so a fixture cannot represent a state
the real reader throws on.
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
