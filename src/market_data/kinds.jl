# `market_data` module: kinds and the selector contract.
#
# A kind is a plain immutable record type that says *what* a datum is. The
# record types themselves live where they are defined today (`OptionBar`,
# `OptionQuote`, `SpotPrice` in `data/`); this file adds the two things the
# protocol depends on for every kind:
#
# - `timestamp::DateTime` is VISIBILITY time: the moment the record became
#   knowable. The time cut filters on it and on nothing else. Any other date
#   a record carries (expiry, ex-date, effective date) is an ordinary field.
#   Polygon minute bars keep their bar-open stamp as the visibility time, a
#   documented one-minute allowance (see docs/modules/market_data.md).
# - `selector(r)` is the field that distinguishes parallel series of the
#   same kind (`Underlying` for market data, `Currency` for rate curves), and
#   `selector_type(R)` is its type. Together they are what `BySelector`,
#   `Clock`, `Constant` and the config loader check.

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
