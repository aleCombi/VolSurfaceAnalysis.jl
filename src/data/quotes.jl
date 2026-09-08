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
