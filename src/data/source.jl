abstract type DataSource end

function available_timestamps end
function get_chain end
function get_spot end
function get_spots end
function clear_cache! end

clear_cache!(::DataSource) = nothing

struct InMemoryDataSource <: DataSource
    underlying::Underlying
    chain_timestamps::Vector{DateTime}
    chains::Vector{Vector{OptionQuote}}
    spot_timestamps::Vector{DateTime}
    spot_prices::Vector{Float64}

    function InMemoryDataSource(
        underlying::Underlying,
        chain_timestamps::Vector{DateTime},
        chains::Vector{Vector{OptionQuote}},
        spot_timestamps::Vector{DateTime},
        spot_prices::Vector{Float64},
    )
        length(chain_timestamps) == length(chains) ||
            throw(ArgumentError("chain_timestamps and chains must have equal length"))
        length(spot_timestamps) == length(spot_prices) ||
            throw(ArgumentError("spot_timestamps and spot_prices must have equal length"))
        issorted(chain_timestamps) ||
            throw(ArgumentError("chain_timestamps must be sorted"))
        issorted(spot_timestamps) ||
            throw(ArgumentError("spot_timestamps must be sorted"))
        allunique(chain_timestamps) ||
            throw(ArgumentError("chain_timestamps must be unique"))
        allunique(spot_timestamps) ||
            throw(ArgumentError("spot_timestamps must be unique"))
        new(underlying, chain_timestamps, chains, spot_timestamps, spot_prices)
    end
end

function InMemoryDataSource(
    underlying::Union{Underlying,AbstractString};
    chains::AbstractDict{DateTime,Vector{OptionQuote}}=Dict{DateTime,Vector{OptionQuote}}(),
    spots::AbstractDict{DateTime,Float64}=Dict{DateTime,Float64}(),
)
    u = underlying isa Underlying ? underlying : Underlying(underlying)
    chain_ts = sort(collect(keys(chains)))
    chain_arr = Vector{OptionQuote}[chains[t] for t in chain_ts]
    spot_ts = sort(collect(keys(spots)))
    spot_arr = Float64[spots[t] for t in spot_ts]
    InMemoryDataSource(u, chain_ts, chain_arr, spot_ts, spot_arr)
end

available_timestamps(s::InMemoryDataSource) = s.chain_timestamps

function available_timestamps(s::InMemoryDataSource, from::DateTime, to::DateTime)
    lo = searchsortedfirst(s.chain_timestamps, from)
    hi = searchsortedlast(s.chain_timestamps, to)
    s.chain_timestamps[lo:hi]
end

function get_chain(s::InMemoryDataSource, ts::DateTime)::Union{Vector{OptionQuote},Nothing}
    i = searchsortedfirst(s.chain_timestamps, ts)
    (i <= length(s.chain_timestamps) && s.chain_timestamps[i] == ts) || return nothing
    return s.chains[i]
end

function get_spot(s::InMemoryDataSource, ts::DateTime)::Union{Float64,Missing}
    i = searchsortedfirst(s.spot_timestamps, ts)
    (i <= length(s.spot_timestamps) && s.spot_timestamps[i] == ts) || return missing
    return s.spot_prices[i]
end

function get_spots(s::InMemoryDataSource, from::DateTime, to::DateTime)::Vector{SpotPrice}
    lo = searchsortedfirst(s.spot_timestamps, from)
    hi = searchsortedlast(s.spot_timestamps, to)
    [SpotPrice(s.underlying, s.spot_prices[i], s.spot_timestamps[i]) for i in lo:hi]
end
