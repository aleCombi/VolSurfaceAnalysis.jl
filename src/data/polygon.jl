using TimeZones

const TZ_ET = tz"America/New_York"

function et_to_utc(date::Date, t::Time)::DateTime
    local_dt = DateTime(date) + Hour(Dates.hour(t)) + Minute(Dates.minute(t))
    DateTime(ZonedDateTime(local_dt, TZ_ET), UTC)
end

et_to_utc(dt::DateTime)::DateTime = DateTime(ZonedDateTime(dt, TZ_ET), UTC)

const _POLYGON_TICKER_RE = r"^O:([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$"

function parse_polygon_ticker(ticker::AbstractString)::Tuple{String,DateTime,OptionType,Float64}
    m = match(_POLYGON_TICKER_RE, ticker)
    m === nothing && throw(ArgumentError("invalid Polygon ticker: $ticker"))
    underlying = m[1]
    year = 2000 + parse(Int, m[2])
    month = parse(Int, m[3])
    day = parse(Int, m[4])
    expiry = et_to_utc(Date(year, month, day), Time(16, 0))
    otype = m[5] == "C" ? Call : Put
    strike = parse(Int, m[6]) / 1000.0
    return (underlying, expiry, otype, strike)
end

_sql_path(p::AbstractString) = replace(String(p), "\\" => "/")

_coerce_dt(x::DateTime) = x
_coerce_dt(x) = DateTime(x)

# Contract identity as the vendor row carries it: the collector's parsed_*
# columns when present, else the ticker. Storage-agnostic; the market_data
# parquet reader builds records from it.
const ContractMeta = NamedTuple{(:expiry, :strike, :option_type),Tuple{DateTime,Float64,OptionType}}

function _contract_meta_from_parsed(parsed_expiry, parsed_strike::Float64,
                                    parsed_option_type::AbstractString)::ContractMeta
    expiry_date = Date(_coerce_dt(parsed_expiry))
    expiry = et_to_utc(expiry_date, Time(16, 0))
    otype = parsed_option_type == "C" ? Call : Put
    return (expiry=expiry, strike=parsed_strike, option_type=otype)
end
