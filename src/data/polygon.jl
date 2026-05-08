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
