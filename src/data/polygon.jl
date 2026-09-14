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

# --- the bar-stamp convention --------------------------------------------
#
# A vendor minute bar is stamped at its OPEN, but every value read off it
# -- close, high, low, and any spread synthesized from them -- is knowable
# only when the minute has finished. The canonical visibility time of a
# record read off such a bar is therefore the row timestamp plus the bar
# interval: the 19:29 row becomes visible at 19:30, and a decision at
# 19:30 reads the completed 19:29-19:30 minute. Stamping the open instead
# hands every bar-based fill and every settlement price up to one minute
# of future information; `TimeCut` cannot catch that, because the record
# admitted through it claims to be knowable before it is.
#
# Both production trees (`options_1min`, `spots_1min`) hold one-minute
# bars, so `BAR_INTERVAL` is one minute and this is THE convention, fixed
# here in code. It is deliberately not a spec option and not a config key:
# one of the two settings would enable lookahead, and offering both would
# invite an experiment to pick the incorrect clock. A source whose bars
# are not one minute needs its own reader, and that reader states its own
# interval -- the general "declare your visibility convention" rule from
# `market_data.md` -- but a completed minute's availability is not a
# choice an experiment gets to make.
const BAR_INTERVAL = Minute(1)

"""
    bar_visible_at(row_timestamp) -> DateTime

The visibility time of a record read off the minute bar stamped
`row_timestamp` at its open: `row_timestamp + BAR_INTERVAL`, the instant
the bar's close, high and low become knowable. The inverse of
[`bar_row_time`](@ref).
"""
bar_visible_at(row_timestamp::DateTime)::DateTime = row_timestamp + BAR_INTERVAL

"""
    bar_row_time(visible_at) -> DateTime

The vendor row timestamp of the bar that becomes visible at `visible_at`.
The inverse of [`bar_visible_at`](@ref); readers use it to translate a
query bound expressed in visibility time into the stored clock. Shifting
by a whole minute preserves both inclusive endpoints and sub-second
precision, so a bound translated through it selects exactly the rows its
untranslated form would have selected one minute earlier.
"""
bar_row_time(visible_at::DateTime)::DateTime = visible_at - BAR_INTERVAL

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
