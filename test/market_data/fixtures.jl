# Shared fixtures for the market_data suites. Constants are prefixed `_MD_`
# because runtests.jl includes every suite into one module.

const _MD_SPY = Underlying("SPY")
const _MD_SPX = Underlying("SPX")
const _MD_USD = Currency("USD")

const _MD_T1 = DateTime(2024, 1, 15, 19, 30)
const _MD_T2 = DateTime(2024, 1, 15, 19, 31)
const _MD_T3 = DateTime(2024, 1, 16, 19, 30)
const _MD_EXPIRY = DateTime(2024, 2, 16, 21, 0)

# One option bar; `underlying` and `timestamp` are the selector and the
# visibility time the protocol keys on.
function _md_bar(u::Underlying, ts::DateTime, strike::Float64;
                 otype::OptionType=Call, open=1.00, high=1.20, low=0.80, close=1.00, volume=10.0)
    OptionBar("O:$(ticker(u))240216$(otype == Call ? "C" : "P")$(lpad(round(Int, strike * 1000), 8, '0'))",
              u, _MD_EXPIRY, strike, otype, open, high, low, close, volume, ts)
end

_md_spot(u::Underlying, ts::DateTime, price::Float64) = SpotPrice(u, price, ts)

# Bars for SPY and SPX at the three timestamps, two strikes each, SPX rows
# interleaved with SPY rows in input order so selector filtering is tested.
function _md_bars()
    out = OptionBar[]
    for ts in (_MD_T1, _MD_T2, _MD_T3), u in (_MD_SPY, _MD_SPX)
        base = u === _MD_SPY ? 480.0 : 4800.0
        push!(out, _md_bar(u, ts, base - 10.0; otype=Put))
        push!(out, _md_bar(u, ts, base + 10.0; otype=Call))
    end
    out
end

function _md_spots()
    out = SpotPrice[]
    for (i, ts) in enumerate((_MD_T1, _MD_T2, _MD_T3))
        push!(out, _md_spot(_MD_SPY, ts, 480.0 + i))
        push!(out, _md_spot(_MD_SPX, ts, 4800.0 + i))
    end
    out
end

# ---------- parquet fixture writers ----------
# Ported from test/data/test_parquet_source.jl; the old suite keeps its own
# copies until the old layer is deleted at step 3.

using DuckDB
using DuckDB: DBInterface

function _md_write_options_parquet(path::AbstractString, rows::Vector{<:NamedTuple};
                                   include_volume::Bool=true, include_ohlc::Bool=true)
    mkpath(dirname(path))
    db = DuckDB.DB(":memory:")
    schema_parts = ["ticker VARCHAR", "close DOUBLE"]
    include_volume && push!(schema_parts, "volume DOUBLE")
    if include_ohlc
        push!(schema_parts, "open DOUBLE")
        push!(schema_parts, "high DOUBLE")
        push!(schema_parts, "low DOUBLE")
    end
    push!(schema_parts, "timestamp TIMESTAMP")
    DBInterface.execute(db, "CREATE TABLE bars (" * join(schema_parts, ", ") * ")")
    _val(r, k) = haskey(r, k) ? string(getfield(r, k)) : "NULL"
    for r in rows
        ts_str = Dates.format(r.timestamp, "yyyy-mm-dd HH:MM:SS")
        vals = ["'$(r.ticker)'", string(r.close)]
        include_volume && push!(vals, _val(r, :volume))
        if include_ohlc
            push!(vals, _val(r, :open))
            push!(vals, _val(r, :high))
            push!(vals, _val(r, :low))
        end
        push!(vals, "'$ts_str'")
        DBInterface.execute(db, "INSERT INTO bars VALUES (" * join(vals, ", ") * ")")
    end
    DBInterface.execute(db, "COPY bars TO '$(replace(path, "\\" => "/"))' (FORMAT PARQUET)")
    DBInterface.close!(db)
end

function _md_write_spot_parquet(path::AbstractString, ts::Vector{DateTime}, prices::Vector{Float64})
    mkpath(dirname(path))
    db = DuckDB.DB(":memory:")
    DBInterface.execute(db, "CREATE TABLE bars (timestamp TIMESTAMP, close DOUBLE)")
    for (t, p) in zip(ts, prices)
        DBInterface.execute(db, "INSERT INTO bars VALUES ('$(Dates.format(t, "yyyy-mm-dd HH:MM:SS"))', $p)")
    end
    DBInterface.execute(db, "COPY bars TO '$(replace(path, "\\" => "/"))' (FORMAT PARQUET)")
    DBInterface.close!(db)
end

# Two option days (three chain timestamps) and two spot days, plus one spot
# row at 2024-01-16T00:30 written into the date=2024-01-15 partition: the
# after-midnight spill the real spots tree has.
function _md_build_parquet_fixture(root::AbstractString)
    opts = joinpath(root, "options_1min")
    spot = joinpath(root, "spots_1min")
    d1 = Date(2024, 1, 15)
    d2 = Date(2024, 1, 16)
    t1a = DateTime(d1, Time(15, 30))
    t1b = DateTime(d1, Time(15, 31))
    t2a = DateTime(d2, Time(15, 30))
    spill = DateTime(d2, Time(0, 30))

    _md_write_options_parquet(
        joinpath(opts, "date=2024-01-15", "symbol=SPY", "data.parquet"),
        [
            (ticker="O:SPY240129C00406000", close=1.05, volume=12.0,
             open=1.00, high=1.10, low=0.95, timestamp=t1a),
            (ticker="O:SPY240129P00400000", close=2.10, volume=5.0,
             open=2.05, high=2.20, low=2.00, timestamp=t1a),
            (ticker="O:SPY240129C00406000", close=1.07, volume=20.0,
             open=1.05, high=1.12, low=1.04, timestamp=t1b),
        ])
    _md_write_options_parquet(
        joinpath(opts, "date=2024-01-16", "symbol=SPY", "data.parquet"),
        [(ticker="O:SPY240129C00406000", close=1.20, volume=30.0,
          open=1.18, high=1.25, low=1.15, timestamp=t2a)])

    _md_write_spot_parquet(
        joinpath(spot, "date=2024-01-15", "symbol=SPY", "data.parquet"),
        [t1a, t1b, spill], [480.0, 480.5, 480.7])
    _md_write_spot_parquet(
        joinpath(spot, "date=2024-01-16", "symbol=SPY", "data.parquet"),
        [t2a], [481.0])

    (opts_root=opts, spot_root=spot, t1a=t1a, t1b=t1b, t2a=t2a, spill=spill, d1=d1, d2=d2)
end
