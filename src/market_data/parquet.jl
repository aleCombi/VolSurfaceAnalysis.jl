# `market_data` module: parquet specs and readers for the Polygon tree.
#
# Storage layout (one tree per kind, the collector's Hive layout):
#   <root>/date=YYYY-MM-DD/symbol=<TICKER>/data.parquet
# `ParquetOptionBars(root)` serves OptionBar from an options tree,
# `ParquetSpots(root)` serves SpotPrice from a spots tree; `root` is the
# kind-specific directory. The selector is a query argument, so one spec
# serves every symbol= partition under its root.
#
# Partition convention: partition D may hold any timestamp in
# [D 00:00, D+1 02:00) UTC -- the collector writes a US session into the
# partition of its local date, and after-midnight UTC rows spill past
# Date(ts). Every shape therefore consults partitions Date(ts)-1 and
# Date(ts) (bounded by the partition list), which is what makes
# `at == collect(between(ts, ts))` an identity rather than a coincidence.
#
# The convention is TIME-ORDERED: every row in partition D-1 precedes
# every row in partition D. That is what a local-date collector produces
# -- one contiguous session per partition, the after-midnight spill
# belonging to the earlier session -- and the four shapes only agree with
# each other under it. `asof` returns at the newest candidate partition
# holding a row <= ts while `at` and `timestamps` merge both candidates,
# so an interleaved layout would let `asof` disagree with
# `at(last(timestamps(...)))`; and the lazy `PartitionBars` iterator
# concatenates D-1 then D without a cross-partition sort, so it would
# yield out-of-order records and make `by_timestamp` throw. Under the
# ordering both disagreements vanish by construction. A feed that
# genuinely interleaves partitions needs a maximum over both candidates
# in `asof` and a lazy two-way merge in `between`; no collector writes
# one today.
#
# Vendor rows carry the bar-open timestamp; it is kept as the visibility
# time (documented one-minute allowance, see data.md / market_data.md).

using DuckDB
using DuckDB: DBInterface
using Tables

"""
    ParquetOptionBars(root)

Spec for `OptionBar` records under an options tree (`.../options_1min`).
Construction is pure: no directory check, so a saved run rehydrates
silently off-machine; `open_data` throws if the root is missing.
"""
struct ParquetOptionBars
    root::String
    ParquetOptionBars(root::AbstractString) = new(String(root))
end

"""
    ParquetSpots(root)

Spec for `SpotPrice` records under a spots tree (`.../spots_1min`).
Same construction rule as `ParquetOptionBars`.
"""
struct ParquetSpots
    root::String
    ParquetSpots(root::AbstractString) = new(String(root))
end

kind(::ParquetOptionBars) = OptionBar
kind(::ParquetSpots) = SpotPrice

# A spec cannot answer the structural question: the partition list is a
# readdir walk over a tree that is not open yet, and `build_market_data`
# holds specs. The readers can, and do.
serves(::ParquetOptionBars, ::Any, ::Type{OptionBar}, ::Any) = missing
serves(::ParquetSpots, ::Any, ::Type{SpotPrice}, ::Any) = missing

served_description(s::ParquetOptionBars) = "ParquetOptionBars under $(s.root)"
served_description(s::ParquetSpots) = "ParquetSpots under $(s.root)"

# --- partitions -----------------------------------------------------------

_partition_path(root::AbstractString, u::Underlying, d::Date) = joinpath(
    root, "date=" * Dates.format(d, "yyyy-mm-dd"), "symbol=" * ticker(u), "data.parquet")

# The sorted partition dates that hold a file for `u`; listed once per
# selector per reader. This list bounds every walk (`asof`, `between`).
function _list_partitions(root::AbstractString, u::Underlying)::Vector{Date}
    out = Date[]
    for name in readdir(root)
        startswith(name, "date=") || continue
        d = tryparse(Date, name[6:end])
        d === nothing && continue
        isfile(_partition_path(root, u, d)) && push!(out, d)
    end
    sort!(out)
end

# Partitions that can hold a timestamp in [from, to] under the convention.
function _candidate_partitions(parts::Vector{Date}, from::DateTime, to::DateTime)
    lo = searchsortedfirst(parts, Date(from) - Day(1))
    hi = searchsortedlast(parts, Date(to))
    view(parts, lo:hi)
end

# Millisecond precision, not whole seconds: `between` is public and its
# bounds are passed through untouched (a TOML datetime with a fractional
# second, a TimeCut cutoff), and truncating the lower bound would admit
# the row at its floor while `at` and `timestamps` compare at full
# precision. DuckDB parses the fractional part; `at`'s exact
# `timestamp = ...` predicate stays exact.
_ts_sql(ts::DateTime) = "TIMESTAMP '" * Dates.format(ts, "yyyy-mm-dd HH:MM:SS.sss") * "'"

function _query_distinct_timestamps(con::DuckDB.DB, path::AbstractString)::Vector{DateTime}
    sql = "SELECT DISTINCT timestamp FROM '$(_sql_path(path))' ORDER BY timestamp"
    out = DateTime[]
    for row in Tables.rows(DBInterface.execute(con, sql))
        push!(out, _coerce_dt(row.timestamp))
    end
    out
end

function _parquet_columns(con::DuckDB.DB, path::AbstractString)::Set{Symbol}
    sql = "SELECT name FROM parquet_schema('$(_sql_path(path))')"
    cols = Set{Symbol}()
    for row in Tables.rows(DBInterface.execute(con, sql))
        push!(cols, Symbol(row.name))
    end
    cols
end

# --- option bars ----------------------------------------------------------

# Per-partition metadata: the distinct timestamps (always filled) plus the
# column-presence flags, filled the first time bars from this partition
# are actually read. One `parquet_schema` query per partition, not per
# timestamp.
mutable struct PartitionMeta
    timestamps::Vector{DateTime}
    cols_loaded::Bool
    has_volume::Bool
    has_open::Bool
    has_high::Bool
    has_low::Bool
    has_parsed::Bool
end

struct ParquetBarsReader
    spec::ParquetOptionBars
    con::DuckDB.DB
    partitions::Dict{Underlying,Vector{Date}}
    days::LRU{Tuple{Underlying,Date},PartitionMeta}
    chains::LRU{Tuple{Underlying,DateTime},Vector{OptionBar}}
    contracts::Dict{String,ContractMeta}
    closed::Base.RefValue{Bool}
end

# DuckDB segfaults on a query against a closed database handle, so use
# after close cannot be left to the storage: every shape checks the flag.
_assert_open(r) = r.closed[] && throw(ArgumentError("$(typeof(r).name.name) is closed"))

kind(::ParquetBarsReader) = OptionBar

"""
    open_data(s::ParquetOptionBars; max_days=200, max_chains=10)

Open a DuckDB connection over the tree. `max_days` bounds the
per-partition metadata cache, `max_chains` the exact-instant chain
cache (`at` only; range reads are never cached). Throws `ArgumentError`
if the root is not a directory.
"""
function open_data(s::ParquetOptionBars; max_days::Int=200, max_chains::Int=10)
    isdir(s.root) || throw(ArgumentError("ParquetOptionBars: root is not a directory: $(s.root)"))
    ParquetBarsReader(s, DuckDB.DB(":memory:"), Dict{Underlying,Vector{Date}}(),
                      LRU{Tuple{Underlying,Date},PartitionMeta}(max_days),
                      LRU{Tuple{Underlying,DateTime},Vector{OptionBar}}(max_chains),
                      Dict{String,ContractMeta}(), Ref(false))
end

function close_data!(r::ParquetBarsReader)
    r.closed[] && return nothing
    r.closed[] = true
    DBInterface.close!(r.con)
    nothing
end

_partitions(r::ParquetBarsReader, u::Underlying) =
    get!(() -> _list_partitions(r.spec.root, u), r.partitions, u)

# An empty date= list means the tree holds nothing for this symbol at any
# instant. The list is already cached per selector, so this costs one
# readdir per underlying per reader.
serves(r::ParquetBarsReader, ::Any, ::Type{OptionBar}, u::Underlying) =
    !isempty(_partitions(r, u))
served_description(r::ParquetBarsReader) =
    "ParquetOptionBars under $(r.spec.root), with no date= partition holding a file for it"

# Only called for dates in the partition list, so no per-day isfile.
_meta(r::ParquetBarsReader, u::Underlying, d::Date)::PartitionMeta =
    get!(r.days, (u, d)) do
        PartitionMeta(_query_distinct_timestamps(r.con, _partition_path(r.spec.root, u, d)),
                      false, false, false, false, false, false)
    end

function _load_cols!(r::ParquetBarsReader, m::PartitionMeta, path::AbstractString)
    m.cols_loaded && return m
    cols = _parquet_columns(r.con, path)
    m.has_volume = :volume in cols
    m.has_open   = :open in cols
    m.has_high   = :high in cols
    m.has_low    = :low in cols
    m.has_parsed = :parsed_expiry in cols && :parsed_strike in cols &&
                   :parsed_option_type in cols && :parsed_underlying in cols
    m.cols_loaded = true
    m
end

# Rows of one partition matching `where_sql`, as OptionBar. Columnar
# materialization via Tables.columntable, then a typed index loop. The
# contract-meta dict is shared across partitions (a few thousand entries
# per symbol). A ticker whose underlying is not `u` throws: under symbol=
# partitioning that is a corrupt store, not a row to skip.
function _query_bars(r::ParquetBarsReader, u::Underlying, d::Date, m::PartitionMeta,
                     where_sql::AbstractString)::Vector{OptionBar}
    path = _partition_path(r.spec.root, u, d)
    _load_cols!(r, m, path)

    base = "ticker, close, timestamp"
    base = m.has_volume ? base * ", volume" : base
    base = m.has_open   ? base * ", open"   : base
    base = m.has_high   ? base * ", high"   : base
    base = m.has_low    ? base * ", low"    : base
    select_list = m.has_parsed ?
        base * ", parsed_underlying, parsed_expiry, parsed_strike, parsed_option_type" :
        base
    sql = "SELECT $select_list FROM '$(_sql_path(path))' WHERE $where_sql"
    ct = Tables.columntable(DBInterface.execute(r.con, sql))

    tickers = ct.ticker
    closes  = ct.close
    tstamps = ct.timestamp
    volumes = m.has_volume ? ct.volume : nothing
    opens   = m.has_open   ? ct.open   : nothing
    highs   = m.has_high   ? ct.high   : nothing
    lows    = m.has_low    ? ct.low    : nothing
    p_und   = m.has_parsed ? ct.parsed_underlying  : nothing
    p_exp   = m.has_parsed ? ct.parsed_expiry      : nothing
    p_strk  = m.has_parsed ? ct.parsed_strike      : nothing
    p_otype = m.has_parsed ? ct.parsed_option_type : nothing

    n = length(tickers)
    out = Vector{OptionBar}(undef, n)
    expected = ticker(u)
    @inbounds for i in 1:n
        tk = String(tickers[i])
        meta = get(r.contracts, tk, nothing)
        if meta === nothing
            u_str, cm = if m.has_parsed
                String(p_und[i]),
                _contract_meta_from_parsed(p_exp[i], Float64(p_strk[i]), String(p_otype[i]))
            else
                pu, expiry, otype, strike = parse_polygon_ticker(tk)
                pu, (expiry=expiry, strike=strike, option_type=otype)
            end
            u_str == expected || throw(ArgumentError(
                "ticker $tk (underlying $u_str) does not match partition underlying $expected"))
            meta = cm
            r.contracts[tk] = meta
        end
        cv = closes[i]
        close_val = cv === missing ? missing : Float64(cv)
        open_val  = opens   === nothing ? missing : (opens[i]   === missing ? missing : Float64(opens[i]))
        high_val  = highs   === nothing ? missing : (highs[i]   === missing ? missing : Float64(highs[i]))
        low_val   = lows    === nothing ? missing : (lows[i]    === missing ? missing : Float64(lows[i]))
        vol       = volumes === nothing ? missing : (volumes[i] === missing ? missing : Float64(volumes[i]))
        out[i] = OptionBar(tk, u, meta.expiry, meta.strike, meta.option_type,
                           open_val, high_val, low_val, close_val, vol, _coerce_dt(tstamps[i]))
    end
    out
end

# Exact instant, through the chain cache. Consults both candidate
# partitions; the cached timestamp list says which one (usually one) has it.
function at(r::ParquetBarsReader, ::Any, ::Type{OptionBar}, u::Underlying, ts::DateTime)
    _assert_open(r)
    get!(r.chains, (u, ts)) do
        out = OptionBar[]
        for d in _candidate_partitions(_partitions(r, u), ts, ts)
            m = _meta(r, u, d)
            insorted(ts, m.timestamps) || continue
            append!(out, _query_bars(r, u, d, m, "timestamp = " * _ts_sql(ts)))
        end
        out
    end
end

# One partition's rows in [from, to], sorted, as a fresh vector: never
# cached, so a range read holds one partition in memory at a time.
function _day_bars(r::ParquetBarsReader, u::Underlying, d::Date, from::DateTime, to::DateTime)
    _assert_open(r)                        # the lazy iterator may outlive the reader
    m = _meta(r, u, d)
    isempty(m.timestamps) && return OptionBar[]
    (from <= last(m.timestamps) && to >= first(m.timestamps)) || return OptionBar[]
    _query_bars(r, u, d, m,
        "timestamp BETWEEN " * _ts_sql(from) * " AND " * _ts_sql(to) * " ORDER BY timestamp")
end

# The lazy range iterator: walks candidate partitions in order, loading
# one at a time. Valid only while the reader is open.
struct PartitionBars
    r::ParquetBarsReader
    u::Underlying
    days::Vector{Date}
    from::DateTime
    to::DateTime
end

Base.IteratorSize(::Type{PartitionBars}) = Base.SizeUnknown()
Base.eltype(::Type{PartitionBars}) = OptionBar

function Base.iterate(it::PartitionBars, state=(0, OptionBar[], 1))
    di, buf, i = state
    while i > length(buf)
        di += 1
        di > length(it.days) && return nothing
        buf = _day_bars(it.r, it.u, it.days[di], it.from, it.to)
        i = 1
    end
    (buf[i], (di, buf, i + 1))
end

function between(r::ParquetBarsReader, ::Any, ::Type{OptionBar}, u::Underlying,
                 from::DateTime, to::DateTime)
    _assert_open(r)
    from <= to || return PartitionBars(r, u, Date[], from, to)
    PartitionBars(r, u, collect(_candidate_partitions(_partitions(r, u), from, to)), from, to)
end

# Walk the partition list backward from Date(ts) until a partition has a
# timestamp <= ts (one cached DISTINCT query per visited partition), then
# read that instant. Bounded by the partitions that exist.
function asof(r::ParquetBarsReader, ctx, ::Type{OptionBar}, u::Underlying, ts::DateTime)
    _assert_open(r)
    parts = _partitions(r, u)
    for j in searchsortedlast(parts, Date(ts)):-1:1
        t = _meta(r, u, parts[j]).timestamps
        k = searchsortedlast(t, ts)
        k == 0 && continue
        return at(r, ctx, OptionBar, u, t[k])
    end
    OptionBar[]
end

function timestamps(r::ParquetBarsReader, ::Any, ::Type{OptionBar}, u::Underlying,
                    from::DateTime, to::DateTime)
    _assert_open(r)
    out = DateTime[]
    from <= to || return out
    for d in _candidate_partitions(_partitions(r, u), from, to)
        t = _meta(r, u, d).timestamps
        append!(out, view(t, searchsortedfirst(t, from):searchsortedlast(t, to)))
    end
    issorted(out) || sort!(out)
    out
end

# --- spots ----------------------------------------------------------------

struct SpotBlock
    timestamps::Vector{DateTime}
    prices::Vector{Float64}
end

struct ParquetSpotsReader
    spec::ParquetSpots
    con::DuckDB.DB
    partitions::Dict{Underlying,Vector{Date}}
    blocks::LRU{Tuple{Underlying,Date},SpotBlock}
    closed::Base.RefValue{Bool}
end

kind(::ParquetSpotsReader) = SpotPrice

"""
    open_data(s::ParquetSpots; max_days=200)

Open a DuckDB connection over the spots tree; `max_days` bounds the
per-partition block cache. Throws `ArgumentError` if the root is not a
directory.
"""
function open_data(s::ParquetSpots; max_days::Int=200)
    isdir(s.root) || throw(ArgumentError("ParquetSpots: root is not a directory: $(s.root)"))
    ParquetSpotsReader(s, DuckDB.DB(":memory:"), Dict{Underlying,Vector{Date}}(),
                       LRU{Tuple{Underlying,Date},SpotBlock}(max_days), Ref(false))
end

function close_data!(r::ParquetSpotsReader)
    r.closed[] && return nothing
    r.closed[] = true
    DBInterface.close!(r.con)
    nothing
end

_partitions(r::ParquetSpotsReader, u::Underlying) =
    get!(() -> _list_partitions(r.spec.root, u), r.partitions, u)

serves(r::ParquetSpotsReader, ::Any, ::Type{SpotPrice}, u::Underlying) =
    !isempty(_partitions(r, u))
served_description(r::ParquetSpotsReader) =
    "ParquetSpots under $(r.spec.root), with no date= partition holding a file for it"

# Rows without a close are dropped: a spot without a price is no observation.
function _load_spot_block(con::DuckDB.DB, path::AbstractString)::SpotBlock
    sql = "SELECT timestamp, close FROM '$(_sql_path(path))' ORDER BY timestamp"
    ts_buf = DateTime[]
    px_buf = Float64[]
    for row in Tables.rows(DBInterface.execute(con, sql))
        row.close === missing && continue
        push!(ts_buf, _coerce_dt(row.timestamp))
        push!(px_buf, Float64(row.close))
    end
    SpotBlock(ts_buf, px_buf)
end

_block(r::ParquetSpotsReader, u::Underlying, d::Date)::SpotBlock =
    get!(() -> _load_spot_block(r.con, _partition_path(r.spec.root, u, d)), r.blocks, (u, d))

function _append_spots!(out::Vector{SpotPrice}, u::Underlying, b::SpotBlock, from::DateTime, to::DateTime)
    lo = searchsortedfirst(b.timestamps, from)
    hi = searchsortedlast(b.timestamps, to)
    for i in lo:hi
        push!(out, SpotPrice(u, b.prices[i], b.timestamps[i]))
    end
    out
end

# Spots are a snapshot kind, read through `only_or_missing`, so two rows
# at one instant abort the read. Two ways to get there without anyone
# writing bad code: a vendor re-delivers a minute into one partition, or
# the same after-midnight row lands in both a partition's spill and the
# next partition's body. Equal price collapses -- there is no information
# to lose; a disagreement throws, because taking the first is a silent
# choice between two answers.
function _collapse_duplicates!(out::Vector{SpotPrice}, u::Underlying)
    isempty(out) && return out
    w = 1
    for i in 2:length(out)
        prev, cur = out[w], out[i]
        if cur.timestamp == prev.timestamp
            cur.price == prev.price && continue
            throw(ConflictingRecords(SpotPrice, u, cur.timestamp, prev.price, cur.price))
        end
        w += 1
        out[w] = cur
    end
    resize!(out, w)
end

function between(r::ParquetSpotsReader, ::Any, ::Type{SpotPrice}, u::Underlying,
                 from::DateTime, to::DateTime)
    _assert_open(r)
    out = SpotPrice[]
    from <= to || return out
    for d in _candidate_partitions(_partitions(r, u), from, to)
        _append_spots!(out, u, _block(r, u, d), from, to)
    end
    issorted(out; by = s -> s.timestamp) || sort!(out; by = s -> s.timestamp)
    _collapse_duplicates!(out, u)
end

function asof(r::ParquetSpotsReader, ::Any, ::Type{SpotPrice}, u::Underlying, ts::DateTime)
    _assert_open(r)
    parts = _partitions(r, u)
    for j in searchsortedlast(parts, Date(ts)):-1:1
        b = _block(r, u, parts[j])
        k = searchsortedlast(b.timestamps, ts)
        k == 0 && continue
        win = b.timestamps[k]
        return _append_spots!(SpotPrice[], u, b, win, win)
    end
    SpotPrice[]
end

function timestamps(r::ParquetSpotsReader, ::Any, ::Type{SpotPrice}, u::Underlying,
                    from::DateTime, to::DateTime)
    _assert_open(r)
    out = DateTime[]
    from <= to || return out
    for d in _candidate_partitions(_partitions(r, u), from, to)
        t = _block(r, u, d).timestamps
        append!(out, view(t, searchsortedfirst(t, from):searchsortedlast(t, to)))
    end
    issorted(out) || sort!(out)
    unique!(out)                    # the grid is distinct instants, as InMemory's is
end
