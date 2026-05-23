using VolSurfaceAnalysis
using VolSurfaceAnalysis: option_path, spot_path, _load_chain_day, _load_spot_day,
    _ensure_chain_day!, _ensure_spot_day!, ContractMeta, SpotDay
using DuckDB
using DuckDB: DBInterface

# ---------- fixture helpers ----------

function _write_options_parquet(path::AbstractString, rows::Vector{<:NamedTuple};
                                include_volume::Bool=true,
                                include_ohlc::Bool=true)
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
    p = replace(path, "\\" => "/")
    DBInterface.execute(db, "COPY bars TO '$p' (FORMAT PARQUET)")
    DBInterface.close!(db)
end

function _write_spot_parquet(path::AbstractString, ts::Vector{DateTime}, prices::Vector{Float64})
    mkpath(dirname(path))
    db = DuckDB.DB(":memory:")
    DBInterface.execute(db, "CREATE TABLE bars (timestamp TIMESTAMP, close DOUBLE)")
    for (t, p) in zip(ts, prices)
        DBInterface.execute(db, "INSERT INTO bars VALUES ('$(Dates.format(t, "yyyy-mm-dd HH:MM:SS"))', $p)")
    end
    pth = replace(path, "\\" => "/")
    DBInterface.execute(db, "COPY bars TO '$pth' (FORMAT PARQUET)")
    DBInterface.close!(db)
end

function _build_fixture(root::AbstractString)
    opts = joinpath(root, "options_1min")
    spot = joinpath(root, "spots_1min")

    d1 = Date(2024, 1, 15)
    d2 = Date(2024, 1, 16)

    t1a = DateTime(d1, Time(15, 30))
    t1b = DateTime(d1, Time(15, 31))
    t2a = DateTime(d2, Time(15, 30))

    _write_options_parquet(
        joinpath(opts, "date=2024-01-15", "symbol=SPY", "data.parquet"),
        [
            (ticker="O:SPY240129C00406000", close=1.05, volume=12.0,
             open=1.00, high=1.10, low=0.95, timestamp=t1a),
            (ticker="O:SPY240129P00400000", close=2.10, volume=5.0,
             open=2.05, high=2.20, low=2.00, timestamp=t1a),
            (ticker="O:SPY240129C00406000", close=1.07, volume=20.0,
             open=1.05, high=1.12, low=1.04, timestamp=t1b),
        ],
    )
    _write_options_parquet(
        joinpath(opts, "date=2024-01-16", "symbol=SPY", "data.parquet"),
        [(ticker="O:SPY240129C00406000", close=1.20, volume=30.0,
          open=1.18, high=1.25, low=1.15, timestamp=t2a)],
    )

    _write_spot_parquet(
        joinpath(spot, "date=2024-01-15", "symbol=SPY", "data.parquet"),
        [t1a, t1b], [480.0, 480.5],
    )
    _write_spot_parquet(
        joinpath(spot, "date=2024-01-16", "symbol=SPY", "data.parquet"),
        [t2a], [481.0],
    )

    return (opts_root=opts, spot_root=spot, t1a=t1a, t1b=t1b, t2a=t2a, d1=d1, d2=d2)
end

# ---------- tests ----------

const _SYNTH = SpreadFromOHLCV(0.7)

mktempdir() do root
    fx = _build_fixture(root)
    ds = ParquetDataSource("SPY"; options_root=fx.opts_root, spot_root=fx.spot_root,
                           synthesizer=_SYNTH, max_days_cached=2)

    @testset "Path layout: options_1min and spots_1min, both keyed by symbol=" begin
        op = option_path(ds, fx.d1)
        sp = spot_path(ds, fx.d1)
        @test occursin("symbol=SPY", op)
        @test occursin("date=2024-01-15", op)
        @test endswith(op, "data.parquet")
        @test occursin("symbol=SPY", sp)
        @test !occursin("underlying=", op)
    end

    @testset "get_chain: hit, miss timestamp, miss day" begin
        chain = get_chain(ds, fx.t1a)
        @test chain isa Vector{OptionQuote}
        @test length(chain) == 2
        @test all(q -> ticker(q.underlying) == "SPY", chain)
        @test any(q -> q.option_type == Call && q.strike == 406.0, chain)
        @test any(q -> q.option_type == Put && q.strike == 400.0, chain)
        @test all(q -> !ismissing(q.bid) && !ismissing(q.ask), chain)
        @test all(q -> ismissing(q.iv) && ismissing(q.open_interest), chain)
        @test get_chain(ds, fx.t1a) === chain || length(get_chain(ds, fx.t1a)) == 2

        @test get_chain(ds, DateTime(fx.d1, Time(16, 0))) === nothing
        @test get_chain(ds, DateTime(2024, 1, 17, 15, 30)) === nothing
    end

    @testset "Mark mapping + volume" begin
        chain = get_chain(ds, fx.t1a)
        c = first(filter(q -> q.option_type == Call, chain))
        @test c.mark == 1.05
        @test c.volume == 12.0
    end

    @testset "Synthesized bid/ask: SpreadFromOHLCV(0.7) applied at load" begin
        chain = get_chain(ds, fx.t1a)
        c = first(filter(q -> q.option_type == Call, chain))
        # high=1.10, low=0.95, close=1.05, λ=0.7
        # bid = 0.95 + 0.7*(1.05-0.95) = 1.02
        # ask = 1.10 - 0.7*(1.10-1.05) = 1.065
        @test c.bid ≈ 1.02
        @test c.ask ≈ 1.065
    end

    @testset "open_position: fills against a ParquetDataSource chain" begin
        chain = get_chain(ds, fx.t1a)
        q = first(filter(qq -> qq.option_type == Call, chain))
        trd = Trade(q.underlying, q.strike, q.expiry, q.option_type;
                    direction=+1, quantity=1.0)
        pos = open_position(trd, q, 480.0)
        @test pos.entry_price == q.ask
        @test pos.entry_bid == q.bid
        @test pos.entry_ask == q.ask
    end

    @testset "Contract cache populated" begin
        @test length(ds.contract_cache) == 2
        @test haskey(ds.contract_cache, "O:SPY240129C00406000")
        meta = ds.contract_cache["O:SPY240129C00406000"]
        @test meta.strike == 406.0
        @test meta.option_type == Call
    end

    @testset "get_spot: binary search regression (parallel arrays)" begin
        @test get_spot(ds, fx.t1a) == 480.0
        @test get_spot(ds, fx.t1b) == 480.5
        @test ismissing(get_spot(ds, DateTime(fx.d1, Time(15, 32))))
        @test ismissing(get_spot(ds, DateTime(2024, 1, 17, 15, 30)))
    end

    @testset "get_spots: range across two days" begin
        out = get_spots(ds, fx.t1a, fx.t2a)
        @test length(out) == 3
        @test [s.timestamp for s in out] == [fx.t1a, fx.t1b, fx.t2a]
        @test [s.price for s in out] == [480.0, 480.5, 481.0]

        @test isempty(get_spots(ds, fx.t2a + Hour(1), fx.t2a + Hour(2)))
        @test isempty(get_spots(ds, fx.t1b, fx.t1a))  # from > to
    end

    @testset "available_timestamps(ds, from, to) reuses chain cache" begin
        clear_cache!(ds)
        ts = available_timestamps(ds, fx.t1a, fx.t2a)
        @test ts == sort([fx.t1a, fx.t1b, fx.t2a])
        @test haskey(ds.chain_cache, fx.d1)
        @test haskey(ds.chain_cache, fx.d2)
    end

    @testset "available_timestamps(ds) unbounded throws" begin
        @test_throws ArgumentError available_timestamps(ds)
    end

    @testset "LRU eviction: max_days_cached=2" begin
        clear_cache!(ds)
        # load d1 then d2; both fit
        get_chain(ds, fx.t1a)
        get_chain(ds, fx.t2a)
        @test collect(keys(ds.chain_cache)) == [fx.d1, fx.d2]

        # touching d1 should bump it; loading a third (missing) day evicts d2
        get_chain(ds, fx.t1a)
        get_chain(ds, DateTime(2024, 1, 17, 15, 30))  # missing day still gets cached
        ks = collect(keys(ds.chain_cache))
        @test length(ks) == 2
        @test fx.d1 in ks
        @test !(fx.d2 in ks)
    end

    @testset "clear_cache! empties chain+spot, keeps contract_cache" begin
        get_chain(ds, fx.t1a)
        get_spot(ds, fx.t1a)
        n_contracts = length(ds.contract_cache)
        clear_cache!(ds)
        @test isempty(ds.chain_cache)
        @test isempty(ds.spot_cache)
        @test length(ds.contract_cache) == n_contracts
    end
    close(ds); GC.gc()
end

# ---------- volume-absent + ticker-mismatch (separate fixture) ----------

mktempdir() do root
    opts = joinpath(root, "options_1min")
    spot = joinpath(root, "spots_1min")
    mkpath(joinpath(spot, "date=2024-01-15", "symbol=SPY"))
    d = Date(2024, 1, 15)
    t = DateTime(d, Time(15, 30))

    _write_options_parquet(
        joinpath(opts, "date=2024-01-15", "symbol=SPY", "data.parquet"),
        [(ticker="O:SPY240129C00406000", close=1.05, timestamp=t)];
        include_volume=false, include_ohlc=false,
    )
    # spot file stub so isdir passes
    _write_spot_parquet(
        joinpath(spot, "date=2024-01-15", "symbol=SPY", "data.parquet"),
        [t], [480.0],
    )

    ds = ParquetDataSource("SPY"; options_root=opts, spot_root=spot, synthesizer=_SYNTH)

    @testset "Volume column absent -> missing" begin
        chain = get_chain(ds, t)
        @test chain !== nothing
        @test ismissing(first(chain).volume)
        @test first(chain).mark == 1.05
    end

    @testset "OHLC columns absent -> synthesized bid/ask missing, mark still close" begin
        chain = get_chain(ds, t)
        q = first(chain)
        @test ismissing(q.bid)
        @test ismissing(q.ask)
        @test q.mark == 1.05
    end
    close(ds); GC.gc()
end

mktempdir() do root
    opts = joinpath(root, "options_1min")
    spot = joinpath(root, "spots_1min")
    d = Date(2024, 1, 15)
    t = DateTime(d, Time(15, 30))

    # a SPY-partitioned file containing a QQQ ticker (corrupt store)
    _write_options_parquet(
        joinpath(opts, "date=2024-01-15", "symbol=SPY", "data.parquet"),
        [(ticker="O:QQQ240129C00406000", close=1.05, volume=1.0, timestamp=t)],
    )
    _write_spot_parquet(
        joinpath(spot, "date=2024-01-15", "symbol=SPY", "data.parquet"),
        [t], [480.0],
    )

    ds = ParquetDataSource("SPY"; options_root=opts, spot_root=spot, synthesizer=_SYNTH)

    @testset "Ticker-underlying mismatch throws" begin
        @test_throws ArgumentError get_chain(ds, t)
    end
    close(ds); GC.gc()
end

# ---------- interface equivalence vs InMemoryDataSource ----------

mktempdir() do root
    fx = _build_fixture(root)
    parquet_ds = ParquetDataSource("SPY"; options_root=fx.opts_root, spot_root=fx.spot_root,
                                   synthesizer=_SYNTH)

    # build in-memory mirror by reading via the parquet ds itself
    chains = Dict{DateTime,Vector{OptionQuote}}()
    spots = Dict{DateTime,Float64}()
    for ts in (fx.t1a, fx.t1b, fx.t2a)
        chains[ts] = get_chain(parquet_ds, ts)
        spots[ts] = get_spot(parquet_ds, ts)
    end
    inmem_ds = InMemoryDataSource("SPY"; chains=chains, spots=spots)

    @testset "Interface equivalence: ParquetDataSource vs InMemoryDataSource" begin
        for ts in (fx.t1a, fx.t1b, fx.t2a)
            pq = get_chain(parquet_ds, ts)
            mem = get_chain(inmem_ds, ts)
            @test length(pq) == length(mem)
            @test get_spot(parquet_ds, ts) == get_spot(inmem_ds, ts)
        end
        @test available_timestamps(parquet_ds, fx.t1a, fx.t2a) ==
              available_timestamps(inmem_ds, fx.t1a, fx.t2a)

        spans_pq = get_spots(parquet_ds, fx.t1a, fx.t2a)
        spans_mem = get_spots(inmem_ds, fx.t1a, fx.t2a)
        @test [s.timestamp for s in spans_pq] == [s.timestamp for s in spans_mem]
        @test [s.price for s in spans_pq] == [s.price for s in spans_mem]
    end
    close(parquet_ds); GC.gc()
end

# ---------- single-root convenience constructor ----------

mktempdir() do root
    fx = _build_fixture(root)

    @testset "Single-root constructor derives options_1min and spots_1min" begin
        ds = ParquetDataSource("SPY", root; synthesizer=_SYNTH)
        @test ds.options_root == joinpath(root, "options_1min")
        @test ds.spot_root == joinpath(root, "spots_1min")
        chain = get_chain(ds, fx.t1a)
        @test chain !== nothing && length(chain) == 2
        @test get_spot(ds, fx.t1a) == 480.0
        close(ds); GC.gc()
    end

    @testset "with_parquet_source closes after callback" begin
        ds_ref = Ref{Any}(nothing)
        n = with_parquet_source("SPY", root; synthesizer=_SYNTH) do ds
            ds_ref[] = ds
            length(get_chain(ds, fx.t1a))
        end
        ds = ds_ref[]::ParquetDataSource
        @test n == 2
        @test !isopen(ds)
        @test isempty(ds.chain_cache)
        @test isempty(ds.spot_cache)
        @test_throws ArgumentError get_chain(ds, fx.t1a)
        ds_ref[] = nothing
        ds = nothing
        GC.gc()
    end

    @testset "Missing root: warns at construction, throws on first read" begin
        bad = joinpath(root, "nonexistent")
        ds = @test_logs (:warn,) (:warn,) ParquetDataSource("SPY";
            options_root=bad, spot_root=bad, synthesizer=_SYNTH)
        @test_throws ArgumentError get_chain(ds, DateTime(2024, 1, 15, 15, 30))
        @test_throws ArgumentError get_spot(ds, DateTime(2024, 1, 15, 15, 30))
        close(ds); GC.gc()
    end
end

# ---------- parsed_* columns preferred when present ----------

mktempdir() do root
    opts = joinpath(root, "options_1min")
    spot = joinpath(root, "spots_1min")
    d = Date(2024, 1, 15)
    t = DateTime(d, Time(15, 30))

    db = DuckDB.DB(":memory:")
    DBInterface.execute(db, """
        CREATE TABLE bars (
            ticker VARCHAR, close DOUBLE, volume DOUBLE, timestamp TIMESTAMP,
            parsed_underlying VARCHAR, parsed_expiry TIMESTAMP,
            parsed_strike DOUBLE, parsed_option_type VARCHAR
        )
    """)
    ts_str = Dates.format(t, "yyyy-mm-dd HH:MM:SS")
    # ticker says C 406, parsed columns say P 999 — loader must trust parsed_*
    DBInterface.execute(db, """
        INSERT INTO bars VALUES (
            'O:SPY240129C00406000', 1.05, 12.0, '$ts_str',
            'SPY', '2024-01-29 00:00:00', 999.0, 'P'
        )
    """)
    p = joinpath(opts, "date=2024-01-15", "symbol=SPY", "data.parquet")
    mkpath(dirname(p))
    DBInterface.execute(db, "COPY bars TO '$(replace(p, "\\" => "/"))' (FORMAT PARQUET)")
    DBInterface.close!(db)

    _write_spot_parquet(joinpath(spot, "date=2024-01-15", "symbol=SPY", "data.parquet"),
                        [t], [480.0])

    ds = ParquetDataSource("SPY"; options_root=opts, spot_root=spot, synthesizer=_SYNTH)

    @testset "parsed_* columns are authoritative over ticker text" begin
        chain = get_chain(ds, t)
        @test length(chain) == 1
        q = first(chain)
        @test q.option_type == Put
        @test q.strike == 999.0
        @test q.expiry == DateTime(2024, 1, 29, 21, 0)  # 4PM ET (EST) -> 21:00 UTC
    end
    close(ds); GC.gc()
end

# ---------- opt-in real-data smoke ----------

if haskey(ENV, "VSA_POLYGON_ROOT")
    @testset "Real-data smoke (VSA_POLYGON_ROOT)" begin
        ds = ParquetDataSource("AAPL", ENV["VSA_POLYGON_ROOT"]; synthesizer=_SYNTH)
        @info "smoke: probing one date for AAPL"
        d = Date(2016, 3, 28)
        ts = available_timestamps(ds, DateTime(d), DateTime(d, Time(23, 59)))
        @test !isempty(ts)
        @test get_chain(ds, first(ts)) !== nothing
        close(ds); GC.gc()
    end
else
    @info "skipping real-data smoke (set VSA_POLYGON_ROOT to enable)"
end
