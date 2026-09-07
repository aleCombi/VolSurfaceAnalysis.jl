# Parquet specs and readers on the new protocol. Every assertion of the old
# test_parquet_source.jl re-expressed, plus the protocol identities
# (at == collect(between), asof walks, the after-midnight spill) and a
# cross-check against the old ParquetDataSource while it still exists.

const _MD_PQ_SYNTH = SpreadFromOHLCV(0.7)

_md_pq_map(fx) = MarketData(ParquetOptionBars(fx.opts_root), QuotesFromBars(_MD_PQ_SYNTH),
                            ParquetSpots(fx.spot_root))

@testset "parquet specs: pure construction, kind, lifecycle presence" begin
    s = @test_logs ParquetOptionBars("/nonexistent/opts")          # no warning
    @test kind(s) === OptionBar
    @test kind(ParquetSpots("/x")) === SpotPrice
    @test inputs(s) == ()
    @test VolSurfaceAnalysis.has_lifecycle(s)
    @test VolSurfaceAnalysis.has_lifecycle(ParquetSpots("/x"))
    @test_throws ArgumentError open_data(s)
    @test_throws ArgumentError open_data(ParquetSpots("/nonexistent/spots"))
    @test_throws ArgumentError open_data(MarketData(s))
end

mktempdir() do root
    fx = _md_build_parquet_fixture(root)

    @testset "parquet bars: at hit / miss timestamp / miss day, fields" begin
        with_data(_md_pq_map(fx)) do d
            chain = at(d, OptionBar, _MD_SPY, fx.t1a)
            @test chain isa Vector{OptionBar}
            @test length(chain) == 2
            @test all(b -> b.underlying === _MD_SPY, chain)
            @test any(b -> b.option_type == Call && b.strike == 406.0, chain)
            @test any(b -> b.option_type == Put && b.strike == 400.0, chain)
            c = first(filter(b -> b.option_type == Call, chain))
            @test c.close == 1.05 && c.volume == 12.0 && c.open == 1.00 && c.high == 1.10 && c.low == 0.95
            @test c.timestamp == fx.t1a
            @test c.expiry == DateTime(2024, 1, 29, 21, 0)
            @test at(d, OptionBar, _MD_SPY, DateTime(fx.d1, Time(16, 0))) == OptionBar[]
            @test at(d, OptionBar, _MD_SPY, DateTime(2024, 1, 17, 15, 30)) == OptionBar[]
            @test at(d, OptionBar, Underlying("QQQ"), fx.t1a) == OptionBar[]
            r = entry(d, OptionBar)
            @test length(r.contracts) == 2
            @test r.contracts["O:SPY240129C00406000"].strike == 406.0
        end
    end

    @testset "parquet bars: synthesized quotes through QuotesFromBars" begin
        with_data(_md_pq_map(fx)) do d
            quotes = at(d, OptionQuote, _MD_SPY, fx.t1a)
            @test quotes isa Vector{OptionQuote}
            c = first(filter(q -> q.option_type == Call, quotes))
            @test c.bid ≈ 1.02 && c.ask ≈ 1.065 && c.mark == 1.05 && c.volume == 12.0
            @test ismissing(c.iv) && ismissing(c.open_interest)
            trd = Trade(c.underlying, c.strike, c.expiry, c.option_type; direction=+1, quantity=1.0)
            pos = open_position(trd, c, 480.0)
            @test pos.entry_price == c.ask
        end
    end

    @testset "parquet bars: at == collect(between), range across days" begin
        with_data(_md_pq_map(fx)) do d
            for ts in (fx.t1a, fx.t1b, fx.t2a, DateTime(fx.d1, Time(16, 0)), fx.spill)
                @test at(d, OptionBar, _MD_SPY, ts) == collect(between(d, OptionBar, _MD_SPY, ts, ts))
            end
            rng = between(d, OptionBar, _MD_SPY, fx.t1a, fx.t2a)
            @test eltype(rng) === OptionBar
            v = collect(rng)
            @test v isa Vector{OptionBar}
            @test v == vcat(at(d, OptionBar, _MD_SPY, fx.t1a), at(d, OptionBar, _MD_SPY, fx.t1b),
                            at(d, OptionBar, _MD_SPY, fx.t2a))
            @test first.(collect(by_timestamp(rng))) == [fx.t1a, fx.t1b, fx.t2a]
            @test isempty(collect(between(d, OptionBar, _MD_SPY, fx.t1b, fx.t1a)))          # from > to
            @test isempty(collect(between(d, OptionBar, _MD_SPY, fx.t2a + Hour(1), fx.t2a + Hour(2))))
            @test [b.timestamp for b in between(d, OptionBar, _MD_SPY, fx.t1b, fx.t2a - Minute(1))] == [fx.t1b]
            qs = collect(between(d, OptionQuote, _MD_SPY, fx.t1a, fx.t2a))
            @test length(qs) == 4 && qs[1].bid ≈ 1.02
        end
        # range reads never enter the chain cache (fresh reader, no `at` yet)
        with_data(_md_pq_map(fx)) do d
            @test length(collect(between(d, OptionBar, _MD_SPY, fx.t1a, fx.t2a))) == 4
            @test length(entry(d, OptionBar).chains) == 0
        end
    end

    @testset "parquet bars: asof walks partitions backward" begin
        with_data(_md_pq_map(fx)) do d
            @test asof(d, OptionBar, _MD_SPY, fx.t1b + Minute(1)) == at(d, OptionBar, _MD_SPY, fx.t1b)
            @test asof(d, OptionBar, _MD_SPY, fx.t2a - Minute(1)) == at(d, OptionBar, _MD_SPY, fx.t1b)   # day gap
            @test asof(d, OptionBar, _MD_SPY, fx.t2a) == at(d, OptionBar, _MD_SPY, fx.t2a)
            @test asof(d, OptionBar, _MD_SPY, fx.t2a + Day(30)) == at(d, OptionBar, _MD_SPY, fx.t2a)
            @test asof(d, OptionBar, _MD_SPY, fx.t1a - Minute(1)) == OptionBar[]
            @test asof(d, OptionBar, _MD_SPY, DateTime(2020)) == OptionBar[]
            @test asof(d, OptionQuote, _MD_SPY, fx.t1b + Minute(1)) == at(d, OptionQuote, _MD_SPY, fx.t1b)
        end
    end

    @testset "parquet bars: timestamps == old available_timestamps" begin
        with_data(_md_pq_map(fx)) do d
            @test timestamps(d, OptionBar, _MD_SPY, fx.t1a, fx.t2a) == [fx.t1a, fx.t1b, fx.t2a]
            @test timestamps(d, OptionQuote, _MD_SPY, fx.t1a, fx.t2a) == [fx.t1a, fx.t1b, fx.t2a]
            @test timestamps(d, Clock{OptionQuote}(_MD_SPY), fx.t1b, fx.t2a) == [fx.t1b, fx.t2a]
            @test timestamps(d, OptionBar, _MD_SPY, fx.t2a + Day(1), fx.t2a + Day(2)) == DateTime[]
            @test timestamps(d, OptionBar, _MD_SPY, fx.t2a, fx.t1a) == DateTime[]
            @test timestamps(d, OptionBar, Underlying("QQQ"), fx.t1a, fx.t2a) == DateTime[]
            # only the partition list was consulted; no chain was loaded
            @test length(entry(d, OptionBar).chains) == 0
            old = ParquetDataSource("SPY"; options_root=fx.opts_root, spot_root=fx.spot_root,
                                    synthesizer=_MD_PQ_SYNTH)
            @test timestamps(d, OptionBar, _MD_SPY, fx.t1a, fx.t2a) == available_timestamps(old, fx.t1a, fx.t2a)
            close(old)
        end
    end

    @testset "parquet bars: chain LRU bounded (max_chains=2)" begin
        r = open_data(ParquetOptionBars(fx.opts_root); max_chains=2)
        at(r, nothing, OptionBar, _MD_SPY, fx.t1a)
        at(r, nothing, OptionBar, _MD_SPY, fx.t1b)
        @test collect(keys(r.chains)) == [(_MD_SPY, fx.t1a), (_MD_SPY, fx.t1b)]
        at(r, nothing, OptionBar, _MD_SPY, fx.t1a)          # touch -> MRU
        at(r, nothing, OptionBar, _MD_SPY, fx.t2a)          # evicts t1b
        ks = collect(keys(r.chains))
        @test length(ks) == 2 && (_MD_SPY, fx.t1a) in ks && (_MD_SPY, fx.t2a) in ks
        @test !((_MD_SPY, fx.t1b) in ks)
        close_data!(r)
    end

    @testset "parquet spots: at / asof / between / timestamps, midnight spill" begin
        with_data(_md_pq_map(fx)) do d
            @test only_or_missing(at(d, SpotPrice, _MD_SPY, fx.t1a)).price == 480.0
            @test only_or_missing(at(d, SpotPrice, _MD_SPY, fx.t1b)).price == 480.5
            @test at(d, SpotPrice, _MD_SPY, DateTime(fx.d1, Time(15, 32))) == SpotPrice[]
            @test at(d, SpotPrice, _MD_SPY, DateTime(2024, 1, 17, 15, 30)) == SpotPrice[]
            # the 00:30 row lives in the date=2024-01-15 partition
            s = only_or_missing(at(d, SpotPrice, _MD_SPY, fx.spill))
            @test s.price == 480.7 && s.timestamp == fx.spill
            @test only_or_missing(asof(d, SpotPrice, _MD_SPY, DateTime(2024, 1, 16, 0, 45))).timestamp == fx.spill
            @test only_or_missing(asof(d, SpotPrice, _MD_SPY, fx.t2a + Hour(1))).timestamp == fx.t2a
            @test only_or_missing(asof(d, SpotPrice, _MD_SPY, fx.t1a)).timestamp == fx.t1a
            @test asof(d, SpotPrice, _MD_SPY, fx.t1a - Minute(1)) == SpotPrice[]

            out = between(d, SpotPrice, _MD_SPY, fx.t1a, fx.t2a)
            @test [s.timestamp for s in out] == [fx.t1a, fx.t1b, fx.spill, fx.t2a]
            @test [s.price for s in out] == [480.0, 480.5, 480.7, 481.0]
            @test [s.timestamp for s in between(d, SpotPrice, _MD_SPY, fx.spill, fx.spill)] == [fx.spill]
            @test isempty(between(d, SpotPrice, _MD_SPY, fx.t2a + Hour(1), fx.t2a + Hour(2)))
            @test isempty(between(d, SpotPrice, _MD_SPY, fx.t1b, fx.t1a))
            for ts in (fx.t1a, fx.t1b, fx.spill, fx.t2a, fx.t1a + Second(1))
                @test at(d, SpotPrice, _MD_SPY, ts) == between(d, SpotPrice, _MD_SPY, ts, ts)
            end
            @test timestamps(d, SpotPrice, _MD_SPY, fx.t1a, fx.t2a) == [fx.t1a, fx.t1b, fx.spill, fx.t2a]
            @test timestamps(d, SpotPrice, _MD_SPY, DateTime(2024, 1, 16), fx.t2a) == [fx.spill, fx.t2a]
        end
    end

    @testset "parquet: use after close_data! throws (cached or not)" begin
        d = open_data(_md_pq_map(fx))
        at(d, SpotPrice, _MD_SPY, fx.t1a)
        at(d, OptionBar, _MD_SPY, fx.t1a)
        rng = between(d, OptionBar, _MD_SPY, fx.t1a, fx.t2a)               # lazy, not yet iterated
        close_data!(d)
        @test_throws ArgumentError at(d, SpotPrice, _MD_SPY, fx.t1a)         # cached block
        @test_throws ArgumentError at(d, SpotPrice, _MD_SPY, fx.t2a)         # uncached
        @test_throws ArgumentError at(d, OptionBar, _MD_SPY, fx.t1a)         # cached chain
        @test_throws ArgumentError asof(d, OptionBar, _MD_SPY, fx.t2a)
        @test_throws ArgumentError timestamps(d, SpotPrice, _MD_SPY, fx.t1a, fx.t2a)
        @test_throws ArgumentError collect(rng)                              # iterator outlived the reader
        @test close_data!(d) === nothing                                     # idempotent
    end

    @testset "parquet: cross-check against the old ParquetDataSource" begin
        old = ParquetDataSource("SPY"; options_root=fx.opts_root, spot_root=fx.spot_root,
                                synthesizer=_MD_PQ_SYNTH)
        with_data(_md_pq_map(fx)) do d
            for ts in (fx.t1a, fx.t1b, fx.t2a)
                @test at(d, OptionQuote, _MD_SPY, ts) == get_chain(old, ts)
                @test only_or_missing(at(d, SpotPrice, _MD_SPY, ts)).price == get_spot(old, ts)
            end
            @test [s.price for s in between(d, SpotPrice, _MD_SPY, fx.t1a, fx.t2a)] ==
                  [s.price for s in get_spots(old, fx.t1a, fx.t2a)]
        end
        close(old)
    end
end

# ---------- volume / OHLC absent, ticker mismatch, parsed_* authoritative ----------

mktempdir() do root
    opts = joinpath(root, "options_1min")
    t = DateTime(2024, 1, 15, 15, 30)
    _md_write_options_parquet(joinpath(opts, "date=2024-01-15", "symbol=SPY", "data.parquet"),
                              [(ticker="O:SPY240129C00406000", close=1.05, timestamp=t)];
                              include_volume=false, include_ohlc=false)
    @testset "parquet bars: volume and OHLC columns absent -> missing" begin
        with_data(MarketData(ParquetOptionBars(opts), QuotesFromBars(_MD_PQ_SYNTH))) do d
            b = only(at(d, OptionBar, _MD_SPY, t))
            @test ismissing(b.volume) && ismissing(b.open) && ismissing(b.high) && ismissing(b.low)
            @test b.close == 1.05
            q = only(at(d, OptionQuote, _MD_SPY, t))
            @test ismissing(q.bid) && ismissing(q.ask) && q.mark == 1.05
        end
    end
end

mktempdir() do root
    opts = joinpath(root, "options_1min")
    t = DateTime(2024, 1, 15, 15, 30)
    _md_write_options_parquet(joinpath(opts, "date=2024-01-15", "symbol=SPY", "data.parquet"),
                              [(ticker="O:QQQ240129C00406000", close=1.05, volume=1.0, timestamp=t)])
    @testset "parquet bars: ticker-underlying mismatch throws" begin
        with_data(MarketData(ParquetOptionBars(opts))) do d
            @test_throws ArgumentError at(d, OptionBar, _MD_SPY, t)
            @test_throws ArgumentError collect(between(d, OptionBar, _MD_SPY, t, t))
        end
    end
end

mktempdir() do root
    opts = joinpath(root, "options_1min")
    t = DateTime(2024, 1, 15, 15, 30)
    db = DuckDB.DB(":memory:")
    DBInterface.execute(db, """
        CREATE TABLE bars (
            ticker VARCHAR, close DOUBLE, volume DOUBLE, timestamp TIMESTAMP,
            parsed_underlying VARCHAR, parsed_expiry TIMESTAMP,
            parsed_strike DOUBLE, parsed_option_type VARCHAR
        )
    """)
    DBInterface.execute(db, """
        INSERT INTO bars VALUES (
            'O:SPY240129C00406000', 1.05, 12.0, '$(Dates.format(t, "yyyy-mm-dd HH:MM:SS"))',
            'SPY', '2024-01-29 00:00:00', 999.0, 'P'
        )
    """)
    p = joinpath(opts, "date=2024-01-15", "symbol=SPY", "data.parquet")
    mkpath(dirname(p))
    DBInterface.execute(db, "COPY bars TO '$(replace(p, "\\" => "/"))' (FORMAT PARQUET)")
    DBInterface.close!(db)
    @testset "parquet bars: parsed_* columns authoritative over ticker text" begin
        with_data(MarketData(ParquetOptionBars(opts))) do d
            b = only(at(d, OptionBar, _MD_SPY, t))
            @test b.option_type == Put && b.strike == 999.0
            @test b.expiry == DateTime(2024, 1, 29, 21, 0)
        end
    end
end

# ---------- opt-in real-data smoke and old-vs-new cross-check ----------

if haskey(ENV, "VSA_POLYGON_ROOT")
    @testset "parquet real data (VSA_POLYGON_ROOT): smoke + cross-check" begin
        root = ENV["VSA_POLYGON_ROOT"]
        m = MarketData(ParquetOptionBars(joinpath(root, "options_1min")), QuotesFromBars(_MD_PQ_SYNTH),
                       ParquetSpots(joinpath(root, "spots_1min")))
        old = ParquetDataSource("SPY", root; synthesizer=_MD_PQ_SYNTH)
        day = Date(2024, 1, 16)
        # from 03:00, past any after-midnight spill of the previous partition,
        # which the old layer cannot see and the new one can
        w0, w1 = DateTime(day, Time(3, 0)), DateTime(day, Time(23, 59))
        with_data(m) do d
            ts = timestamps(d, OptionBar, _MD_SPY, w0, w1)
            @info "real-data cross-check" day n_timestamps = length(ts)
            @test !isempty(ts)
            @test ts == available_timestamps(old, w0, w1)
            for t in (first(ts), ts[end ÷ 2], last(ts))
                new_q = at(d, OptionQuote, _MD_SPY, t)
                @test new_q == get_chain(old, t)
                @test at(d, OptionBar, _MD_SPY, t) == collect(between(d, OptionBar, _MD_SPY, t, t))
                @test only_or_missing(at(d, SpotPrice, _MD_SPY, t)).price == get_spot(old, t)
            end
            @test asof(d, OptionQuote, _MD_SPY, w1) == get_chain(old, last(ts))
            spots = between(d, SpotPrice, _MD_SPY, DateTime(day), DateTime(day + Day(1), Time(2, 0)))
            @test !isempty(spots) && issorted(spots; by = s -> s.timestamp)
            @test any(s -> Date(s.timestamp) == day + Day(1), spots)      # the after-midnight spill exists
        end
        close(old)
    end
else
    @info "skipping parquet real-data cross-check (set VSA_POLYGON_ROOT to enable)"
end
