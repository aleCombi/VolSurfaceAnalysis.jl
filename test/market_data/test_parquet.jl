# Parquet specs and readers: hit/miss, fields, the protocol identities
# (at == collect(between), asof walks, the after-midnight spill), LRU
# bounds, use after close, and an opt-in real-data smoke.

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
    # a spec cannot answer the structural question: the tree is not open
    @test serves(s, nothing, OptionBar, Underlying("SPY")) === missing
    @test serves(ParquetSpots("/x"), nothing, SpotPrice, Underlying("SPY")) === missing
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
            # no date= partition holds a file for QQQ: structural, not temporal
            @test_throws UnservedSelector at(d, OptionBar, Underlying("QQQ"), fx.t1a)
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

    @testset "parquet bars: timestamps from the partition lists" begin
        with_data(_md_pq_map(fx)) do d
            @test timestamps(d, OptionBar, _MD_SPY, fx.t1a, fx.t2a) == [fx.t1a, fx.t1b, fx.t2a]
            @test timestamps(d, OptionQuote, _MD_SPY, fx.t1a, fx.t2a) == [fx.t1a, fx.t1b, fx.t2a]
            @test timestamps(d, Clock{OptionQuote}(_MD_SPY), fx.t1b, fx.t2a) == [fx.t1b, fx.t2a]
            @test timestamps(d, OptionBar, _MD_SPY, fx.t2a + Day(1), fx.t2a + Day(2)) == DateTime[]
            @test timestamps(d, OptionBar, _MD_SPY, fx.t2a, fx.t1a) == DateTime[]
            @test_throws UnservedSelector timestamps(d, OptionBar, Underlying("QQQ"), fx.t1a, fx.t2a)
            # only the partition list was consulted; no chain was loaded
            @test length(entry(d, OptionBar).chains) == 0
            # the reader answers the structural question from that same list
            r = entry(d, OptionBar)
            @test serves(r, nothing, OptionBar, _MD_SPY) === true
            @test serves(r, nothing, OptionBar, Underlying("QQQ")) === false
            @test serves(entry(d, SpotPrice), nothing, SpotPrice, _MD_SPY) === true
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

end

# ---------- time-ordered partitions; sub-second range bounds ----------
# Its own tree in the spill layout, so the shared fixture's assertions are
# untouched.

mktempdir() do root
    opts = joinpath(root, "options_1min")
    t1 = DateTime(2024, 1, 15, 15, 30)          # the 01-15 session
    spill = DateTime(2024, 1, 16, 0, 30)        # ... spilling past midnight UTC
    t2 = DateTime(2024, 1, 16, 15, 30)          # the 01-16 session
    _md_write_options_parquet(joinpath(opts, "date=2024-01-15", "symbol=SPY", "data.parquet"),
                              [(ticker="O:SPY240129C00406000", close=1.05, volume=1.0,
                                open=1.0, high=1.1, low=1.0, timestamp=t1),
                               (ticker="O:SPY240129C00406000", close=1.06, volume=1.0,
                                open=1.0, high=1.1, low=1.0, timestamp=spill)])
    _md_write_options_parquet(joinpath(opts, "date=2024-01-16", "symbol=SPY", "data.parquet"),
                              [(ticker="O:SPY240129C00406000", close=1.20, volume=1.0,
                                open=1.2, high=1.3, low=1.1, timestamp=t2)])

    @testset "parquet bars: partitions are time-ordered, so the shapes agree" begin
        with_data(MarketData(ParquetOptionBars(opts))) do d
            r = entry(d, OptionBar)
            p15 = VolSurfaceAnalysis._meta(r, _MD_SPY, Date(2024, 1, 15)).timestamps
            p16 = VolSurfaceAnalysis._meta(r, _MD_SPY, Date(2024, 1, 16)).timestamps
            # the convention: every row in D-1 precedes every row in D, spill included
            @test last(p15) < first(p16)

            # under it, asof agrees with the newest instant the grid reports
            for ts in (t1, spill, spill + Minute(1), t2, t2 + Hour(1))
                grid = timestamps(d, OptionBar, _MD_SPY, DateTime(2024, 1, 15), ts)
                @test !isempty(grid)
                @test asof(d, OptionBar, _MD_SPY, ts) == at(d, OptionBar, _MD_SPY, last(grid))
            end

            # ... and the lazy cross-partition walk stays sorted, so by_timestamp holds
            rng = between(d, OptionBar, _MD_SPY, t1, t2)
            @test issorted([b.timestamp for b in rng])
            @test first.(collect(by_timestamp(rng))) == [t1, spill, t2]
        end
    end

    @testset "parquet bars: a sub-second lower bound excludes the row at its floor" begin
        with_data(MarketData(ParquetOptionBars(opts))) do d
            lo = t1 + Millisecond(500)
            @test [b.timestamp for b in between(d, OptionBar, _MD_SPY, lo, spill)] == [spill]
            @test timestamps(d, OptionBar, _MD_SPY, lo, spill) == [spill]
            # the exact-instant predicate is unaffected
            @test [b.timestamp for b in at(d, OptionBar, _MD_SPY, t1)] == [t1]
            @test at(d, OptionBar, _MD_SPY, lo) == OptionBar[]
        end
    end
end

# ---------- spot duplicates: collapse identical, throw on conflict ----------
# Its own small tree rather than the shared fixture, so the assertions above
# keep describing a store with no duplicates.

mktempdir() do root
    spots = joinpath(root, "spots_1min")
    t1 = DateTime(2024, 1, 15, 15, 30)
    spill = DateTime(2024, 1, 16, 0, 30)
    t2 = DateTime(2024, 1, 16, 15, 30)
    # t1 delivered twice inside one partition; the spill row present in both
    # the 01-15 partition (per the convention) and the 01-16 body.
    _md_write_spot_parquet(joinpath(spots, "date=2024-01-15", "symbol=SPY", "data.parquet"),
                           [t1, t1, spill], [480.0, 480.0, 480.7])
    _md_write_spot_parquet(joinpath(spots, "date=2024-01-16", "symbol=SPY", "data.parquet"),
                           [spill, t2], [480.7, 481.0])

    @testset "parquet spots: identical duplicates collapse, in and across partitions" begin
        with_data(MarketData(ParquetSpots(spots))) do d
            @test only_or_missing(at(d, SpotPrice, _MD_SPY, t1)).price == 480.0
            @test only_or_missing(at(d, SpotPrice, _MD_SPY, spill)).price == 480.7
            out = between(d, SpotPrice, _MD_SPY, t1, t2)
            @test [s.timestamp for s in out] == [t1, spill, t2]
            @test [s.price for s in out] == [480.0, 480.7, 481.0]
            @test timestamps(d, SpotPrice, _MD_SPY, t1, t2) == [t1, spill, t2]
        end
    end
end

mktempdir() do root
    spots = joinpath(root, "spots_1min")
    t = DateTime(2024, 1, 15, 15, 30)
    _md_write_spot_parquet(joinpath(spots, "date=2024-01-15", "symbol=SPY", "data.parquet"),
                           [t, t], [480.0, 481.0])

    @testset "parquet spots: two prices at one instant throw ConflictingRecords" begin
        with_data(MarketData(ParquetSpots(spots))) do d
            @test_throws ConflictingRecords at(d, SpotPrice, _MD_SPY, t)
            @test_throws ConflictingRecords between(d, SpotPrice, _MD_SPY, t, t)
            err = try
                at(d, SpotPrice, _MD_SPY, t)
            catch e
                e
            end
            msg = sprint(showerror, err)
            @test occursin("480.0", msg) && occursin("481.0", msg)
            @test occursin("SPY", msg) && occursin(string(t), msg)
        end
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

# ---------- opt-in real-data smoke ----------

if haskey(ENV, "VSA_POLYGON_ROOT")
    @testset "parquet real data (VSA_POLYGON_ROOT): smoke" begin
        root = ENV["VSA_POLYGON_ROOT"]
        m = MarketData(ParquetOptionBars(joinpath(root, "options_1min")), QuotesFromBars(_MD_PQ_SYNTH),
                       ParquetSpots(joinpath(root, "spots_1min")))
        day = Date(2024, 1, 16)
        w0, w1 = DateTime(day, Time(3, 0)), DateTime(day, Time(23, 59))
        with_data(m) do d
            ts = timestamps(d, OptionBar, _MD_SPY, w0, w1)
            @info "real-data smoke" day n_timestamps = length(ts)
            @test !isempty(ts)
            for t in (first(ts), ts[end ÷ 2], last(ts))
                @test !isempty(at(d, OptionQuote, _MD_SPY, t))
                @test at(d, OptionBar, _MD_SPY, t) == collect(between(d, OptionBar, _MD_SPY, t, t))
                @test !ismissing(only_or_missing(at(d, SpotPrice, _MD_SPY, t)))
            end
            @test asof(d, OptionQuote, _MD_SPY, w1) == at(d, OptionQuote, _MD_SPY, last(ts))
            spots = between(d, SpotPrice, _MD_SPY, DateTime(day), DateTime(day + Day(1), Time(2, 0)))
            @test !isempty(spots) && issorted(spots; by = s -> s.timestamp)
            @test any(s -> Date(s.timestamp) == day + Day(1), spots)      # the after-midnight spill exists
        end
    end
else
    @info "skipping parquet real-data smoke (set VSA_POLYGON_ROOT to enable)"
end
