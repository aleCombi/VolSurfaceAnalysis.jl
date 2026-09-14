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
            @test fill_price(:cross_spread, c.bid, c.ask, Long,  1) == 1.07   # the ask 1.065 rounded up to the tick
            @test fill_price(:cross_spread, c.bid, c.ask, Short, 1) == 1.02   # the bid, already on the tick
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

    @testset "parquet: a record is visible at bar end, not at bar open" begin
        # The rows are stamped a minute before the instants below; the
        # fixture keeps both so the mapping is asserted, not assumed.
        with_data(_md_pq_map(fx)) do d
            @test fx.t1a == fx.row_t1a + Minute(1)
            @test fx.spill == fx.row_spill + Minute(1)
            # nothing is knowable at the row stamp
            @test at(d, OptionBar, _MD_SPY, fx.row_t1a) == OptionBar[]
            @test at(d, OptionQuote, _MD_SPY, fx.row_t1a) == OptionQuote[]
            @test at(d, SpotPrice, _MD_SPY, fx.row_t1a) == SpotPrice[]
            # ... and the whole completed minute is, one minute later
            c = only(filter(b -> b.option_type == Call, at(d, OptionBar, _MD_SPY, fx.t1a)))
            @test (c.open, c.high, c.low, c.close) == (1.00, 1.10, 0.95, 1.05)
            @test c.timestamp == fx.t1a
            @test only_or_missing(at(d, SpotPrice, _MD_SPY, fx.t1a)).price == 480.0

            # synthesis preserves the instant: bar end, not a second minute
            q = only(filter(x -> x.option_type == Call, at(d, OptionQuote, _MD_SPY, fx.t1a)))
            @test q.timestamp == fx.t1a
            @test q.bid ≈ 0.95 + 0.7 * (1.05 - 0.95)      # low + λ(close - low)
            @test q.ask ≈ 1.10 - 0.7 * (1.10 - 1.05)      # high - λ(high - close)
            @test q.mark == 1.05
            @test synthesize(_MD_PQ_SYNTH, c).timestamp == fx.t1a
        end
    end

    @testset "parquet: a cut before bar end hides the bar and its quote" begin
        # This is the defect the convention exists to prevent: under a
        # bar-open stamp the cut at 15:30 admitted a record built from the
        # 15:30-15:31 minute, which had not finished.
        with_data(_md_pq_map(fx)) do d
            before = TimeCut(d, fx.t1a - Millisecond(1))
            @test at(before, OptionBar, _MD_SPY, fx.t1a) == OptionBar[]
            @test at(before, OptionQuote, _MD_SPY, fx.t1a) == OptionQuote[]
            @test at(before, SpotPrice, _MD_SPY, fx.t1a) == SpotPrice[]
            @test asof(before, OptionBar, _MD_SPY, fx.t1a) == OptionBar[]
            @test asof(before, OptionQuote, _MD_SPY, fx.t1a) == OptionQuote[]
            @test asof(before, SpotPrice, _MD_SPY, fx.t1a) == SpotPrice[]
            @test isempty(collect(between(before, OptionBar, _MD_SPY, fx.row_t1a, fx.t1a)))
            @test timestamps(before, OptionBar, _MD_SPY, fx.row_t1a, fx.t1a) == DateTime[]
            @test timestamps(before, SpotPrice, _MD_SPY, fx.row_t1a, fx.t1a) == DateTime[]

            at_end = TimeCut(d, fx.t1a)
            @test length(at(at_end, OptionBar, _MD_SPY, fx.t1a)) == 2
            @test length(at(at_end, OptionQuote, _MD_SPY, fx.t1a)) == 2
            @test only_or_missing(at(at_end, SpotPrice, _MD_SPY, fx.t1a)).price == 480.0
            @test timestamps(at_end, OptionBar, _MD_SPY, fx.row_t1a, fx.t1a) == [fx.t1a]
            @test timestamps(at_end, SpotPrice, _MD_SPY, fx.row_t1a, fx.t1a) == [fx.t1a]
        end
    end

    @testset "parquet: the four shapes agree on bar end, in both trees" begin
        with_data(_md_pq_map(fx)) do d
            before = fx.t1a - Millisecond(1)                 # a fractional bound
            after  = fx.t1a + Millisecond(1)
            for (R, n) in ((OptionBar, 2), (SpotPrice, 1))
                # before
                @test isempty(at(d, R, _MD_SPY, before))
                @test isempty(collect(between(d, R, _MD_SPY, fx.row_t1a, before)))
                @test isempty(asof(d, R, _MD_SPY, before))
                @test timestamps(d, R, _MD_SPY, fx.row_t1a, before) == DateTime[]
                # at
                @test length(at(d, R, _MD_SPY, fx.t1a)) == n
                @test [r.timestamp for r in collect(between(d, R, _MD_SPY, before, fx.t1a))] ==
                      fill(fx.t1a, n)
                @test [r.timestamp for r in asof(d, R, _MD_SPY, fx.t1a)] == fill(fx.t1a, n)
                @test timestamps(d, R, _MD_SPY, before, fx.t1a) == [fx.t1a]
                # after: still the completed bar, and no new instant
                @test isempty(at(d, R, _MD_SPY, after))
                @test [r.timestamp for r in asof(d, R, _MD_SPY, after)] == fill(fx.t1a, n)
                @test timestamps(d, R, _MD_SPY, after, fx.t1b - Millisecond(1)) == DateTime[]
            end
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
    # visibility instants; the rows are written one bar earlier
    t1 = DateTime(2024, 1, 15, 15, 30)          # the 01-15 session
    spill = DateTime(2024, 1, 16, 0, 30)        # ... spilling past midnight UTC
    t2 = DateTime(2024, 1, 16, 15, 30)          # the 01-16 session
    _md_write_options_parquet(joinpath(opts, "date=2024-01-15", "symbol=SPY", "data.parquet"),
                              [(ticker="O:SPY240129C00406000", close=1.05, volume=1.0,
                                open=1.0, high=1.1, low=1.0, timestamp=_md_row(t1)),
                               (ticker="O:SPY240129C00406000", close=1.06, volume=1.0,
                                open=1.0, high=1.1, low=1.0, timestamp=_md_row(spill))])
    _md_write_options_parquet(joinpath(opts, "date=2024-01-16", "symbol=SPY", "data.parquet"),
                              [(ticker="O:SPY240129C00406000", close=1.20, volume=1.0,
                                open=1.2, high=1.3, low=1.1, timestamp=_md_row(t2))])

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
    # the 01-15 partition and the 01-16 body -- a layout the time-ordered
    # convention forbids and nothing enforces.
    _md_write_spot_parquet(joinpath(spots, "date=2024-01-15", "symbol=SPY", "data.parquet"),
                           _md_row.([t1, t1, spill]), [480.0, 480.0, 480.7])
    _md_write_spot_parquet(joinpath(spots, "date=2024-01-16", "symbol=SPY", "data.parquet"),
                           _md_row.([spill, t2]), [480.7, 481.0])

    @testset "parquet spots: identical duplicates collapse, in and across partitions" begin
        with_data(MarketData(ParquetSpots(spots))) do d
            @test only_or_missing(at(d, SpotPrice, _MD_SPY, t1)).price == 480.0
            @test only_or_missing(at(d, SpotPrice, _MD_SPY, spill)).price == 480.7
            out = between(d, SpotPrice, _MD_SPY, t1, t2)
            @test [s.timestamp for s in out] == [t1, spill, t2]
            @test [s.price for s in out] == [480.0, 480.7, 481.0]
            @test timestamps(d, SpotPrice, _MD_SPY, t1, t2) == [t1, spill, t2]

            # asof obeys the rule too: it reads its winning instant through
            # `between`, so the repeat inside one partition and the copy
            # across the overlap both collapse.
            @test only_or_missing(asof(d, SpotPrice, _MD_SPY, t1)).price == 480.0
            @test only_or_missing(asof(d, SpotPrice, _MD_SPY, spill + Minute(1))).price == 480.7

            # ... and the identity the shapes owe each other still holds here
            for ts in (t1, spill, spill + Minute(1), t2, t2 + Hour(1))
                grid = timestamps(d, SpotPrice, _MD_SPY, t1, ts)
                @test !isempty(grid)
                @test asof(d, SpotPrice, _MD_SPY, ts) == at(d, SpotPrice, _MD_SPY, last(grid))
            end
        end
    end
end

# The same overlap, disagreeing. `asof` reading the winning block directly
# would return the later partition's price with no diagnostic while `at`
# threw -- the silent choice between two answers the rule exists to remove.
mktempdir() do root
    spots = joinpath(root, "spots_1min")
    t1 = DateTime(2024, 1, 15, 15, 30)
    spill = DateTime(2024, 1, 16, 0, 30)
    _md_write_spot_parquet(joinpath(spots, "date=2024-01-15", "symbol=SPY", "data.parquet"),
                           _md_row.([t1, spill]), [480.0, 480.7])
    _md_write_spot_parquet(joinpath(spots, "date=2024-01-16", "symbol=SPY", "data.parquet"),
                           _md_row.([spill]), [499.9])

    @testset "parquet spots: a conflict across the overlap throws in asof too" begin
        with_data(MarketData(ParquetSpots(spots))) do d
            @test_throws ConflictingRecords asof(d, SpotPrice, _MD_SPY, spill)
            @test_throws ConflictingRecords asof(d, SpotPrice, _MD_SPY, spill + Hour(1))
            @test_throws ConflictingRecords at(d, SpotPrice, _MD_SPY, spill)
            # the instant before it is untouched
            @test only_or_missing(asof(d, SpotPrice, _MD_SPY, t1)).price == 480.0
        end
    end
end

# The convention's allowance is one day: a partition for date D may hold
# rows spilling into the early hours of D + 1, and every shape reads the
# two candidate partitions D - 1 and D. A row stamped further past its
# partition is outside the convention, and the shapes agree it is not
# there -- `asof` included, now that it reads its winning instant through
# `between` rather than the block its walk found it in. Note what `asof`
# does *not* do: it does not fall back to the in-allowance row either;
# its walk stops at the stray row and `between` finds nothing there.
# Pinned so the allowance is widened deliberately or not at all.
mktempdir() do root
    spots = joinpath(root, "spots_1min")
    body   = DateTime(2024, 1, 12, 15, 30)
    inside = DateTime(2024, 1, 13, 0, 30)      # D + 1, within the allowance
    beyond = DateTime(2024, 1, 14, 0, 30)      # D + 2, outside it
    _md_write_spot_parquet(joinpath(spots, "date=2024-01-12", "symbol=SPY", "data.parquet"),
                           _md_row.([body, inside, beyond]), [480.0, 480.5, 480.9])

    @testset "parquet spots: a row more than one day past its partition is invisible to every shape" begin
        with_data(MarketData(ParquetSpots(spots))) do d
            # within the allowance, every shape sees it
            @test only_or_missing(at(d, SpotPrice, _MD_SPY, inside)).price == 480.5
            @test only_or_missing(asof(d, SpotPrice, _MD_SPY, inside + Hour(1))).timestamp == inside
            @test timestamps(d, SpotPrice, _MD_SPY, body, inside) == [body, inside]
            # beyond it, none does -- asof included, and it reports nothing
            # rather than the in-allowance row
            @test at(d, SpotPrice, _MD_SPY, beyond) == SpotPrice[]
            @test between(d, SpotPrice, _MD_SPY, beyond, beyond) == SpotPrice[]
            @test timestamps(d, SpotPrice, _MD_SPY, beyond, beyond) == DateTime[]
            @test asof(d, SpotPrice, _MD_SPY, beyond) == SpotPrice[]
            @test asof(d, SpotPrice, _MD_SPY, beyond + Hour(1)) == SpotPrice[]
        end
    end
end

mktempdir() do root
    spots = joinpath(root, "spots_1min")
    t = DateTime(2024, 1, 15, 15, 30)
    _md_write_spot_parquet(joinpath(spots, "date=2024-01-15", "symbol=SPY", "data.parquet"),
                           _md_row.([t, t]), [480.0, 481.0])

    @testset "parquet spots: two prices at one instant throw ConflictingRecords" begin
        with_data(MarketData(ParquetSpots(spots))) do d
            @test_throws ConflictingRecords at(d, SpotPrice, _MD_SPY, t)
            @test_throws ConflictingRecords between(d, SpotPrice, _MD_SPY, t, t)
            @test_throws ConflictingRecords asof(d, SpotPrice, _MD_SPY, t)
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
                              [(ticker="O:SPY240129C00406000", close=1.05, timestamp=_md_row(t))];
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
                              [(ticker="O:QQQ240129C00406000", close=1.05, volume=1.0, timestamp=_md_row(t))])
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
            'O:SPY240129C00406000', 1.05, 12.0, '$(Dates.format(_md_row(t), "yyyy-mm-dd HH:MM:SS"))',
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

# ---------- bar end across midnight and at the session close ----------
# Bar-end visibility moves a 23:59 row into the NEXT calendar date while it
# stays in the partition of its own. The partition walk already consults
# Date(ts) - 1, so every shape must still find it at next-day 00:00 -- with
# no next-day partition in existence, which is the case the walk could most
# easily lose.

mktempdir() do root
    opts = joinpath(root, "options_1min")
    spots = joinpath(root, "spots_1min")
    early = DateTime(2024, 1, 15, 23, 59)       # visible; row at 23:58
    cross = DateTime(2024, 1, 16, 0, 0)         # visible; row at 23:59, same partition
    _md_write_options_parquet(joinpath(opts, "date=2024-01-15", "symbol=SPY", "data.parquet"),
                              [(ticker="O:SPY240129C00406000", close=1.05, volume=1.0,
                                open=1.0, high=1.1, low=1.0, timestamp=_md_row(early)),
                               (ticker="O:SPY240129C00406000", close=2.05, volume=1.0,
                                open=2.0, high=2.1, low=2.0, timestamp=_md_row(cross))])
    _md_write_spot_parquet(joinpath(spots, "date=2024-01-15", "symbol=SPY", "data.parquet"),
                           _md_row.([early, cross]), [480.0, 481.0])

    @testset "parquet: a 23:59 row is found at next-day 00:00 by every shape" begin
        @test readdir(opts) == ["date=2024-01-15"]          # no next-day partition
        @test readdir(spots) == ["date=2024-01-15"]
        m = MarketData(ParquetOptionBars(opts), QuotesFromBars(_MD_PQ_SYNTH), ParquetSpots(spots))
        with_data(m) do d
            for R in (OptionBar, SpotPrice)
                @test [r.timestamp for r in at(d, R, _MD_SPY, cross)] == [cross]
                @test [r.timestamp for r in collect(between(d, R, _MD_SPY, cross, cross))] == [cross]
                @test [r.timestamp for r in asof(d, R, _MD_SPY, cross)] == [cross]
                @test [r.timestamp for r in asof(d, R, _MD_SPY, cross + Hour(6))] == [cross]
                @test timestamps(d, R, _MD_SPY, cross, cross) == [cross]
                @test timestamps(d, R, _MD_SPY, early, cross) == [early, cross]
                # the identity `at == collect(between(ts, ts))` survives the date change
                @test at(d, R, _MD_SPY, cross) == collect(between(d, R, _MD_SPY, cross, cross))
                # a fractional bound just short of midnight excludes it
                @test isempty(at(d, R, _MD_SPY, cross - Millisecond(1)))
                @test [r.timestamp for r in collect(
                        between(d, R, _MD_SPY, early + Millisecond(500), cross))] == [cross]
                @test timestamps(d, R, _MD_SPY, early + Millisecond(500), cross) == [cross]
                @test [r.timestamp for r in asof(d, R, _MD_SPY, cross - Millisecond(1))] == [early]
            end
            # and the derived quote crosses with its bar
            @test only(at(d, OptionQuote, _MD_SPY, cross)).timestamp == cross
            @test only_or_missing(at(d, SpotPrice, _MD_SPY, cross)).price == 481.0
        end
    end
end

# The session-close lookup the marked-curve round needs. The last option bar
# of a US session is the 15:59-16:00 ET minute, stamped 20:59 UTC; under
# bar-end visibility it is knowable at 21:00 UTC, which is the close itself,
# so an exact quote lookup at the close finds it with no staleness window
# and no offset in the marking code.
mktempdir() do root
    opts = joinpath(root, "options_1min")
    d = Date(2024, 1, 16)
    close_utc = et_to_utc(d, Time(16, 0))                 # 21:00 UTC
    _md_write_options_parquet(
        joinpath(opts, "date=2024-01-16", "symbol=SPY", "data.parquet"),
        [(ticker="O:SPY240129C00406000", close=1.05, volume=1.0,
          open=1.0, high=1.1, low=1.0, timestamp=et_to_utc(d, Time(15, 58))),
         (ticker="O:SPY240129C00406000", close=1.11, volume=1.0,
          open=1.1, high=1.2, low=1.1, timestamp=et_to_utc(d, Time(15, 59)))])

    @testset "parquet: an exact lookup at the session close finds the last completed bar" begin
        with_data(MarketData(ParquetOptionBars(opts), QuotesFromBars(_MD_PQ_SYNTH))) do data
            b = only(at(data, OptionBar, _MD_SPY, close_utc))
            @test b.timestamp == close_utc && b.close == 1.11
            q = only(at(data, OptionQuote, _MD_SPY, close_utc))
            @test q.timestamp == close_utc && q.mark == 1.11
            # through a cut stamped at the close, which is how the mark reads it
            cut = TimeCut(data, close_utc)
            @test only(at(cut, OptionQuote, _MD_SPY, close_utc)).mark == 1.11
            @test last(timestamps(data, OptionBar, _MD_SPY, DateTime(d), close_utc)) == close_utc
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
