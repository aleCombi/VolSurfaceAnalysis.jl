# Known-red regression specifications for the confirmed PR #9 review findings.
# This file is intentionally excluded from the main test gate.

using VolSurfaceAnalysis
using Test
using Dates

if !isdefined(@__MODULE__, :_md_write_spot_parquet)
    include(joinpath(@__DIR__, "..", "market_data", "fixtures.jl"))
end

const _RF_SPY = Underlying("SPY")
const _RF_QQQ = Underlying("QQQ")
const _RF_USD = Currency("USD")

struct _RF_OpenOnceAt <: Policy
    when::DateTime
    trades::Vector{Trade}
end

function VolSurfaceAnalysis.decide(p::_RF_OpenOnceAt, t::DateTime,
                                   ::TimeCut,
                                   ::AbstractVector{Position})::Vector{Trade}
    t == p.when ? copy(p.trades) : Trade[]
end

@testset "PR #9 known-red review findings" begin

# src/experiment/experiment.jl:126 settles every residual lot from the clock's
# SPY spot: the in-window leg is marked at SPY's expiry spot instead of QQQ's,
# and the past-the-window leg at the SPY window-end spot instead of QQQ's.
# Both cases presume settlement follows each lot's own trade underlying.
@testset "settlement uses trade underlying" begin
    t1 = DateTime(2024, 1, 15, 15, 30)
    expiry = DateTime(2024, 1, 15, 15, 31)       # inside the window (= the window end)
    far_expiry = DateTime(2024, 1, 15, 15, 35)   # past the window end
    inside = Trade(_RF_QQQ, 100.0, expiry, Call)
    past   = Trade(_RF_QQQ, 110.0, far_expiry, Call)
    qqq_quotes = [
        OptionQuote("QQQ-in", _RF_QQQ, expiry, 100.0, Call,
                    0.90, 1.00, 0.95, missing, missing, missing, t1),
        OptionQuote("QQQ-past", _RF_QQQ, far_expiry, 110.0, Call,
                    0.90, 1.00, 0.95, missing, missing, missing, t1),
    ]
    spy_clock = [
        OptionQuote("SPY", _RF_SPY, expiry, 90.0, Call,
                    0.90, 1.00, 0.95, missing, missing, missing, ts)
        for ts in (t1, expiry)
    ]
    spots = [
        SpotPrice(_RF_SPY, 90.0, t1),
        SpotPrice(_RF_QQQ, 100.0, t1),
        SpotPrice(_RF_SPY, 90.0, expiry),
        SpotPrice(_RF_QQQ, 120.0, expiry),
    ]
    data = MarketData(InMemory(vcat(spy_clock, qqq_quotes)), InMemory(spots))
    exp = Experiment(name="mismatched-underlyings",
                     agent=StaticAgent(_RF_OpenOnceAt(t1, [inside, past])),
                     data=data, clock=Clock{OptionQuote}(_RF_SPY),
                     from=t1, to=expiry,
                     outputs=OutputSpec(metrics=Symbol[], artifacts=Symbol[]))

    result = run_experiment(exp)

    # In-window leg: QQQ spot at its own expiry is 120, so max(120-100,0) - 1.00.
    # Past-the-window leg: QQQ spot at the window end is 120, so max(120-110,0) - 1.00.
    # Stamped at `expiry` and `far_expiry`, so the series is in that order.
    @test result.pnl_series.pnl == [19.0, 9.0]
end

# src/experiment/config.jl:212 checks only input kinds, not Constant selectors;
# the USD rate below is accepted for an EUR SurfaceFrom instead of failing at load time.
# The roots point at nonexistent trees on purpose: the load-time check must be
# structural, not a filesystem probe, or this passes for the wrong reason. The
# message is matched rather than the type, for the same reason.
@testset "constant curve selector matches surface selector" begin
    mismatched_currency = """
        name = "mismatched-currency"
        from = 2024-01-15T15:30:00
        to = 2024-01-15T15:31:00
        clock = { kind = "option_quote", underlying = "SPY" }

        [data.option_bar]
        type = "parquet_option_bars"
        root = "/nonexistent/options"

        [data.option_quote]
        type = "from_bars"
        synthesizer = { type = "ohlcv_spread", lambda = 0.7 }

        [data.spot_price]
        type = "parquet_spots"
        root = "/nonexistent/spots"

        [data.rate_curve]
        type = "constant"
        currency = "USD"
        value = 0.04

        [data.div_curve]
        type = "constant"
        underlying = "SPY"
        value = 0.015

        [data.vol_surface]
        type = "surface_from"
        currency = "EUR"

        [agent]
        type = "static"

        [agent.policy]
        type = "noop"
        """

    err = try
        load_experiment_str(mismatched_currency)
        nothing
    catch e
        e
    end
    @test err !== nothing
    msg = err === nothing ? "" : sprint(showerror, err)
    @test occursin("rate_curve", msg) || occursin("RateCurve", msg)
    @test occursin("EUR", msg)
    @test occursin("USD", msg)
end

# src/market_data/by_selector.jl:38 throws a bare KeyError when no route matches,
# and every other provider answers the same question with an empty vector. Per the
# decision on findings 2 and 3, structural absence is a named error everywhere:
# BySelector is the provider that was already right, and all four shapes must
# report it as `UnservedSelector` rather than as ordinary emptiness.
@testset "BySelector throws for an unrouted selector" begin
    t1 = DateTime(2024, 1, 15, 15, 30)
    t2 = t1 + Minute(1)
    routed = InMemory([SpotPrice(_RF_SPY, 480.0, t1)])
    data = MarketData(BySelector{SpotPrice}(_RF_SPY => routed))

    @test_throws UnservedSelector at(data, SpotPrice, _RF_QQQ, t1)
    @test_throws UnservedSelector collect(between(data, SpotPrice, _RF_QQQ, t1, t2))
    @test_throws UnservedSelector asof(data, SpotPrice, _RF_QQQ, t2)
    @test_throws UnservedSelector timestamps(data, SpotPrice, _RF_QQQ, t1, t2)
end

# src/surfaces/surface_from.jl:74 tries only the newest quote timestamp;
# because t2's expired chain is unbuildable, asof returns empty instead of t1's surface.
@testset "surface asof walks back past unbuildable chains" begin
    t1 = DateTime(2024, 1, 15, 15, 30)
    t2 = t1 + Minute(1)
    valid_expiry = DateTime(2024, 2, 16, 21, 0)
    expired = t2 - Day(1)
    spot = 480.0
    sigma = 0.20
    valid_mark = bs_price(spot, 480.0, time_to_expiry(valid_expiry, t1), sigma, Call;
                          r=0.04, q=0.015)
    quotes = [
        OptionQuote("valid", _RF_SPY, valid_expiry, 480.0, Call,
                    valid_mark, valid_mark, valid_mark, missing, missing, missing, t1),
        OptionQuote("expired", _RF_SPY, expired, 480.0, Call,
                    1.0, 1.0, 1.0, missing, missing, missing, t2),
    ]
    data = MarketData(
        InMemory(quotes),
        InMemory([SpotPrice(_RF_SPY, spot, t1), SpotPrice(_RF_SPY, spot, t2)]),
        Constant(RateCurve(_RF_USD, FlatCurve(0.04))),
        Constant(DivCurve(_RF_SPY, FlatCurve(0.015))),
        SurfaceFrom(currency=_RF_USD),
    )

    with_data(data) do opened
        prior = at(opened, VolatilitySurface, _RF_SPY, t1)
        # Without this the comparison below can pass vacuously, empty against empty.
        @test !isempty(prior)
        @test asof(opened, VolatilitySurface, _RF_SPY, t2) == prior
    end
end

# src/market_data/parquet.jl:392 concatenates spot rows without timestamp de-duplication,
# so the duplicated vendor row survives at(...) and its length is two today.
@testset "spot reads de-duplicate timestamps" begin
    mktempdir() do root
        ts = DateTime(2024, 1, 15, 15, 30)
        spots_root = joinpath(root, "spots_1min")
        path = joinpath(spots_root, "date=2024-01-15", "symbol=SPY", "data.parquet")
        _md_write_spot_parquet(path, [ts, ts], [480.0, 480.0])

        with_data(MarketData(ParquetSpots(spots_root))) do data
            @test length(at(data, SpotPrice, _RF_SPY, ts)) == 1
        end
    end
end

# src/market_data/providers.jl:59 ignores Constant.record.timestamp,
# so the June curve is incorrectly visible to an asof query in January.
@testset "constant asof respects record timestamp" begin
    visible_from = DateTime(2024, 6, 1)
    record = RateCurve(_RF_USD, FlatCurve(0.04), visible_from)
    data = MarketData(Constant(record))

    actual = (
        asof(data, RateCurve, _RF_USD, DateTime(2024, 1, 15)),
        asof(data, RateCurve, _RF_USD, visible_from),
    )
    @test actual == (RateCurve[], [record])
end

end
