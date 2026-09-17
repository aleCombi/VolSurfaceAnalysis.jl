# Regression specifications for the confirmed PR #9 review findings. Each was
# red when it was written and all of them pass now; the file runs inside the
# main test gate (`test/runtests.jl`) so they stay that way.

using VolSurfaceAnalysis
using Test
using Dates

if !isdefined(@__MODULE__, :_md_write_spot_parquet)
    include(joinpath(@__DIR__, "..", "market_data", "fixtures.jl"))
end

const _RF_SPY = Underlying("SPY")
const _RF_QQQ = Underlying("QQQ")
const _RF_USD = Currency("USD")

# One opening order per contract at one instant, and nothing else.
struct _RF_OpenOnceAt <: Policy
    when::DateTime
    contracts::Vector{ContractKey}
end

function VolSurfaceAnalysis.decide(p::_RF_OpenOnceAt, t::DateTime,
                                   ::TimeCut, ::Book)::Vector{Order}
    t == p.when ? Order[Order(:leg, [Leg(c, Long, 1, Open)]) for c in p.contracts] : Order[]
end

@testset "PR #9 known-red review findings" begin

# The finding: every residual lot was settled from the clock's SPY spot, so a
# QQQ leg under a SPY clock filled against QQQ and settled against SPY. The
# engine now prices each leg against its own underlying and records what it
# saw, and settlement is a lifecycle event booked in the tick loop.
@testset "settlement uses trade underlying" begin
    t1 = DateTime(2024, 1, 15, 15, 30)
    expiry = DateTime(2024, 1, 15, 15, 31)       # inside the window (= the window end)
    far_expiry = DateTime(2024, 1, 15, 15, 35)   # past the window end
    inside = ContractKey(_RF_QQQ, 100.0, expiry, Call)
    past   = ContractKey(_RF_QQQ, 110.0, far_expiry, Call)
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
    L = result.ledger

    # Both fills are QQQ's ask (1.00, not SPY's) and their observations hold
    # QQQ's spot at t1 (100, not SPY's 90).
    fills = [e for e in L.events if e isa Fill]
    @test length(fills) == 2 && length(L.orders) == 2
    @test all(f.price == 1.00 for f in fills)
    @test [f.contract for f in fills] == [inside, past]
    @test all(only(r.observations).spot == 100.0 for r in L.orders)
    @test all(only(r.observations).spot_at == t1 for r in L.orders)
    @test all(only(r.observations).bid == 0.90 && only(r.observations).ask == 1.00 for r in L.orders)

    # Lifecycle in the tick loop settles the in-window leg by an Expiry against
    # QQQ's own spot at its expiry (120, not SPY's 90), a round trip of
    # (20.00 - 1.00) * 100 * 100 - 100 = 189900 cents, the 100 being the
    # commission on its opening fill (a lone contract raised to the USD 1.00
    # minimum).
    @test any(e isa Expiry && e.settlement_price == 120.0 && e.contract == inside for e in L.events) &&
          [r.pnl for r in round_trips(L)] == [189900]

    # The finding's second assertion, a window-end mark of the past-the-window
    # leg as a PnL sample of 9.00, is dropped: under proposal decision 8 an open
    # lot at the window end is marked at the evaluation endpoint by the equity
    # curve (slice 5), never force-settled into the realized series. The
    # in-window leg settles now, so the past-the-window lot is what is left.
    @test [l.contract for l in open_lots(book_effective(L, exp.to))] == [past]
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

# src/data/protocol/by_selector.jl:38 throws a bare KeyError when no route matches,
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

# `SurfaceFrom` once tried only the newest quote timestamp; because t2's expired
# chain is unbuildable, asof returned empty instead of t1's surface.
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

# src/data/providers/parquet.jl:392 concatenates spot rows without timestamp de-duplication,
# so the duplicated vendor row survives at(...) and its length is two today.
@testset "spot reads de-duplicate timestamps" begin
    mktempdir() do root
        ts = DateTime(2024, 1, 15, 15, 30)        # visible instant; the row is a bar earlier
        spots_root = joinpath(root, "spots_1min")
        path = joinpath(spots_root, "date=2024-01-15", "symbol=SPY", "data.parquet")
        _md_write_spot_parquet(path, _md_row.([ts, ts]), [480.0, 480.0])

        with_data(MarketData(ParquetSpots(spots_root))) do data
            @test length(at(data, SpotPrice, _RF_SPY, ts)) == 1
        end
    end
end

# src/data/providers/providers.jl:59 ignores Constant.record.timestamp,
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
