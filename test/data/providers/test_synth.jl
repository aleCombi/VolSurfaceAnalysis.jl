# Tests for the OHLCV -> OptionQuote synthesis adapter.

const _SYNTH_U = Underlying("SPY")
const _SYNTH_EXP = DateTime(2024, 1, 29, 21, 0)
const _SYNTH_TS  = DateTime(2024, 1, 15, 15, 30)

_mk_bar(; open=1.00, high=1.20, low=0.80, close=1.00, volume=10.0) = OptionBar(
    "O:SPY240129C00406000", _SYNTH_U, _SYNTH_EXP, 406.0, Call,
    open, high, low, close, volume, _SYNTH_TS,
)

@testset "OptionBar: all fields round-trip" begin
    bar = _mk_bar()
    @test bar.instrument_id == "O:SPY240129C00406000"
    @test ticker(bar.underlying) == "SPY"
    @test bar.expiry == _SYNTH_EXP
    @test bar.strike == 406.0
    @test bar.option_type == Call
    @test bar.open == 1.00
    @test bar.high == 1.20
    @test bar.low  == 0.80
    @test bar.close == 1.00
    @test bar.volume == 10.0
    @test bar.timestamp == _SYNTH_TS
end

@testset "OptionBar: OHLCV fields accept missing" begin
    bar = OptionBar("X", _SYNTH_U, _SYNTH_EXP, 406.0, Put,
                    missing, missing, missing, missing, missing, _SYNTH_TS)
    @test ismissing(bar.open)
    @test ismissing(bar.high)
    @test ismissing(bar.low)
    @test ismissing(bar.close)
    @test ismissing(bar.volume)
end

@testset "SpreadFromOHLCV: lambda required and validated in [0, 1]" begin
    @test_throws MethodError SpreadFromOHLCV()       # no default
    @test_throws ArgumentError SpreadFromOHLCV(-0.1)
    @test_throws ArgumentError SpreadFromOHLCV(1.5)
    @test SpreadFromOHLCV(0.0).lambda == 0.0
    @test SpreadFromOHLCV(1.0).lambda == 1.0
    @test SpreadFromOHLCV(0.7).lambda == 0.7
end

@testset "SpreadFromOHLCV(0.0): widest spread, bid=low, ask=high" begin
    q = synthesize(SpreadFromOHLCV(0.0), _mk_bar(high=1.20, low=0.80, close=1.00))
    @test q.bid == 0.80
    @test q.ask == 1.20
    @test q.mark == 1.00
end

@testset "SpreadFromOHLCV(1.0): midpoint, bid=ask=close" begin
    q = synthesize(SpreadFromOHLCV(1.0), _mk_bar(high=1.20, low=0.80, close=1.00))
    @test q.bid == 1.00
    @test q.ask == 1.00
    @test q.mark == 1.00
end

@testset "SpreadFromOHLCV(0.7): canonical interpolation matches formula" begin
    q = synthesize(SpreadFromOHLCV(0.7), _mk_bar(high=1.20, low=0.80, close=1.00))
    # bid = 0.80 + 0.7*(1.00-0.80) = 0.94 ; ask = 1.20 - 0.7*(1.20-1.00) = 1.06
    @test q.bid ≈ 0.94
    @test q.ask ≈ 1.06
    @test q.mark == 1.00
end

@testset "SpreadFromOHLCV: mark = close (always, when close present)" begin
    for λ in (0.0, 0.3, 0.7, 1.0)
        q = synthesize(SpreadFromOHLCV(λ), _mk_bar(close=2.50))
        @test q.mark == 2.50
    end
end

@testset "SpreadFromOHLCV: missing high -> bid/ask missing, mark still close" begin
    bar = OptionBar("X", _SYNTH_U, _SYNTH_EXP, 406.0, Call,
                    1.00, missing, 0.80, 1.00, 10.0, _SYNTH_TS)
    q = synthesize(SpreadFromOHLCV(0.7), bar)
    @test ismissing(q.bid)
    @test ismissing(q.ask)
    @test q.mark == 1.00
end

@testset "SpreadFromOHLCV: missing low -> bid/ask missing, mark still close" begin
    bar = OptionBar("X", _SYNTH_U, _SYNTH_EXP, 406.0, Call,
                    1.00, 1.20, missing, 1.00, 10.0, _SYNTH_TS)
    q = synthesize(SpreadFromOHLCV(0.7), bar)
    @test ismissing(q.bid)
    @test ismissing(q.ask)
    @test q.mark == 1.00
end

@testset "SpreadFromOHLCV: missing close -> bid/ask/mark all missing" begin
    bar = OptionBar("X", _SYNTH_U, _SYNTH_EXP, 406.0, Call,
                    1.00, 1.20, 0.80, missing, 10.0, _SYNTH_TS)
    q = synthesize(SpreadFromOHLCV(0.7), bar)
    @test ismissing(q.bid)
    @test ismissing(q.ask)
    @test ismissing(q.mark)
end

@testset "synthesize: contract identity + timestamp + volume pass through" begin
    bar = _mk_bar(volume=42.0)
    q = synthesize(SpreadFromOHLCV(0.7), bar)
    @test q.instrument_id == bar.instrument_id
    @test q.underlying === bar.underlying
    @test q.expiry == bar.expiry
    @test q.strike == bar.strike
    @test q.option_type == bar.option_type
    @test q.timestamp == bar.timestamp
    @test q.volume == 42.0
    @test ismissing(q.iv)
    @test ismissing(q.open_interest)
end
