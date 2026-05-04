@testset "Underlying" begin
    @test ticker(Underlying("spy")) == "SPY"
    @test ticker(Underlying("BTC")) == "BTC"
    @test sprint(show, Underlying("SPY")) == "SPY"
end

@testset "OptionQuote" begin
    q = OptionQuote(
        "O:SPY240129C00406000",
        Underlying("SPY"),
        DateTime(2024, 1, 29, 21, 0),
        406.0,
        Call,
        1.20, 1.25, 1.225, 18.5, 1234.0, 567.0,
        DateTime(2024, 1, 15, 15, 30),
    )
    @test q.strike == 406.0
    @test q.option_type == Call
    @test q.iv == 18.5
    @test q.bid == 1.20
end

@testset "OptionQuote with missings" begin
    q = OptionQuote(
        "X", Underlying("SPY"), DateTime(2024, 1, 29), 100.0, Put,
        missing, missing, missing, missing, missing, missing,
        DateTime(2024, 1, 15),
    )
    @test ismissing(q.bid)
    @test ismissing(q.ask)
    @test ismissing(q.iv)
    @test q.option_type == Put
end

@testset "SpotPrice" begin
    s = SpotPrice(Underlying("SPY"), 480.5, DateTime(2024, 1, 15, 15, 30))
    @test s.price == 480.5
    @test ticker(s.underlying) == "SPY"
end
