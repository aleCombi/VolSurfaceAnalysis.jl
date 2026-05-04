_quote(strike, t) = OptionQuote(
    "X", Underlying("SPY"), DateTime(2024, 1, 29), strike, Call,
    1.0, 1.1, 1.05, 20.0, 100.0, 50.0, t,
)

@testset "InMemoryDataSource: dict constructor" begin
    t1 = DateTime(2024, 1, 15, 15, 30)
    t2 = DateTime(2024, 1, 15, 15, 31)
    chains = Dict(
        t2 => [_quote(400.0, t2)],
        t1 => [_quote(400.0, t1), _quote(405.0, t1)],
    )
    spots = Dict(t1 => 480.0, t2 => 480.5)
    src = InMemoryDataSource("SPY"; chains=chains, spots=spots)

    @test ticker(src.underlying) == "SPY"
    @test available_timestamps(src) == [t1, t2]
    @test length(get_chain(src, t1)) == 2
    @test length(get_chain(src, t2)) == 1
    @test get_chain(src, DateTime(2024, 1, 15, 15, 32)) === nothing
    @test get_spot(src, t1) == 480.0
    @test get_spot(src, t2) == 480.5
    @test ismissing(get_spot(src, DateTime(2024, 1, 15, 15, 32)))
end

@testset "InMemoryDataSource: get_spots range" begin
    ts = [DateTime(2024, 1, 15, 9, 30) + Minute(i) for i in 0:9]
    spots = Dict(ts[i] => 480.0 + i for i in 1:10)
    src = InMemoryDataSource("SPY"; spots=spots)

    out = get_spots(src, ts[3], ts[7])
    @test length(out) == 5
    @test out[1].timestamp == ts[3]
    @test out[end].timestamp == ts[7]
    @test out[1].price == 483.0
    @test ticker(out[1].underlying) == "SPY"

    @test isempty(get_spots(src, ts[1] - Hour(1), ts[1] - Minute(1)))
    @test length(get_spots(src, ts[1], ts[end])) == 10
end

@testset "InMemoryDataSource: empty" begin
    src = InMemoryDataSource("SPY")
    @test isempty(get_spots(src, DateTime(2024, 1, 1), DateTime(2024, 1, 2)))
    @test isempty(available_timestamps(src))
    @test get_chain(src, DateTime(2024, 1, 1)) === nothing
    @test ismissing(get_spot(src, DateTime(2024, 1, 1)))
end

@testset "InMemoryDataSource: validation" begin
    t1 = DateTime(2024, 1, 15, 15, 30)
    t2 = DateTime(2024, 1, 15, 15, 31)

    @test_throws ArgumentError InMemoryDataSource(
        Underlying("SPY"), [t2, t1], [OptionQuote[], OptionQuote[]],
        DateTime[], Float64[],
    )

    @test_throws ArgumentError InMemoryDataSource(
        Underlying("SPY"), [t1], [OptionQuote[], OptionQuote[]],
        DateTime[], Float64[],
    )

    @test_throws ArgumentError InMemoryDataSource(
        Underlying("SPY"), [t1, t1], [OptionQuote[], OptionQuote[]],
        DateTime[], Float64[],
    )
end
