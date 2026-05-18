const _SPY = Underlying("SPY")
const _EXP = DateTime(2024, 3, 15, 21, 0)

@testset "Trade: construction and validation" begin
    t = Trade(_SPY, 100.0, _EXP, Call)
    @test t.direction == 1
    @test t.quantity == 1.0
    @test t.option_type == Call

    short = Trade(_SPY, 100.0, _EXP, Put; direction=-1, quantity=2.0)
    @test short.direction == -1
    @test short.quantity == 2.0

    int_inputs = Trade(_SPY, 100, _EXP, Call; quantity=1)
    @test int_inputs.strike === 100.0
    @test int_inputs.quantity === 1.0

    @test_throws ArgumentError Trade(_SPY, 100.0, _EXP, Call; direction=0)
    @test_throws ArgumentError Trade(_SPY, 100.0, _EXP, Call; direction=2)
    @test_throws ArgumentError Trade(_SPY, 100.0, _EXP, Call; quantity=0.0)
    @test_throws ArgumentError Trade(_SPY, 100.0, _EXP, Call; quantity=-1.0)
    @test_throws ArgumentError Trade(_SPY, -10.0, _EXP, Call)
end

@testset "Trade: payoff -- long call" begin
    t = Trade(_SPY, 100.0, _EXP, Call)
    @test payoff(t, 90.0)  == 0.0
    @test payoff(t, 100.0) == 0.0
    @test payoff(t, 110.0) == 10.0
end

@testset "Trade: payoff -- short call" begin
    t = Trade(_SPY, 100.0, _EXP, Call; direction=-1)
    @test payoff(t, 90.0)  == 0.0
    @test payoff(t, 110.0) == -10.0
end

@testset "Trade: payoff -- long put" begin
    t = Trade(_SPY, 100.0, _EXP, Put)
    @test payoff(t, 110.0) == 0.0
    @test payoff(t, 100.0) == 0.0
    @test payoff(t,  90.0) == 10.0
end

@testset "Trade: payoff -- short put" begin
    t = Trade(_SPY, 100.0, _EXP, Put; direction=-1)
    @test payoff(t, 110.0) == 0.0
    @test payoff(t,  90.0) == -10.0
end

@testset "Trade: payoff scales with quantity" begin
    t = Trade(_SPY, 100.0, _EXP, Call; quantity=3.5)
    @test payoff(t, 110.0) == 35.0
end
