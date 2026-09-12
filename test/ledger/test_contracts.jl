# Case 10: the contract table.

@testset "contracts: the table resolves the listed ETFs" begin
    spec = contract_spec(Underlying("SPY"))
    @test spec.multiplier == 100.0
    @test spec.exercise == American
    @test spec.settlement == PMSettled
    @test spec.delivery == Physical
    @test contract_spec(Underlying("qqq")) == spec        # Underlying uppercases
    @test contract_spec(Underlying("IWM")) == spec
end

@testset "contracts: an unknown ticker throws UnknownContract" begin
    @test_throws UnknownContract contract_spec(Underlying("SPX"))
    err = try contract_spec(Underlying("XYZ")); nothing catch e; e end
    @test err isa UnknownContract
    @test err.underlying == Underlying("XYZ")
    @test occursin("XYZ", sprint(showerror, err))
end
