# The MarkedCurve value type, the cents boundary and the two derived
# views. The builder is exercised in test_marks.jl; here the struct, its
# refusals and the unmarked-point policy in isolation.

_mc_ts(n; start=DateTime(2024, 1, 16, 21, 0)) = [start + Day(i - 1) for i in 1:n]

_mc(profit::Vector{Float64}; start=DateTime(2024, 1, 16, 21, 0),
    unmarked=DateTime[], reasons=Symbol[:no_mark for _ in unmarked]) =
    MarkedCurve(_mc_ts(length(profit); start=start), profit, unmarked, reasons)

@testset "cents_to_usd: the one boundary out of whole cents" begin
    @test cents_to_usd(0) === 0.0
    @test cents_to_usd(3_200_866) ≈ 32008.66      # the ten-year strangle's cash
    @test cents_to_usd(-4500) ≈ -45.0
    @test cents_to_usd(1) ≈ 0.01
    # Exactly the inverse of the ledger's one rounding point for a price
    # that is whole cents per contract.
    spec = ContractSpec(100, American, PMSettled, Physical)
    @test cents_to_usd(VolSurfaceAnalysis.contract_cents(0.85, spec)) ≈ 85.0
end

@testset "MarkedCurve: parallel vectors, counts and no sentinel" begin
    ts = _mc_ts(3)
    c = MarkedCurve(ts, [0.0, 10.0, -5.0], DateTime[], Symbol[])
    @test c.timestamps == ts
    @test c.profit == [0.0, 10.0, -5.0]
    @test n_marked(c) == 3
    @test n_unmarked(c) == 0
    @test !ismutable(c)
    # An unmarked session appears in the second pair only: no NaN, no
    # placeholder value, nothing to mistake for a number.
    u = DateTime(2024, 1, 20, 21, 0)
    c2 = MarkedCurve(ts, [0.0, 10.0, -5.0], [u], [:no_mark])
    @test n_marked(c2) == 3 && n_unmarked(c2) == 1
    @test c2.unmarked_reason == [:no_mark]
    @test !any(isnan, c2.profit)
    empty = MarkedCurve(DateTime[], Float64[], DateTime[], Symbol[])
    @test n_marked(empty) == 0 && n_unmarked(empty) == 0
end

@testset "MarkedCurve: mismatched or unsorted vectors are refused by name" begin
    ts = _mc_ts(3)
    for (name, bad) in (
        ("timestamps vs profit",
         () -> MarkedCurve(ts, [1.0, 2.0], DateTime[], Symbol[])),
        ("unmarked vs reasons",
         () -> MarkedCurve(ts, [1.0, 2.0, 3.0], [ts[1] + Day(9)], Symbol[])),
        ("timestamps descending",
         () -> MarkedCurve(reverse(ts), [1.0, 2.0, 3.0], DateTime[], Symbol[])),
        ("unmarked descending",
         () -> MarkedCurve(ts, [1.0, 2.0, 3.0], reverse(_mc_ts(2; start=ts[1] + Day(9))),
                           [:no_mark, :no_mark])),
    )
        err = try; bad(); nothing; catch e; e; end
        @test err isa ArgumentError
        @test occursin("MarkedCurve", err.msg)
        # The refusal builds nothing: a curve that fails construction cannot
        # masquerade as a complete one.
        @test !(err isa MarkedCurve)
        println("  refused ($name): ", err.msg)
    end
end

@testset "session_changes: one observation per adjacent marked pair" begin
    c = _mc([0.0, 10.0, 4.0, 9.0])
    @test session_changes(c) ≈ [10.0, -6.0, 5.0]
    @test length(session_changes(c)) == n_marked(c) - 1
    @test isempty(session_changes(_mc([1.0])))
    @test isempty(session_changes(MarkedCurve(DateTime[], Float64[], DateTime[], Symbol[])))
end

@testset "session_changes: an unmarked session breaks the curve, never spans it" begin
    # Sessions 1, 2, [3 unmarked], 4: the 2 -> 4 step covers two periods and
    # is not one observation, so it is dropped rather than scaled.
    ts = _mc_ts(4)
    marked = [ts[1], ts[2], ts[4]]
    c = MarkedCurve(marked, [0.0, 10.0, 30.0], [ts[3]], [:no_mark])
    @test session_changes(c) ≈ [10.0]
    @test n_unmarked(c) == 1
    # Every other session unmarked leaves no observation at all, which is
    # the honest answer rather than a series of double-length steps.
    ts6 = _mc_ts(6)
    c2 = MarkedCurve([ts6[1], ts6[3], ts6[5]], [0.0, 1.0, 2.0],
                     [ts6[2], ts6[4]], [:no_mark, :unexpected_gap])
    @test isempty(session_changes(c2))
end
