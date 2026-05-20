# Tests for the Agent abstraction and the StaticAgent base case.

const _AG_UND = Underlying("SPY")

function _ag_fixture()
    ts1 = DateTime(2024, 1, 15, 15, 30)
    spot = 480.0; r = 0.04; q = 0.015
    expiry = DateTime(2024, 2, 16, 21, 0)
    qte = OptionQuote("X", _AG_UND, expiry, 480.0, Call,
                      missing, missing, 5.0, missing, missing, missing, ts1)
    inner = InMemoryDataSource(_AG_UND;
        chains=Dict(ts1 => [qte]),
        spots=Dict(ts1 => spot))
    mds = ModelDataSource(inner; rate=FlatCurve(r), div=FlatCurve(q))
    (mds=mds, ts1=ts1)
end

@testset "StaticAgent: returns its wrapped policy on every call" begin
    f = _ag_fixture()
    cut = TimeCutModelDataSource(f.mds, f.ts1)
    p = NoOpPolicy()
    a = StaticAgent(p)
    @test current_policy(a, f.ts1, cut, Position[]) === p
end

# A custom Agent without a current_policy method must fall through to the
# error stub on the abstract type.
struct _UnimplementedAgent <: Agent end

@testset "Agent: missing current_policy method errors" begin
    f = _ag_fixture()
    cut = TimeCutModelDataSource(f.mds, f.ts1)
    @test_throws ErrorException current_policy(_UnimplementedAgent(), f.ts1, cut, Position[])
end
