# Tests for the Policy abstraction and the NoOpPolicy base case.

const _PL_UND = Underlying("SPY")

function _pl_fixture()
    ts1 = DateTime(2024, 1, 15, 15, 30)
    spot = 480.0; r = 0.04; q = 0.015
    expiry = DateTime(2024, 2, 16, 21, 0)
    T = time_to_expiry(expiry, ts1)
    mark = bs_price(spot, 480.0, T, 0.20, Call; r=r, q=q)
    qte = OptionQuote("X", _PL_UND, expiry, 480.0, Call,
                      missing, missing, mark, missing, missing, missing, ts1)
    inner = InMemoryDataSource(_PL_UND;
        chains=Dict(ts1 => [qte]),
        spots=Dict(ts1 => spot))
    mds = ModelDataSource(inner; rate=FlatCurve(r), div=FlatCurve(q))
    (mds=mds, ts1=ts1)
end

@testset "NoOpPolicy: decide returns empty" begin
    f = _pl_fixture()
    cut = TimeCutModelDataSource(f.mds, f.ts1)
    @test decide(NoOpPolicy(), f.ts1, cut, Position[]) == Trade[]
end

# A custom Policy without a decide method must fall through to the
# error stub on the abstract type.
struct _UnimplementedPolicy <: Policy end

@testset "Policy: missing decide method errors" begin
    f = _pl_fixture()
    cut = TimeCutModelDataSource(f.mds, f.ts1)
    @test_throws ErrorException decide(_UnimplementedPolicy(), f.ts1, cut, Position[])
end
