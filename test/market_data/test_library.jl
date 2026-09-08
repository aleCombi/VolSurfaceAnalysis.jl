# Library functions over protocol results: only_or_missing, by_timestamp.

# An iterator that counts how many records it has handed out, to show
# by_timestamp is lazy (it reads one group plus one record ahead).
mutable struct _MD_Counting{V}
    rows::V
    handed::Int
end
_MD_Counting(rows) = _MD_Counting(rows, 0)
Base.IteratorSize(::Type{<:_MD_Counting}) = Base.SizeUnknown()
Base.eltype(::Type{_MD_Counting{V}}) where {V} = eltype(V)
function Base.iterate(c::_MD_Counting, i=1)
    i > length(c.rows) && return nothing
    c.handed += 1
    (c.rows[i], i + 1)
end

@testset "only_or_missing" begin
    s = _md_spot(_MD_SPY, _MD_T1, 481.0)
    @test only_or_missing(SpotPrice[]) === missing
    @test only_or_missing([s]) === s
    @test_throws ArgumentError only_or_missing([s, s])
end

@testset "by_timestamp: groups a sorted mixed vector" begin
    rows = sort(_md_bars(); by = r -> r.timestamp)      # stable: SPY before SPX within a tick
    groups = collect(by_timestamp(rows))
    @test length(groups) == 3
    @test first.(groups) == [_MD_T1, _MD_T2, _MD_T3]
    for (ts, g) in groups
        @test g isa Vector{OptionBar}
        @test length(g) == 4
        @test all(r -> r.timestamp == ts, g)
    end
    @test groups[1][2] == rows[1:4]
    @test eltype(by_timestamp(rows)) == Tuple{DateTime,Vector{OptionBar}}
end

@testset "by_timestamp: lazy, one group ahead" begin
    rows = sort(_md_bars(); by = r -> r.timestamp)
    c = _MD_Counting(rows)
    first_group = first(Iterators.take(by_timestamp(c), 1))
    @test first_group[1] == _MD_T1
    @test length(first_group[2]) == 4
    @test c.handed == 5                                 # the group plus the record that closed it
end

@testset "by_timestamp: generator input gets a concrete group type" begin
    rows = sort(_md_bars(); by = r -> r.timestamp)
    it = Iterators.map(identity, rows)                  # eltype Any, like a lazy between
    g = first(by_timestamp(it))
    @test g[2] isa Vector{OptionBar}
end

@testset "by_timestamp: empty input, unsorted input" begin
    @test collect(by_timestamp(OptionBar[])) == []
    @test iterate(by_timestamp(SpotPrice[])) === nothing
    bad = [_md_spot(_MD_SPY, _MD_T2, 1.0), _md_spot(_MD_SPY, _MD_T1, 2.0)]
    @test_throws ArgumentError collect(by_timestamp(bad))
end
