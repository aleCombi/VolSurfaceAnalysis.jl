# Lifecycle: open/close order, unwind on open failure, best-effort close,
# with_data, and the loader's has_lifecycle check. Uses test-local specs
# that log their open/close events into a shared vector.

struct _MD_TrackSpec
    id::Symbol
    log::Vector{Tuple{Symbol,Symbol}}
end
struct _MD_TrackReader
    id::Symbol
    log::Vector{Tuple{Symbol,Symbol}}
end
VolSurfaceAnalysis.kind(::_MD_TrackSpec) = SpotPrice
VolSurfaceAnalysis.kind(::_MD_TrackReader) = SpotPrice
function VolSurfaceAnalysis.open_data(s::_MD_TrackSpec)
    push!(s.log, (:open, s.id))
    _MD_TrackReader(s.id, s.log)
end
function VolSurfaceAnalysis.close_data!(r::_MD_TrackReader)
    push!(r.log, (:close, r.id))
    nothing
end

struct _MD_FailOpenSpec end
VolSurfaceAnalysis.kind(::_MD_FailOpenSpec) = VolatilitySurface   # distinct from the other test kinds
VolSurfaceAnalysis.open_data(::_MD_FailOpenSpec) = error("open failed")

struct _MD_FailCloseSpec
    log::Vector{Tuple{Symbol,Symbol}}
end
struct _MD_FailCloseReader
    log::Vector{Tuple{Symbol,Symbol}}
end
VolSurfaceAnalysis.kind(::_MD_FailCloseSpec) = OptionBar
VolSurfaceAnalysis.kind(::_MD_FailCloseReader) = OptionBar
VolSurfaceAnalysis.open_data(s::_MD_FailCloseSpec) = (push!(s.log, (:open, :F)); _MD_FailCloseReader(s.log))
VolSurfaceAnalysis.close_data!(r::_MD_FailCloseReader) = (push!(r.log, (:close, :F)); error("close failed"))

struct _MD_NoLifecycleSpec end
VolSurfaceAnalysis.kind(::_MD_NoLifecycleSpec) = OptionQuote

# Three tracked specs need three distinct kinds inside one map; give the
# tracked spec a kind parameter for that.
struct _MD_TrackSpecK{R}
    id::Symbol
    log::Vector{Tuple{Symbol,Symbol}}
end
struct _MD_TrackReaderK{R}
    id::Symbol
    log::Vector{Tuple{Symbol,Symbol}}
end
VolSurfaceAnalysis.kind(::_MD_TrackSpecK{R}) where {R} = R
VolSurfaceAnalysis.kind(::_MD_TrackReaderK{R}) where {R} = R
function VolSurfaceAnalysis.open_data(s::_MD_TrackSpecK{R}) where {R}
    push!(s.log, (:open, s.id))
    _MD_TrackReaderK{R}(s.id, s.log)
end
VolSurfaceAnalysis.close_data!(r::_MD_TrackReaderK) = (push!(r.log, (:close, r.id)); nothing)

_md_log() = Tuple{Symbol,Symbol}[]

@testset "lifecycle: resource-free specs are their own readers" begin
    for s in (InMemory(_md_spots()), Constant(SpotPrice(_MD_SPY, 1.0, typemin(DateTime))),
              QuotesFromBars(SpreadFromOHLCV(0.7)))
        @test open_data(s) === s
        @test close_data!(s) === nothing
        @test VolSurfaceAnalysis.has_lifecycle(s)
    end
    @test !VolSurfaceAnalysis.has_lifecycle(_MD_NoLifecycleSpec())
    @test_throws MethodError open_data(_MD_NoLifecycleSpec())
end

@testset "lifecycle: map opens in order, closes in reverse" begin
    log = _md_log()
    m = MarketData(_MD_TrackSpecK{SpotPrice}(:A, log), _MD_TrackSpecK{OptionBar}(:B, log))
    d = open_data(m)
    @test d isa MarketData
    @test d.entries[1] isa _MD_TrackReaderK{SpotPrice}
    @test log == [(:open, :A), (:open, :B)]
    close_data!(d)
    @test log == [(:open, :A), (:open, :B), (:close, :B), (:close, :A)]
end

@testset "lifecycle: open failure unwinds what was opened" begin
    log = _md_log()
    m = MarketData(_MD_TrackSpecK{SpotPrice}(:A, log), _MD_TrackSpecK{OptionQuote}(:B, log), _MD_FailOpenSpec())
    err = try open_data(m); nothing catch e; e end
    @test err isa ErrorException && err.msg == "open failed"
    @test log == [(:open, :A), (:open, :B), (:close, :B), (:close, :A)]
end

@testset "lifecycle: unwind close error does not mask the open error" begin
    log = _md_log()
    m = MarketData(_MD_TrackSpecK{SpotPrice}(:A, log), _MD_FailCloseSpec(log), _MD_FailOpenSpec())
    err = @test_logs (:warn, r"close_data! failed during unwind") begin
        try open_data(m); nothing catch e; e end
    end
    @test err isa ErrorException && err.msg == "open failed"
    @test log == [(:open, :A), (:open, :F), (:close, :F), (:close, :A)]
end

@testset "lifecycle: close is best-effort, first error rethrown" begin
    log = _md_log()
    m = MarketData(_MD_TrackSpecK{SpotPrice}(:A, log), _MD_FailCloseSpec(log), _MD_TrackSpecK{OptionQuote}(:C, log))
    d = open_data(m)
    err = try close_data!(d); nothing catch e; e end
    @test err isa ErrorException && err.msg == "close failed"
    @test log == [(:open, :A), (:open, :F), (:open, :C), (:close, :C), (:close, :F), (:close, :A)]
end

@testset "lifecycle: with_data closes on success and on failure" begin
    log = _md_log()
    m = MarketData(_MD_TrackSpecK{SpotPrice}(:A, log))
    r = with_data(d -> (@test d.entries[1] isa _MD_TrackReaderK; 42), m)
    @test r == 42
    @test log == [(:open, :A), (:close, :A)]

    empty!(log)
    err = try with_data(d -> error("f failed"), m); nothing catch e; e end
    @test err isa ErrorException && err.msg == "f failed"
    @test log == [(:open, :A), (:close, :A)]

    # f's error wins over a close error
    log2 = _md_log()
    m2 = MarketData(_MD_FailCloseSpec(log2))
    err = @test_logs (:warn, r"close_data! failed during unwind") begin
        try with_data(d -> error("f failed"), m2); nothing catch e; e end
    end
    @test err isa ErrorException && err.msg == "f failed"
    @test log2 == [(:open, :F), (:close, :F)]
    # on success a close error propagates
    @test_throws ErrorException with_data(d -> 1, m2)
end

@testset "lifecycle: BySelector opens and closes its parts in order" begin
    log = _md_log()
    b = BySelector{SpotPrice}(_MD_SPY => _MD_TrackSpec(:SPY, log), _MD_SPX => _MD_TrackSpec(:SPX, log))
    r = open_data(b)
    @test r isa BySelector{SpotPrice}
    @test last(r.parts[1]) isa _MD_TrackReader && first(r.parts[1]) === _MD_SPY
    @test log == [(:open, :SPY), (:open, :SPX)]
    close_data!(r)
    @test log == [(:open, :SPY), (:open, :SPX), (:close, :SPX), (:close, :SPY)]

    # inside a map, the unwind reaches into the composite
    empty!(log)
    m = MarketData(b, _MD_FailOpenSpec())
    @test_throws ErrorException open_data(m)
    @test log == [(:open, :SPY), (:open, :SPX), (:close, :SPX), (:close, :SPY)]
end

@testset "lifecycle: open_data on a map is type-stable" begin
    m = MarketData(InMemory(_md_bars()), QuotesFromBars(SpreadFromOHLCV(0.7)), InMemory(_md_spots()))
    @test @inferred(open_data(m)) isa typeof(m)
    @test Base.return_types(open_data, (typeof(m),)) == [typeof(m)]
end
