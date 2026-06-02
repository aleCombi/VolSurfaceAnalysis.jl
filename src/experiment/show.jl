# Pretty-printing for `ExperimentResult`. The Julia convention is
# `show(::IO, ::MIME"text/plain", x)`: hooking the canonical render
# point makes the REPL, `display`, and `print(stdout, ...)` all do the
# right thing without a separate verb.

function Base.show(io::IO, ::MIME"text/plain", r::ExperimentResult)
    e = r.experiment
    s = r.pnl_series
    nfills = s.n_opens + s.n_closes
    nrts   = length(s.pnl)
    println(io, "ExperimentResult: ", e.name)
    println(io, "  window      ", e.from, "  to  ", e.to)
    println(io, "  source      ", typeof(e.source))
    println(io, "  agent       ", typeof(e.agent))
    println(io, "  positions   ", length(r.positions),
                "  (fills: ", nfills,
                ", round trips: ", nrts,
                ", unmarked: ", s.n_unmarked, ")")
    println(io)
    println(io, "Metrics:")
    width = isempty(keys(r.metrics)) ? 0 : maximum(length(string(k)) for k in keys(r.metrics))
    for k in keys(r.metrics)
        println(io, "  ", rpad(string(k), width), "  ", r.metrics[k])
    end
    if nrts > 0 || s.n_unmarked > 0
        println(io)
        println(io, "Window-end spot: ", s.window_end_spot)
    end
end
