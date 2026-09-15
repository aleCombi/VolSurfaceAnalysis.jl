# Pretty-printing for `ExperimentResult`. The Julia convention is
# `show(::IO, ::MIME"text/plain", x)`: hooking the canonical render
# point makes the REPL, `display`, and `print(stdout, ...)` all do the
# right thing without a separate verb.

# Whole cents as a USD amount with two decimals, sign first.
function _usd(cents::Integer)::String
    sign = cents < 0 ? "-" : ""
    a = abs(cents)
    return string(sign, a ÷ 100, ".", lpad(a % 100, 2, '0'))
end

function Base.show(io::IO, ::MIME"text/plain", r::ExperimentResult)
    e = r.experiment
    L = r.ledger
    c = r.curve
    kinds = ((Fill, "fills"), (Match, "matches"), (Expiry, "expiries"), (Fee, "fees"))
    per_kind = join(("$(count(x -> x isa T, L.events)) $name" for (T, name) in kinds), ", ")
    book = book_as_known(L, last_sequence(L))
    println(io, "ExperimentResult: ", e.name)
    println(io, "  window      ", e.from, "  to  ", e.to)
    println(io, "  data        ", join((kind_name(kind(p)) for p in e.data.entries), ", "))
    println(io, "  clock       ", kind_name(kind(e.clock)), " / ", e.clock.sel)
    println(io, "  agent       ", typeof(e.agent))
    println(io, "  events      ", length(L), "  (", per_kind, ")")
    println(io, "  orders      ", length(L.orders),
                "  (round trips: ", length(trade_pnl(L)), ")")
    println(io, "  book        ", length(open_lots(book)), " open lots in ",
                length(open_groups(book)), " open groups, cash USD ", _usd(book.cash))
    if c === nothing
        println(io, "  curve       not built (market data unavailable)")
    else
        println(io, "  curve       ", n_marked(c), " marked sessions, ",
                    n_unmarked(c), " unmarked")
    end
    # Zero is an answer: the run asked and nothing went unanswered. It is
    # not the same as a result that never carried the question, which is
    # why the line is always printed.
    println(io, "  failures    ", length(r.failures), " retained",
                isempty(r.failures) ? "" :
                "  (" * join(sort(unique(string(f.stage, ":", f.reason) for f in r.failures)), ", ") * ")")
    println(io)
    println(io, "Metrics:")
    width = isempty(keys(r.metrics)) ? 0 : maximum(length(string(k)) for k in keys(r.metrics))
    for k in keys(r.metrics)
        println(io, "  ", rpad(string(k), width), "  ", r.metrics[k])
    end
end
