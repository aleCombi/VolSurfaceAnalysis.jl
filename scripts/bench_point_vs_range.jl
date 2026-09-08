using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using VolSurfaceAnalysis
using Dates

# Point vs range reads on one month of minute option bars.
#
#   julia --project=. scripts/bench_point_vs_range.jl [root=~/data/massive] [symbol=SPY] [month=2024-01]
#
# Opens ParquetOptionBars once, enumerates the month's timestamps, then:
#   A  point:    `at` at every timestamp (chain LRU of 10, cold)
#   B  range:    one `between` over the month, grouped with `by_timestamp`
#   C  strangle: per day, `at` at 19:30 vs `between` over [19:30, 19:30]
# Reports wall time, allocations, the Sys.maxrss() delta, and asserts A
# and B see the same number of records. One warm-up day runs before
# timing so compilation is not in the numbers.

root   = length(ARGS) >= 1 ? ARGS[1] : joinpath(homedir(), "data", "massive")
symbol = length(ARGS) >= 2 ? ARGS[2] : "SPY"
month  = length(ARGS) >= 3 ? ARGS[3] : "2024-01"

u     = Underlying(symbol)
first_day = Date(month * "-01")
last_day  = Dates.lastdayofmonth(first_day)
from  = DateTime(first_day)
to    = DateTime(last_day, Time(23, 59, 59))
entry = Time(19, 30)

mb(x) = round(x / 2^20; digits = 1)
ms(x) = round(x * 1000; digits = 1)

r = open_data(ParquetOptionBars(joinpath(root, "options_1min")); max_chains = 10)
try
    ts = timestamps(r, nothing, OptionBar, u, from, to)
    days = unique!(Date.(ts))
    println("$symbol $month: $(length(ts)) timestamps over $(length(days)) days, root = $root")
    isempty(ts) && error("no timestamps in the month")

    # warm-up on the first day (compiles both paths)
    let d = first(days), t0 = DateTime(d), t1 = DateTime(d, Time(23, 59, 59))
        foreach(t -> at(r, nothing, OptionBar, u, t), filter(t -> Date(t) == d, ts))
        sum(length(c) for (_, c) in by_timestamp(between(r, nothing, OptionBar, u, t0, t1)))
        empty!(r.chains)
    end

    # A: point reads at every timestamp
    GC.gc(); rss0 = Sys.maxrss()
    n_a = 0
    t_a = @elapsed alloc_a = @allocated begin
        for t in ts
            n_a += length(at(r, nothing, OptionBar, u, t))
        end
    end
    rss_a = Sys.maxrss() - rss0
    empty!(r.chains)

    # B: one range read, grouped by timestamp
    GC.gc(); rss0 = Sys.maxrss()
    n_b = 0; groups_b = 0
    t_b = @elapsed alloc_b = @allocated begin
        for (_, chain) in by_timestamp(between(r, nothing, OptionBar, u, from, to))
            n_b += length(chain); groups_b += 1
        end
    end
    rss_b = Sys.maxrss() - rss0

    # C: the strangle workload, one instant per day
    entry_ts = [DateTime(d, entry) for d in days]
    empty!(r.chains); GC.gc()
    n_c1 = 0
    t_c1 = @elapsed alloc_c1 = @allocated begin
        for t in entry_ts
            n_c1 += length(at(r, nothing, OptionBar, u, t))
        end
    end
    empty!(r.chains); GC.gc()
    n_c2 = 0
    t_c2 = @elapsed alloc_c2 = @allocated begin
        for t in entry_ts
            n_c2 += length(collect(between(r, nothing, OptionBar, u, t, t)))
        end
    end

    println()
    println("A point   (at, every timestamp):      $(ms(t_a)) ms, $(mb(alloc_a)) MB alloc, maxrss +$(mb(rss_a)) MB, $n_a records")
    println("B range   (between + by_timestamp):   $(ms(t_b)) ms, $(mb(alloc_b)) MB alloc, maxrss +$(mb(rss_b)) MB, $n_b records in $groups_b groups")
    println("C1 strangle (at @ $entry per day):     $(ms(t_c1)) ms, $(mb(alloc_c1)) MB alloc, $n_c1 records")
    println("C2 strangle (between [t, t] per day): $(ms(t_c2)) ms, $(mb(alloc_c2)) MB alloc, $n_c2 records")
    println("ratio A/B wall: $(round(t_a / t_b; digits = 2))x; per-timestamp A: $(ms(t_a / length(ts))) ms, B: $(ms(t_b / length(ts))) ms")
    println("peak maxrss: $(mb(Sys.maxrss())) MB")
    n_a == n_b || error("record counts differ: A = $n_a, B = $n_b")
    groups_b == length(ts) || error("group count $groups_b != timestamps $(length(ts))")
    n_c1 == n_c2 || error("strangle counts differ: at = $n_c1, between = $n_c2")
    println("counts agree")
finally
    close_data!(r)
end
