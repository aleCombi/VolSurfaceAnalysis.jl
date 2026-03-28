using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis, Dates, Printf, Statistics
using ChenSignatures: logsig, prepare

store = DEFAULT_STORE

# =============================================================================
# Stage timings for a single date
# =============================================================================

function bench_single(d::Date; entry_time=Time(10, 0))
    open_utc = et_to_utc(d, Time(9, 30))
    entry_utc = et_to_utc(d, entry_time)

    # Spots
    t0 = time()
    sp = polygon_spot_path(store, d, "SPY")
    all_spots = read_polygon_spot_prices(sp; underlying="SPY")
    spots = Dict(k => v for (k, v) in all_spots if open_utc <= k <= entry_utc)
    t_spots = time() - t0

    # Option bars
    t0 = time()
    path = polygon_options_path(store, d, "SPY")
    open_str = Dates.format(open_utc, "yyyy-mm-dd HH:MM:SS")
    close_str = Dates.format(et_to_utc(d, Time(16, 0)), "yyyy-mm-dd HH:MM:SS")
    bars = read_polygon_parquet(path;
        where="timestamp BETWEEN '$open_str' AND '$close_str'")
    t_bars = time() - t0

    # Group + ATM IV
    by_ts = Dict{DateTime, Vector{PolygonBar}}()
    for bar in bars
        push!(get!(by_ts, bar.timestamp, PolygonBar[]), bar)
    end
    cache = BarCache(store)
    cache.data[(d, "SPY")] = by_ts

    t0 = time()
    ivs = VolSurfaceAnalysis._bulk_atm_iv(cache, "SPY", d, spots, open_utc, entry_utc;
        rate=0.045, div_yield=0.013, spot_hint=first(values(spots)))
    t_iv = time() - t0

    # Path construction
    common_ts = sort(intersect(collect(keys(spots)), collect(keys(ivs))))
    sp_vec = [spots[t] for t in common_ts]
    iv_vec = [ivs[t] for t in common_ts]
    n = length(common_ts)

    path_mat = Matrix{Float64}(undef, n, 3)
    for i in 1:n
        path_mat[i, 1] = (i - 1) / (n - 1)
        path_mat[i, 2] = log(sp_vec[i] / sp_vec[1])
        path_mat[i, 3] = iv_vec[i] - iv_vec[1]
    end

    # LogSig
    basis = prepare(3, 3)
    logsig(path_mat, basis)  # warmup
    t0 = time()
    for _ in 1:1000
        logsig(path_mat, basis)
    end
    t_logsig = (time() - t0) / 1000

    return (n_spots=length(spots), n_bars=length(bars), n_ivs=length(ivs),
            n_path=n, t_spots=t_spots, t_bars=t_bars, t_iv=t_iv, t_logsig=t_logsig)
end

# =============================================================================
# Run across multiple dates
# =============================================================================

dates = available_polygon_dates(store, "SPY")
test_dates = filter(d -> d >= Date(2025, 9, 1) && d <= Date(2025, 9, 30), dates)

println("Benchmarking IntradayLogSig on $(length(test_dates)) dates (SPY, Sep 2025)")
println("Path: (time, spot_log_return, ATM_IV_change), depth=3 → 14 dims")
println()

results = []
for d in test_dates
    r = bench_single(d)
    push!(results, r)
    @printf("  %s: spots=%2d bars=%6d iv=%2d path=%2d | spots=%.2fs bars=%.2fs iv=%.3fs logsig=%.0fμs\n",
        d, r.n_spots, r.n_bars, r.n_ivs, r.n_path,
        r.t_spots, r.t_bars, r.t_iv, r.t_logsig * 1e6)
end

println("\n", "=" ^ 70)
println("Summary ($(length(results)) dates)")
println("=" ^ 70)

for (label, field) in [
    ("Spots (DuckDB)",   :t_spots),
    ("Bars (DuckDB)",    :t_bars),
    ("ATM IV compute",   :t_iv),
    ("LogSig (warm)",    :t_logsig),
]
    vals = [getfield(r, field) for r in results]
    if field == :t_logsig
        @printf("  %-18s  mean=%.1fμs  std=%.1fμs\n", label, mean(vals)*1e6, std(vals)*1e6)
    else
        @printf("  %-18s  mean=%.3fs  std=%.3fs  min=%.3fs  max=%.3fs\n",
            label, mean(vals), std(vals), minimum(vals), maximum(vals))
    end
end

total = [r.t_spots + r.t_bars + r.t_iv + r.t_logsig for r in results]
@printf("\n  Total/date:         mean=%.3fs\n", mean(total))

avg_bars = mean([r.n_bars for r in results])
@printf("  Avg bars/date:      %.0f\n", avg_bars)
@printf("  Avg path points:    %.0f\n", mean([r.n_path for r in results]))
