# Strangle-selector sweep: post-hoc replay of many (Conditioning, Score, Picker)
# triplets over a single recorded backtest. Caches the per-entry PnL grid and
# the (rv, mom) state vectors once; each triplet's replay is O(folds × bins ×
# n_p × n_c × bin_size).
#
# Output: a run directory with summary.csv (one row per triplet),
# per_year.csv (triplet × year breakdown), chosen_combos.csv (audit trail),
# and config.toml.

using Dates
using Printf
using Statistics

# Internal: replay a single triplet over the cached pnl grid + state.
# Triplet is a NamedTuple with fields (name, conditioning, score, picker).
# `conditioning` may be a Conditioning subtype, or one of the sentinels
# `:peeking` (fixed `peeking_combo`) or `:frozen` (lock on first-fold pick).
function _replay_triplet(t, pnl::Vector{Matrix{Float64}},
                         state, dates::Vector{Date},
                         fold_starts::Vector{Date}, train_days::Int,
                         test_days::Int, put_deltas::Vector{Float64},
                         call_deltas::Vector{Float64};
                         peeking_combo::Tuple{Float64,Float64}=(0.20, 0.05))
    N = length(pnl)
    n_p, n_c = length(put_deltas), length(call_deltas)
    pnls = Float64[]; pdates = Date[]
    chosen_log = NamedTuple[]   # (fold_idx, bin, i, j, put_delta, call_delta)

    idx_in(lo, hi) = [k for k = 1:N if lo <= dates[k] < hi]

    if t.conditioning === :peeking
        i_fix = findfirst(==(peeking_combo[1]), put_deltas)
        j_fix = findfirst(==(peeking_combo[2]), call_deltas)
        (i_fix === nothing || j_fix === nothing) && error("peeking_combo not on grid: $peeking_combo")
        for ws in fold_starts, k in idx_in(ws, ws + Day(test_days))
            p = pnl[k][i_fix, j_fix]
            isnan(p) || (push!(pnls, p); push!(pdates, dates[k]))
        end
        return (name=t.name, pnls=pnls, dates=pdates, chosen_log=chosen_log)

    elseif t.conditioning === :frozen
        first_train = idx_in(fold_starts[1] - Day(train_days), fold_starts[1])
        S = _score_grid(pnl, first_train, t.score, n_p, n_c)
        ci = pick(t.picker, S, nothing)
        ci === nothing && return (name=t.name, pnls=Float64[], dates=Date[], chosen_log=chosen_log)
        push!(chosen_log, (fold_idx=1, bin=1, i=ci[1], j=ci[2],
                           put_delta=put_deltas[ci[1]], call_delta=call_deltas[ci[2]]))
        for ws in fold_starts, k in idx_in(ws, ws + Day(test_days))
            p = pnl[k][ci[1], ci[2]]
            isnan(p) || (push!(pnls, p); push!(pdates, dates[k]))
        end
        return (name=t.name, pnls=pnls, dates=pdates, chosen_log=chosen_log)

    else
        nb = n_bins(t.conditioning)
        prev_per_bin = Vector{Union{Nothing,CartesianIndex{2}}}(nothing, nb)

        for (fidx, ws) in enumerate(fold_starts)
            train = idx_in(ws - Day(train_days), ws)
            isempty(train) && continue
            thr = fit_threshold(t.conditioning, train, state)
            (thr isa AbstractFloat && isnan(thr)) && continue
            (thr isa Tuple && all(isnan, thr)) && continue

            bins = [Int[] for _ in 1:nb]
            for k in train
                b = classify(t.conditioning, k, thr, state)
                push!(bins[b], k)
            end

            chosen = Vector{Union{Nothing,CartesianIndex{2}}}(nothing, nb)
            for b in 1:nb
                isempty(bins[b]) && continue
                S = _score_grid(pnl, bins[b], t.score, n_p, n_c)
                chosen[b] = pick(t.picker, S, prev_per_bin[b])
                if chosen[b] !== nothing
                    prev_per_bin[b] = chosen[b]
                    push!(chosen_log, (fold_idx=fidx, bin=b,
                                       i=chosen[b][1], j=chosen[b][2],
                                       put_delta=put_deltas[chosen[b][1]],
                                       call_delta=call_deltas[chosen[b][2]]))
                end
            end

            for k in idx_in(ws, ws + Day(test_days))
                b = classify(t.conditioning, k, thr, state)
                ci = chosen[b]
                ci === nothing && (ci = chosen[1])
                ci === nothing && continue
                p = pnl[k][ci[1], ci[2]]
                isnan(p) || (push!(pnls, p); push!(pdates, dates[k]))
            end
        end
        return (name=t.name, pnls=pnls, dates=pdates, chosen_log=chosen_log)
    end
end

# Compute the score grid for a list of training indices.
function _score_grid(pnl::Vector{Matrix{Float64}}, idxs::Vector{Int},
                     score, n_p::Int, n_c::Int)
    S = fill(-Inf, n_p, n_c)
    @inbounds for i = 1:n_p, j = 1:n_c
        v = Float64[]
        for k in idxs
            p = pnl[k][i, j]
            isnan(p) || push!(v, p)
        end
        S[i, j] = score_pnls(score, v)
    end
    S
end

# Compute fold start dates given the date vector and rolling cadence.
function _fold_starts(dates::Vector{Date}, train_days::Int, test_days::Int,
                      step_days::Int)
    isempty(dates) && return Date[]
    starts = Date[]
    cur = dates[1] + Day(train_days)
    last_d = dates[end]
    while cur <= last_d
        push!(starts, cur)
        cur += Day(step_days)
    end
    return starts
end

# Triplet → human-readable description (for logging / config dump).
_describe(::Unconditional) = "Unconditional"
_describe(c::RVBinary)     = "RVBinary(rv_window=$(c.rv_window))"
_describe(c::RVTertile)    = "RVTertile(rv_window=$(c.rv_window))"
_describe(c::MomSign)      = "MomSign(mom_window=$(c.mom_window))"
_describe(c::RVxMom)       = "RVxMom(rv_window=$(c.rv_window), mom_window=$(c.mom_window))"
_describe(::Symbol)        = "(sentinel)"

_describe(::MeanScore)     = "MeanScore"
_describe(s::MeanVarScore) = "MeanVarScore(λ=$(s.lambda))"
_describe(s::SharpeScore)  = "SharpeScore(annualization=$(s.annualization))"
_describe(s::SharpeHeldOut)= "SharpeHeldOut(fraction=$(s.fraction))"
_describe(s::CVaRScore)    = "CVaRScore(z=$(s.z), α=$(s.alpha))"
_describe(::Nothing)       = "(none)"

_describe(::ArgmaxPicker)    = "ArgmaxPicker"
_describe(p::TopKPicker)     = "TopKPicker(K=$(p.K))"
_describe(::ShrinkagePicker) = "ShrinkagePicker"
_describe(p::StickyPicker)   = "StickyPicker(γ=$(p.gamma))"

# =============================================================================
# Public runner
# =============================================================================

"""
    run_strangle_selector_sweep(;
        output_root, symbol, start_date, end_date, entry_time, expiry_interval,
        max_tau_days, spread_lambda, rate, div_yield,
        put_deltas, call_deltas,
        train_days, test_days, step_days,
        rv_window=20, mom_window=5,
        peeking_combo=(0.20, 0.05),
        triplets::Vector{<:NamedTuple},
    )

Run the full strangle-selector sweep:
1. Load the data source for `symbol` over the date range.
2. Run a single base `run_strangle_rolling(...)` so the selector accumulates
   `state.history` (per-entry shadow credits across the n_p × n_c grid).
3. Cache the per-entry PnL grid and the (rv, mom) state vectors.
4. For each triplet `(name, conditioning, score, picker)`, replay over the
   cached data and accumulate per-trade PnL.
5. Write summary, per-year breakdown, chosen-combos audit, and config to
   `output_root/runs/strangle_selector_sweep_<symbol>_<timestamp>/`.

Returns a NamedTuple `(run_dir, results)` where `results` is a vector of
`(name, metrics, pnls, dates, chosen_log)` per triplet.
"""
function run_strangle_selector_sweep(;
    output_root::AbstractString,
    symbol::AbstractString,
    start_date::Date,
    end_date::Date,
    entry_time::Time,
    expiry_interval::Period,
    max_tau_days::Real,
    spread_lambda::Real,
    rate::Real,
    div_yield::Real,
    put_deltas::Vector{Float64},
    call_deltas::Vector{Float64},
    train_days::Integer,
    test_days::Integer,
    step_days::Integer,
    rv_window::Integer=20,
    mom_window::Integer=5,
    peeking_combo::Tuple{Float64,Float64}=(0.20, 0.05),
    triplets::Vector,
)
    run_dir = make_run_dir(output_root, "strangle_selector_sweep_$(symbol)")
    println("Output: $run_dir")
    println("\n  $symbol  $start_date → $end_date")
    println("  entry=$entry_time  expiry=$expiry_interval  max_tau=$(max_tau_days)d")
    println("  grid=$(length(put_deltas))×$(length(call_deltas))  " *
            "train=$(train_days)d  test=$(test_days)d  step=$(step_days)d")
    println("  triplets=$(length(triplets))")

    println("\nLoading $symbol …")
    (; source, sched) = polygon_parquet_source(symbol;
        start_date=start_date, end_date=end_date, entry_time=entry_time,
        rate=rate, div_yield=div_yield, spread_lambda=spread_lambda)

    # === Step 1: base backtest using existing run_strangle_rolling ===
    println("\nRunning base backtest (Sharpe selector — only used to populate state.history) …")
    base = run_strangle_rolling(source, sched, expiry_interval;
        put_deltas=put_deltas, call_deltas=call_deltas,
        train_days=train_days, test_days=test_days, step_days=step_days,
        rate=rate, div_yield=div_yield, max_tau_days=Float64(max_tau_days),
        baseline_combo=peeking_combo)

    history = sort(base.selector.state.history, by=h->h.entry_ts)
    N = length(history)
    println("  history N=$N entries")

    # === Step 2: cache pnl grid + state vectors ===
    println("\nCaching per-entry PnL grid and (rv, mom) state vectors …")
    pnl   = compute_pnl_grid(history, source)
    state = compute_strangle_state(history; rv_window=rv_window, mom_window=mom_window)
    dates = [Date(h.entry_ts) for h in history]
    fold_starts = _fold_starts(dates, train_days, test_days, step_days)
    println("  $(length(fold_starts)) folds")

    # === Step 3: replay each triplet ===
    println("\nReplaying $(length(triplets)) triplets …")
    results = NamedTuple[]
    for t in triplets
        triplet = (name=String(t.name), conditioning=t.conditioning,
                   score=t.score, picker=t.picker)
        r = _replay_triplet(triplet, pnl, state, dates, fold_starts,
                            Int(train_days), Int(test_days),
                            put_deltas, call_deltas;
                            peeking_combo=peeking_combo)
        m = pnl_summary(r.pnls)
        dist = pnl_distribution_stats(r.pnls)
        @printf("  %-46s  N=%4d  Sh=%5.2f  total=%+8.1f  DD=%5.1f  win%%=%4.1f\n",
                triplet.name, m.n, m.sharpe, m.total, m.mdd, 100*dist.win_rate)
        push!(results, (name=triplet.name, triplet=triplet, metrics=m, dist=dist,
                        pnls=r.pnls, dates=r.dates, chosen_log=r.chosen_log))
    end

    # === Step 4: write outputs ===
    println("\nWriting CSV outputs …")
    _write_summary_csv(joinpath(run_dir, "summary.csv"), results)
    _write_per_year_csv(joinpath(run_dir, "per_year.csv"), results)
    _write_chosen_combos_csv(joinpath(run_dir, "chosen_combos.csv"), results)
    _write_config(joinpath(run_dir, "config.txt"), symbol, start_date, end_date,
                  entry_time, expiry_interval, max_tau_days, spread_lambda,
                  rate, div_yield, put_deltas, call_deltas,
                  train_days, test_days, step_days, rv_window, mom_window,
                  peeking_combo, triplets)
    println("  done.")

    return (run_dir=run_dir, results=results)
end

# CSV writers --------------------------------------------------------------

function _write_summary_csv(path, results)
    rows = NamedTuple[]
    for r in results
        push!(rows, (
            triplet         = r.name,
            conditioning    = _describe(r.triplet.conditioning),
            score           = _describe(r.triplet.score),
            picker          = _describe(r.triplet.picker),
            n_trades        = r.metrics.n,
            total_pnl       = r.metrics.total,
            sharpe          = r.metrics.sharpe,
            avg_pnl         = r.metrics.avg,
            max_drawdown    = r.metrics.mdd,
            win_rate        = r.dist.win_rate,
            mean_pnl        = r.dist.mean_pnl,
            std_pnl         = r.dist.std_pnl,
            min_pnl         = r.dist.min_pnl,
            p5_pnl          = r.dist.p5_pnl,
            median_pnl      = r.dist.median_pnl,
            p95_pnl         = r.dist.p95_pnl,
            max_pnl         = r.dist.max_pnl,
            avg_win         = r.dist.avg_win,
            avg_loss        = r.dist.avg_loss,
            skewness        = r.dist.skewness,
        ))
    end
    write_namedtuple_csv(path, rows)
end

function _write_per_year_csv(path, results)
    rows = NamedTuple[]
    for r in results
        years = isempty(r.dates) ? Int[] : sort(unique(year.(r.dates)))
        for y in years
            mask = findall(d -> year(d) == y, r.dates)
            n_y = length(mask)
            n_y < 1 && continue
            pnls_y = r.pnls[mask]
            m_y = pnl_summary(pnls_y)
            dist_y = pnl_distribution_stats(pnls_y)
            push!(rows, (
                triplet      = r.name,
                year         = y,
                n_trades     = m_y.n,
                total_pnl    = m_y.total,
                sharpe       = m_y.sharpe,
                max_drawdown = m_y.mdd,
                win_rate     = dist_y.win_rate,
                mean_pnl     = dist_y.mean_pnl,
                std_pnl      = dist_y.std_pnl,
            ))
        end
    end
    write_namedtuple_csv(path, rows)
end

function _write_chosen_combos_csv(path, results)
    rows = NamedTuple[]
    for r in results
        for c in r.chosen_log
            push!(rows, (
                triplet     = r.name,
                fold_idx    = c.fold_idx,
                bin         = c.bin,
                put_delta   = c.put_delta,
                call_delta  = c.call_delta,
            ))
        end
    end
    write_namedtuple_csv(path, rows)
end

function _write_config(path, symbol, start_date, end_date, entry_time,
                       expiry_interval, max_tau_days, spread_lambda, rate,
                       div_yield, put_deltas, call_deltas, train_days,
                       test_days, step_days, rv_window, mom_window,
                       peeking_combo, triplets)
    open(path, "w") do io
        println(io, "symbol = $symbol")
        println(io, "period = $start_date → $end_date")
        println(io, "entry_time = $entry_time")
        println(io, "expiry_interval = $expiry_interval")
        println(io, "max_tau_days = $max_tau_days")
        println(io, "spread_lambda = $spread_lambda")
        println(io, "rate = $rate")
        println(io, "div_yield = $div_yield")
        println(io, "put_deltas = $put_deltas")
        println(io, "call_deltas = $call_deltas")
        println(io, "train/test/step = $train_days / $test_days / $step_days")
        println(io, "rv_window = $rv_window")
        println(io, "mom_window = $mom_window")
        println(io, "peeking_combo = $peeking_combo")
        println(io, "")
        println(io, "triplets:")
        for t in triplets
            println(io, "  - $(t.name):")
            println(io, "      conditioning = $(_describe(t.conditioning))")
            println(io, "      score        = $(_describe(t.score))")
            println(io, "      picker       = $(_describe(t.picker))")
        end
    end
end
