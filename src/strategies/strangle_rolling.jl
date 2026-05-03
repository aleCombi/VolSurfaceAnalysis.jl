# End-to-end runner + reporter for rolling-delta strangle backtests.
# Wraps RollingDeltaStrangleSelector + ShortStrangleStrategy + backtest_strategy
# and computes all the post-hoc diagnostics (per-fold IS/OOS Sharpe, rank-IC,
# baseline rank, year-by-year tables, equity curves, summary CSVs) so calling
# scripts contain no loops or analysis logic.

using Printf
using Plots: plot, plot!, hline!, savefig, heatmap

# =============================================================================
# Math helpers (private)
# =============================================================================

_sharpe_nan_of(v::AbstractVector{<:Real}) = let c = filter(!isnan, v)
    isempty(c) || std(c) == 0 ? NaN : mean(c) / std(c) * sqrt(252)
end

function _cvar_of(v::AbstractVector{<:Real}, alpha::Float64)
    c = filter(!isnan, v)
    n = length(c); n < 2 && return -Inf
    k = max(1, ceil(Int, n * alpha))
    mean(sort(c)[1:k])
end

function _spearman(x, y)
    n = length(x); n < 3 && return NaN
    rx = sortperm(sortperm(x)); ry = sortperm(sortperm(y))
    d2 = sum((rx .- ry) .^ 2)
    return 1 - 6 * d2 / (n * (n^2 - 1))
end

_year_mask(dates::AbstractVector{Date}, y::Integer) = Dates.year.(dates) .== y

function _annual_sharpe(pnls::AbstractVector{<:Real}, dates::AbstractVector{Date},
                      year::Integer)
    mask = _year_mask(dates, year)
    v = filter(!isnan, pnls[mask])
    return isempty(v) || std(v) == 0 ? NaN : mean(v) / std(v) * sqrt(252)
end

function _annual_total(pnls::AbstractVector{<:Real}, dates::AbstractVector{Date},
                     year::Integer)
    mask = _year_mask(dates, year)
    v = filter(!isnan, pnls[mask])
    return isempty(v) ? 0.0 : sum(v)
end

# =============================================================================
# Selector-state extraction (private)
# =============================================================================

# Compute per-combo PnL (USD) for a single entry in selector.state.history.
# Returns (pnl_matrix::Matrix{Float64}, settled::Bool). NaN where leg failed
# or settlement spot missing.
function _entry_pnl_matrix(entry, source::BacktestDataSource)
    spot_settle = get_settlement_spot(source, entry.expiry)
    if ismissing(spot_settle)
        return (fill(NaN, size(entry.credit_frac)), false)
    end
    s = Float64(spot_settle)
    n_p, n_c = size(entry.credit_frac)
    out = Matrix{Float64}(undef, n_p, n_c)
    @inbounds for i in 1:n_p, j in 1:n_c
        cf = entry.credit_frac[i, j]
        if isnan(cf)
            out[i, j] = NaN
            continue
        end
        sp_K = entry.put_strikes[i]
        sc_K = entry.call_strikes[j]
        credit_usd    = cf * entry.spot_at_entry
        intrinsic_usd = max(sp_K - s, 0.0) + max(s - sc_K, 0.0)
        out[i, j] = credit_usd - intrinsic_usd
    end
    return (out, true)
end

# Build a (n_entries × n_p × n_c) PnL tensor for all entries with settlement.
# Entries without settlement are dropped.
function _build_history_pnl_tensor(selector::RollingDeltaStrangleSelector,
                                  source::BacktestDataSource)
    n_p = length(selector.config.put_deltas)
    n_c = length(selector.config.call_deltas)
    dates  = Date[]
    tensor = Matrix{Float64}[]
    for entry in selector.state.history
        m, settled = _entry_pnl_matrix(entry, source)
        settled || continue
        push!(dates, Date(entry.entry_ts))
        push!(tensor, m)
    end
    return (dates, tensor, n_p, n_c)
end

# =============================================================================
# Fold diagnostics (private)
# =============================================================================

# For each window-boundary recompute, compute the test-window OOS metrics.
# A fold's test window is [window_start, next_window_start - 1] (or until the
# last available entry for the final fold). With step_days = test_days these
# windows partition the OOS period; with step_days < test_days they overlap.
function _fold_diagnostics(selector::RollingDeltaStrangleSelector,
                          source::BacktestDataSource,
                          baseline_combo::Tuple{Float64,Float64})
    cfg = selector.config
    fold_choices = selector.state.fold_choices
    isempty(fold_choices) && return NamedTuple[]

    dates, tensors, n_p, n_c = _build_history_pnl_tensor(selector, source)
    isempty(dates) && return NamedTuple[]

    # baseline indices in the candidate grid
    baseline_pi = findfirst(==(baseline_combo[1]), cfg.put_deltas)
    baseline_pj = findfirst(==(baseline_combo[2]), cfg.call_deltas)

    folds = NamedTuple[]
    for (k, fc) in enumerate(fold_choices)
        train_start = fc.train_start
        train_end   = fc.train_end
        # Test window runs from window_start through one full TEST_DAYS chunk
        test_start  = fc.window_start
        test_end    = test_start + Day(cfg.test_days) - Day(1)

        train_mask = (train_start .<= dates) .& (dates .<= train_end)
        test_mask  = (test_start  .<= dates) .& (dates .<= test_end)
        n_te = sum(test_mask)
        n_tr = sum(train_mask)

        # IS and OOS Sharpe per combo
        train_sh = fill(NaN, n_p, n_c)
        test_sh  = fill(NaN, n_p, n_c)
        for i in 1:n_p, j in 1:n_c
            tr_pnls = [tensors[t][i, j] for t in 1:length(tensors) if train_mask[t]]
            te_pnls = [tensors[t][i, j] for t in 1:length(tensors) if test_mask[t]]
            train_sh[i, j] = _sharpe_nan_of(tr_pnls)
            test_sh[i, j]  = _sharpe_nan_of(te_pnls)
        end

        chosen_pi, chosen_pj = fc.chosen_idx
        chosen_pnls = filter(!isnan,
            [tensors[t][chosen_pi, chosen_pj] for t in 1:length(tensors) if test_mask[t]])
        baseline_pnls = if baseline_pi === nothing || baseline_pj === nothing
            Float64[]
        else
            filter(!isnan,
                [tensors[t][baseline_pi, baseline_pj] for t in 1:length(tensors) if test_mask[t]])
        end

        # Rank of baseline in IS by score
        scores = fc.scores
        baseline_is_rank = if baseline_pi === nothing || baseline_pj === nothing
            -1
        else
            valid = isfinite.(scores)
            valid_lin = findall(valid)
            valid_scores = [scores[idx] for idx in valid_lin]
            ranks_desc = sortperm(valid_scores; rev=true)
            baseline_lin = LinearIndices(scores)[baseline_pi, baseline_pj]
            pos = findfirst(==(baseline_lin), valid_lin)
            pos === nothing ? -1 : findfirst(==(pos), ranks_desc)
        end

        # IS→OOS Spearman ρ over valid combos
        is_vec  = vec(train_sh); oos_vec = vec(test_sh)
        common  = .!isnan.(is_vec) .& .!isnan.(oos_vec)
        rho     = _spearman(is_vec[common], oos_vec[common])

        chosen_train_sh = train_sh[chosen_pi, chosen_pj]
        chosen_test_sh  = test_sh[chosen_pi, chosen_pj]
        base_train_sh   = baseline_pi === nothing ? NaN : train_sh[baseline_pi, baseline_pj]
        base_test_sh    = baseline_pi === nothing ? NaN : test_sh[baseline_pi, baseline_pj]

        push!(folds, (
            idx               = k,
            test              = (test_start, test_end),
            train             = (train_start, train_end),
            chosen            = fc.chosen,
            chosen_idx        = fc.chosen_idx,
            score             = scores[chosen_pi, chosen_pj],
            n_tr              = n_tr,
            n_te              = n_te,
            train_sh_chosen   = chosen_train_sh,
            test_sh_chosen    = chosen_test_sh,
            train_sh_base     = base_train_sh,
            test_sh_base      = base_test_sh,
            test_total_chosen = sum(chosen_pnls),
            test_total_base   = sum(baseline_pnls),
            baseline_is_rank  = baseline_is_rank,
            spearman          = rho,
        ))
    end
    return folds
end

# OOS PnL of the chosen combo per fold's test window (using the chosen at
# fold start, NOT the per-entry chosen_combo recorded in history — this
# matches the original rolling_select semantics).
function _oos_chosen_pnls(selector::RollingDeltaStrangleSelector,
                         source::BacktestDataSource)
    cfg = selector.config
    fold_choices = selector.state.fold_choices
    isempty(fold_choices) && return (Float64[], Date[], Tuple{Float64,Float64}[])

    dates, tensors, _, _ = _build_history_pnl_tensor(selector, source)
    pnls   = Float64[]
    out_d  = Date[]
    combos = Tuple{Float64,Float64}[]
    for fc in fold_choices
        test_start = fc.window_start
        test_end   = test_start + Day(cfg.test_days) - Day(1)
        i, j = fc.chosen_idx
        for t in 1:length(tensors)
            d = dates[t]
            (test_start <= d <= test_end) || continue
            v = tensors[t][i, j]
            isnan(v) && continue
            push!(pnls, v)
            push!(out_d, d)
            push!(combos, fc.chosen)
        end
    end
    return (pnls, out_d, combos)
end

# Baseline PnL on the same OOS test windows (using fixed baseline_combo).
function _oos_baseline_pnls(selector::RollingDeltaStrangleSelector,
                           source::BacktestDataSource,
                           baseline_combo::Tuple{Float64,Float64})
    cfg = selector.config
    bi = findfirst(==(baseline_combo[1]), cfg.put_deltas)
    bj = findfirst(==(baseline_combo[2]), cfg.call_deltas)
    (bi === nothing || bj === nothing) && return (Float64[], Date[])
    fold_choices = selector.state.fold_choices
    isempty(fold_choices) && return (Float64[], Date[])

    dates, tensors, _, _ = _build_history_pnl_tensor(selector, source)
    pnls  = Float64[]
    out_d = Date[]
    for fc in fold_choices
        test_start = fc.window_start
        test_end   = test_start + Day(cfg.test_days) - Day(1)
        for t in 1:length(tensors)
            d = dates[t]
            (test_start <= d <= test_end) || continue
            v = tensors[t][bi, bj]
            isnan(v) && continue
            push!(pnls, v)
            push!(out_d, d)
        end
    end
    return (pnls, out_d)
end

# =============================================================================
# Public runner
# =============================================================================

"""
    run_strangle_rolling(source, schedule, expiry_interval;
        put_deltas, call_deltas,
        train_days, test_days, step_days = test_days,
        rate, div_yield, max_tau_days = Inf,
        score = nothing,
        baseline_combo) -> NamedTuple

Run a `RollingDeltaStrangleSelector` end-to-end via `ShortStrangleStrategy +
backtest_strategy`, then compute per-fold OOS diagnostics. The returned
NamedTuple has:

- `oos_pnls::Vector{Float64}`, `oos_dates::Vector{Date}`,
  `oos_combos::Vector{Tuple{Float64,Float64}}` — per-OOS-entry PnL of the
  chosen combo
- `baseline_pnls`, `baseline_dates` — same OOS windows, fixed
  `baseline_combo` (must lie in the candidate grid)
- `folds::Vector{NamedTuple}` — per-window IS/OOS Sharpe per combo, rank-IC,
  chosen vs baseline gaps, baseline IS rank
- `selector` — raw selector for advanced inspection
- `result::BacktestResult` — raw backtest result

Pass `score::Function` to override the default annualized-Sharpe scorer.
"""
function run_strangle_rolling(
    source::BacktestDataSource,
    schedule::Vector{DateTime},
    expiry_interval::Period;
    put_deltas::Vector{Float64},
    call_deltas::Vector{Float64},
    train_days::Int,
    test_days::Int,
    step_days::Int = test_days,
    rate::Float64,
    div_yield::Float64,
    max_tau_days::Float64 = Inf,
    score = _ann_sharpe,
    baseline_combo::Tuple{Float64,Float64},
)
    selector = RollingDeltaStrangleSelector(;
        put_deltas, call_deltas,
        train_days, test_days, step_days,
        rate, div_yield, max_tau_days, score,
    )
    strategy = ShortStrangleStrategy(schedule, expiry_interval, selector)
    result = backtest_strategy(strategy, source)

    folds = _fold_diagnostics(selector, source, baseline_combo)
    oos_pnls, oos_dates, oos_combos = _oos_chosen_pnls(selector, source)
    baseline_pnls, baseline_dates   = _oos_baseline_pnls(selector, source, baseline_combo)

    return (
        oos_pnls       = oos_pnls,
        oos_dates      = oos_dates,
        oos_combos     = oos_combos,
        baseline_pnls  = baseline_pnls,
        baseline_dates = baseline_dates,
        folds          = folds,
        selector       = selector,
        result         = result,
    )
end

"""
    run_strangle_rolling_ensemble(source, schedule, expiry_interval;
        put_deltas, call_deltas,
        train_days, test_days, step_days = test_days,
        rate, div_yield, max_tau_days = Inf,
        z_values::Vector{Float64}, cvar_alpha::Float64 = 0.05,
        baseline_combo) -> Vector{NamedTuple}

Run `run_strangle_rolling` once per `z` in `z_values` with the score function
`score(v) = mean(v) - z * |CVaR_α(v)|`. Returns a vector of run NamedTuples,
each with an extra `z` field. Useful for sweeping CVaR-regularization.
"""
function run_strangle_rolling_ensemble(
    source::BacktestDataSource,
    schedule::Vector{DateTime},
    expiry_interval::Period;
    put_deltas::Vector{Float64},
    call_deltas::Vector{Float64},
    train_days::Int,
    test_days::Int,
    step_days::Int = test_days,
    rate::Float64,
    div_yield::Float64,
    max_tau_days::Float64 = Inf,
    z_values::Vector{Float64},
    cvar_alpha::Float64 = 0.05,
    baseline_combo::Tuple{Float64,Float64},
)
    runs = NamedTuple[]
    for z in z_values
        score = let z=z, alpha=cvar_alpha
            v -> begin
                c = filter(!isnan, v)
                isempty(c) && return -Inf
                m = mean(c)
                cv = _cvar_of(c, alpha)
                m - z * abs(cv)
            end
        end
        run = run_strangle_rolling(source, schedule, expiry_interval;
            put_deltas, call_deltas,
            train_days, test_days, step_days,
            rate, div_yield, max_tau_days, score, baseline_combo)
        push!(runs, (z = z, run...))
    end
    return runs
end

# =============================================================================
# Reporting (private printers + public dispatcher)
# =============================================================================

function _print_yearly_sharpe(io::IO, ensemble, baseline_combo)
    base_p = ensemble[1].baseline_pnls; base_d = ensemble[1].baseline_dates
    years  = sort(unique(Dates.year.(ensemble[1].oos_dates)))
    println(io, "\n", "=" ^ 80)
    println(io, "  Per-year Sharpe by z")
    println(io, "=" ^ 80)
    @printf io "  %-5s" "year"
    for r in ensemble; @printf io "  z=%-5.1f" r.z; end
    @printf io "    %s\n" "baseline"
    println(io, "  " * "─"^(8 + 9 * length(ensemble) + 12))
    for y in years
        @printf io "  %-5d" y
        for r in ensemble
            @printf io "  %+7.2f" _annual_sharpe(r.oos_pnls, r.oos_dates, y)
        end
        @printf io "    %+7.2f\n" _annual_sharpe(base_p, base_d, y)
    end
end

function _print_yearly_total(io::IO, ensemble)
    base_p = ensemble[1].baseline_pnls; base_d = ensemble[1].baseline_dates
    years  = sort(unique(Dates.year.(ensemble[1].oos_dates)))
    println(io, "\n", "=" ^ 80)
    println(io, "  Per-year total \$ PnL by z")
    println(io, "=" ^ 80)
    @printf io "  %-5s" "year"
    for r in ensemble; @printf io "  z=%-7.1f" r.z; end
    @printf io "    %s\n" "baseline"
    println(io, "  " * "─"^(8 + 11 * length(ensemble) + 12))
    for y in years
        @printf io "  %-5d" y
        for r in ensemble
            @printf io "  %+9.0f" _annual_total(r.oos_pnls, r.oos_dates, y)
        end
        @printf io "    %+7.0f\n" _annual_total(base_p, base_d, y)
    end
end

function _print_full_summary(io::IO, ensemble)
    years = sort(unique(Dates.year.(ensemble[1].oos_dates)))
    println(io, "\n", "=" ^ 80)
    println(io, "  Full-period summary by z")
    println(io, "=" ^ 80)
    @printf io "  %-6s  %5s  %+9s  %+8s  %+8s  %+8s  %5s\n" "z" "trades" "total" "Sharpe" "MeanYr" "MinYr" "+yrs"
    _safe_mean(v) = isempty(v) ? NaN : mean(v)
    _safe_min(v)  = isempty(v) ? NaN : minimum(v)
    for r in ensemble
        p = r.oos_pnls
        full_sh = isempty(p) || std(p) == 0 ? NaN : mean(p) / std(p) * sqrt(252)
        yr_sh = filter(!isnan, [_annual_sharpe(p, r.oos_dates, y) for y in years])
        @printf io "  z=%-4.1f  %5d  %+9.0f  %+8.2f  %+8.2f  %+8.2f  %2d/%-2d\n" (
            r.z, length(p), sum(p), full_sh, _safe_mean(yr_sh), _safe_min(yr_sh),
            count(>(0), yr_sh), length(yr_sh),
        )...
    end
    bp = ensemble[1].baseline_pnls; bd = ensemble[1].baseline_dates
    b_full_sh = isempty(bp) || std(bp) == 0 ? NaN : mean(bp) / std(bp) * sqrt(252)
    b_yr_sh = filter(!isnan, [_annual_sharpe(bp, bd, y) for y in years])
    @printf io "  %-6s  %5d  %+9.0f  %+8.2f  %+8.2f  %+8.2f  %2d/%-2d\n" (
        "fixed", length(bp), sum(bp), b_full_sh, _safe_mean(b_yr_sh), _safe_min(b_yr_sh),
        count(>(0), b_yr_sh), length(b_yr_sh),
    )...
end

function _print_combo_diversity(io::IO, ensemble, baseline_combo)
    println(io, "\n", "=" ^ 80)
    println(io, "  Combo selection diversity by z")
    println(io, "=" ^ 80)
    @printf io "  %-6s  %-5s  %-25s  %-25s\n" "z" "n_uniq" "top combo (% folds)" "$baseline_combo picks"
    for r in ensemble
        folds = r.folds
        counts = Dict{Tuple{Float64,Float64},Int}()
        for f in folds; counts[f.chosen] = get(counts, f.chosen, 0) + 1; end
        isempty(counts) && (println(io, "  z=$(r.z)  (no folds)"); continue)
        sorted = sort(collect(counts); by=x->x[2], rev=true)
        top, top_n = sorted[1]
        base_n = get(counts, baseline_combo, 0)
        @printf io "  z=%-4.1f  %-5d  (%.2f, %.2f) %3d/%-3d (%.0f%%)   %d/%d (%.0f%%)\n" (
            r.z, length(counts), top[1], top[2], top_n, length(folds),
            100*top_n/length(folds), base_n, length(folds),
            100*base_n/length(folds),
        )...
    end
end

function _print_diagnostic_detail(io::IO, run, baseline_combo)
    folds = run.folds
    isempty(folds) && (println(io, "\n  (no folds — no diagnostic detail)"); return)

    n_chosen_eq_base = count(f -> f.chosen == baseline_combo, folds)
    @printf io "\n  Rolling picked %s in %d / %d folds (%.1f%%)\n" (
        baseline_combo, n_chosen_eq_base, length(folds),
        100*n_chosen_eq_base/length(folds),
    )...

    diff_test_sh = [f.test_sh_chosen - f.test_sh_base for f in folds
                    if !isnan(f.test_sh_chosen) && !isnan(f.test_sh_base)]
    diff_total   = [f.test_total_chosen - f.test_total_base for f in folds]
    if !isempty(diff_test_sh)
        @printf io "\n  Test-Sharpe diff (chosen − baseline): mean %+.3f   median %+.3f   wins %d/%d\n" (
            mean(diff_test_sh), median(diff_test_sh),
            count(>(0), diff_test_sh), length(diff_test_sh),
        )...
    end
    @printf io "  Test-PnL diff (chosen − baseline):   mean %+.2f   total %+.0f   wins %d/%d\n" (
        mean(diff_total), sum(diff_total),
        count(>(0), diff_total), length(diff_total),
    )...

    tr_chosen = [f.train_sh_chosen for f in folds if !isnan(f.train_sh_chosen)]
    te_chosen = [f.test_sh_chosen  for f in folds if !isnan(f.test_sh_chosen)]
    tr_base   = [f.train_sh_base   for f in folds if !isnan(f.train_sh_base)]
    te_base   = [f.test_sh_base    for f in folds if !isnan(f.test_sh_base)]
    if !isempty(tr_chosen) && !isempty(te_chosen)
        @printf io "\n  Mean train Sharpe of chosen combo: %+.2f\n" mean(tr_chosen)
        @printf io "  Mean test  Sharpe of chosen combo: %+.2f   (gap %+.2f)\n" mean(te_chosen) (mean(te_chosen) - mean(tr_chosen))
    end
    if !isempty(tr_base) && !isempty(te_base)
        @printf io "  Mean train Sharpe of baseline: %+.2f\n" mean(tr_base)
        @printf io "  Mean test  Sharpe of baseline: %+.2f   (gap %+.2f)\n" mean(te_base) (mean(te_base) - mean(tr_base))
    end

    rhos = [f.spearman for f in folds if !isnan(f.spearman)]
    if !isempty(rhos)
        @printf io "\n  IS→OOS Spearman ρ per fold:  mean %+.3f   median %+.3f   N=%d\n" (
            mean(rhos), median(rhos), length(rhos),
        )...
        @printf io "  Folds with ρ > 0:   %d / %d\n"   count(>(0),   rhos) length(rhos)
        @printf io "  Folds with ρ > 0.3: %d / %d\n"   count(>(0.3), rhos) length(rhos)
    end

    baseline_ranks = [f.baseline_is_rank for f in folds if f.baseline_is_rank > 0]
    if !isempty(baseline_ranks)
        n_combos = length(folds[1].chosen_idx) > 0 ? size(run.selector.config.put_deltas, 1) * size(run.selector.config.call_deltas, 1) : 0
        @printf io "\n  Baseline %s IS rank:  median %d  mean %.1f  (out of %d)\n" (
            baseline_combo, median(baseline_ranks), mean(baseline_ranks), n_combos,
        )...
        @printf io "  In-sample top-5 contains baseline:  %d / %d  (%.1f%%)\n" (
            count(<=(5), baseline_ranks), length(baseline_ranks),
            100*count(<=(5), baseline_ranks)/length(baseline_ranks),
        )...
    end

    println(io, "\n  ── Per-fold detail ──")
    @printf io "  %3s  %-23s  %-12s  %+8s  %+8s  %+8s  %+8s  %4s\n" "#" "test window" "chosen" "trS_ch" "teS_ch" "trS_b" "teS_b" "rkB"
    for f in folds
        @printf io "  %3d  %s → %s  (%.2f,%.2f)  %+8.2f  %+8.2f  %+8.2f  %+8.2f  %4d\n" (
            f.idx, f.test[1], f.test[2], f.chosen[1], f.chosen[2],
            f.train_sh_chosen, f.test_sh_chosen,
            f.train_sh_base,   f.test_sh_base,
            f.baseline_is_rank,
        )...
    end
end

function _save_summary_csv(path::AbstractString, ensemble)
    years = sort(unique(Dates.year.(ensemble[1].oos_dates)))
    open(path, "w") do io
        println(io, "z,year,n,total,sharpe")
        for r in ensemble, y in years
            mask = _year_mask(r.oos_dates, y)
            v = filter(!isnan, r.oos_pnls[mask])
            isempty(v) && continue
            sh = std(v) == 0 ? NaN : mean(v) / std(v) * sqrt(252)
            @printf io "%.1f,%d,%d,%.2f,%.4f\n" r.z y length(v) sum(v) sh
        end
    end
end

function _save_fold_diagnostic_csv(path::AbstractString, run)
    open(path, "w") do io
        println(io, "fold,test_start,test_end,chosen_pd,chosen_cd,train_sh_chosen,test_sh_chosen,train_sh_base,test_sh_base,test_total_chosen,test_total_base,baseline_is_rank,spearman")
        for f in run.folds
            @printf io "%d,%s,%s,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%d,%.4f\n" (
                f.idx, f.test[1], f.test[2], f.chosen[1], f.chosen[2],
                f.train_sh_chosen, f.test_sh_chosen,
                f.train_sh_base,   f.test_sh_base,
                f.test_total_chosen, f.test_total_base,
                f.baseline_is_rank, f.spearman,
            )...
        end
    end
end

function _save_equity_curves(path::AbstractString, ensemble; baseline_combo, title="")
    plt = plot(; xlabel="date", ylabel="cumulative PnL (USD)",
        title=title, size=(1100, 600), legend=:topleft)
    palette = [:steelblue, :seagreen, :darkorange, :firebrick, :purple,
               :teal, :goldenrod, :navy, :crimson]
    for (i, r) in enumerate(ensemble)
        ord = sortperm(r.oos_dates)
        plot!(plt, r.oos_dates[ord], cumsum(r.oos_pnls[ord]);
            label="z=$(r.z)", lw=2, color=palette[mod1(i, length(palette))])
    end
    base_p = ensemble[1].baseline_pnls; base_d = ensemble[1].baseline_dates
    if !isempty(base_p)
        ord = sortperm(base_d)
        plot!(plt, base_d[ord], cumsum(base_p[ord]);
            label="fixed $baseline_combo", lw=2, color=:black, ls=:dash)
    end
    hline!(plt, [0]; color=:gray, ls=:dash, label="")
    savefig(plt, path)
end

function _save_yearly_heatmap(path::AbstractString, ensemble; title="")
    years = sort(unique(Dates.year.(ensemble[1].oos_dates)))
    sh_matrix = [_annual_sharpe(r.oos_pnls, r.oos_dates, y)
                 for r in ensemble, y in years]
    hm = heatmap(string.(years), ["z=$(r.z)" for r in ensemble], sh_matrix;
        color=:RdBu, clims=(-3.0, 5.0),
        xlabel="year", ylabel="regularization z",
        title=title, size=(900, 350),
    )
    savefig(hm, path)
end

"""
    report_strangle_rolling(ensemble; run_dir, baseline_combo,
                           diagnostic = (length(ensemble) == 1),
                           title_prefix = "")

Print all standard tables (per-year Sharpe, per-year total, full-period
summary, combo diversity) and save the standard artefacts (`summary.csv`,
`equity_curves.png`, and either `fold_diagnostic.csv` in diagnostic mode or
`yearly_sharpe_by_z.png` in sweep mode) under `run_dir`.

`ensemble` should be the output of `run_strangle_rolling_ensemble`.
"""
function report_strangle_rolling(ensemble;
    run_dir::AbstractString,
    baseline_combo::Tuple{Float64,Float64},
    diagnostic::Bool = (length(ensemble) == 1),
    title_prefix::AbstractString = "",
)
    isempty(ensemble) && (println("(empty ensemble)"); return)

    _print_yearly_sharpe(stdout, ensemble, baseline_combo)
    _print_yearly_total(stdout, ensemble)
    _print_full_summary(stdout, ensemble)
    _print_combo_diversity(stdout, ensemble, baseline_combo)
    if diagnostic
        _print_diagnostic_detail(stdout, ensemble[1], baseline_combo)
        _save_fold_diagnostic_csv(joinpath(run_dir, "fold_diagnostic.csv"), ensemble[1])
        println("\n  Saved: fold_diagnostic.csv")
    end
    _save_summary_csv(joinpath(run_dir, "summary.csv"), ensemble)
    println("\n  Saved: summary.csv")
    eq_title = "$(title_prefix) — rolling regularized selector by z"
    _save_equity_curves(joinpath(run_dir, "equity_curves.png"), ensemble;
                       baseline_combo, title=eq_title)
    println("  Saved: equity_curves.png")
    if !diagnostic
        hm_title = "$(title_prefix): per-year Sharpe by z"
        _save_yearly_heatmap(joinpath(run_dir, "yearly_sharpe_by_z.png"), ensemble;
                            title=hm_title)
        println("  Saved: yearly_sharpe_by_z.png")
    end
    return nothing
end
