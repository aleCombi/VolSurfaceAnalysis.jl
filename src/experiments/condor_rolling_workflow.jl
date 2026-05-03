# Workflow wrapper for rolling iron-condor experiments.

using Dates
using Printf

function _condor_delta_dataset(symbol, start_date, end_date, entry_time, expiry_interval,
                               max_tau_days, delta_combos, wing_width,
                               rate, div_yield, spread_lambda)
    (; source, sched) = polygon_parquet_source(symbol;
        start_date=start_date, end_date=end_date, entry_time=entry_time,
        rate=rate, div_yield=div_yield, spread_lambda=spread_lambda,
    )

    dates = Date[]
    pnls_by_combo = Vector{Vector{Float64}}()
    n_total = 0
    n_skip = 0

    println("\nBuilding dataset (per-entry PnL x $(length(delta_combos)) combos)...")
    each_entry(source, expiry_interval, sched; clear_cache=true) do ctx, settlement
        ismissing(settlement) && return
        n_total += 1
        dctx = delta_context(ctx; rate=rate, div_yield=div_yield)
        if dctx === nothing || dctx.tau * 365.25 > max_tau_days
            n_skip += 1
            return
        end
        spot = dctx.spot

        row = Float64[]
        for (pd, cd) in delta_combos
            sp_K = delta_strike(dctx, -pd, Put)
            sc_K = delta_strike(dctx, cd, Call)
            if sp_K === nothing || sc_K === nothing
                push!(row, NaN)
                continue
            end
            lp_K = nearest_otm_strike(dctx, sp_K, wing_width, Put)
            lc_K = nearest_otm_strike(dctx, sc_K, wing_width, Call)
            if lp_K === nothing || lc_K === nothing
                push!(row, NaN)
                continue
            end
            positions = open_condor_positions(ctx, sp_K, sc_K, lp_K, lc_K)
            if length(positions) != 4
                push!(row, NaN)
                continue
            end
            push!(row, settle(positions, Float64(settlement)) * spot)
        end
        push!(dates, Date(ctx.surface.timestamp))
        push!(pnls_by_combo, row)
    end

    ord = sortperm(dates)
    pnl = isempty(ord) ? Matrix{Float64}(undef, 0, length(delta_combos)) : hcat(pnls_by_combo[ord]...)'
    return dates[ord], pnl, n_total, n_skip
end

function _condor_wing_dataset(symbol, start_date, end_date, entry_time, expiry_interval,
                              max_tau_days, put_delta, call_delta, wing_widths,
                              rate, div_yield, spread_lambda)
    (; source, sched) = polygon_parquet_source(symbol;
        start_date=start_date, end_date=end_date, entry_time=entry_time,
        rate=rate, div_yield=div_yield, spread_lambda=spread_lambda,
    )

    dates = Date[]
    pnls_by_width = Vector{Vector{Float64}}()
    n_total = 0
    n_skip = 0

    println("\nBuilding dataset (per-entry PnL x $(length(wing_widths)) widths)...")
    each_entry(source, expiry_interval, sched; clear_cache=true) do ctx, settlement
        ismissing(settlement) && return
        n_total += 1
        dctx = delta_context(ctx; rate=rate, div_yield=div_yield)
        if dctx === nothing || dctx.tau * 365.25 > max_tau_days
            n_skip += 1
            return
        end
        spot = dctx.spot

        sp_K = delta_strike(dctx, -put_delta, Put)
        sc_K = delta_strike(dctx, call_delta, Call)
        if sp_K === nothing || sc_K === nothing
            n_skip += 1
            return
        end

        row = Float64[]
        ok = true
        for wing in wing_widths
            lp_K = nearest_otm_strike(dctx, sp_K, wing, Put)
            lc_K = nearest_otm_strike(dctx, sc_K, wing, Call)
            if lp_K === nothing || lc_K === nothing
                ok = false
                break
            end
            positions = open_condor_positions(ctx, sp_K, sc_K, lp_K, lc_K)
            if length(positions) != 4
                ok = false
                break
            end
            push!(row, settle(positions, Float64(settlement)) * spot)
        end
        if !ok
            n_skip += 1
            return
        end
        push!(dates, Date(ctx.surface.timestamp))
        push!(pnls_by_width, row)
    end

    ord = sortperm(dates)
    pnl = isempty(ord) ? Matrix{Float64}(undef, 0, length(wing_widths)) : hcat(pnls_by_width[ord]...)'
    return dates[ord], pnl, n_total, n_skip
end

function _print_choice_usage(title, values, universe; formatter=string)
    println("\n  $title:")
    n_total = length(values)
    n_total == 0 && return
    for value in universe
        n = count(==(value), values)
        n == 0 && continue
        @printf("    %s -> %d trades (%.1f%%)\n", formatter(value), n, 100 * n / n_total)
    end
end

function _write_oos_series(path, dates, choices, pnls; choice_name::AbstractString)
    open(path, "w") do io
        println(io, "date,$choice_name,oos_pnl_usd")
        for i in eachindex(dates)
            @printf(io, "%s,%s,%.6f\n", string(dates[i]), string(choices[i]), pnls[i])
        end
    end
    return path
end

function run_condor_rolling_experiment(;
    output_root::AbstractString,
    mode::AbstractString=lowercase(get(ENV, "MODE", "delta")),
    symbol::AbstractString=get(ENV, "SYM", "SPY"),
    start_date::Date=Date(parse(Int, get(ENV, "START_YEAR", "2017")), 1, 1),
    end_date::Date=Date(get(ENV, "END_DATE", "2024-01-31")),
    spread_lambda::Real=parse(Float64, get(ENV, "SPREAD_LAMBDA", "0.7")),
    rate::Real=parse(Float64, get(ENV, "RATE", "0.045")),
    div_yield::Real=parse(Float64, get(ENV, "DIV", "0.013")),
    train_days::Integer=parse(Int, get(ENV, "TRAIN_DAYS", "90")),
    test_days::Integer=parse(Int, get(ENV, "TEST_DAYS", "30")),
    step_days::Integer=parse(Int, get(ENV, "STEP_DAYS", "30")),
    entry_time::Time=Time(parse(Int, get(ENV, "ENTRY_HOUR", "14")), 0),
    expiry_interval=nothing,
    max_tau_days=nothing,
    put_deltas=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    call_deltas=[0.05, 0.10, 0.15, 0.20],
    wing_widths=[1.0, 2.0, 3.0, 5.0, 8.0, 12.0],
    selection_wing_width::Real=parse(Float64, get(ENV, "SELECTION_WING", "12.0")),
    put_delta::Real=parse(Float64, get(ENV, "PUT_DELTA", "0.20")),
    call_delta::Real=parse(Float64, get(ENV, "CALL_DELTA", "0.05")),
    in_sample_end::Date=Date(parse(Int, get(ENV, "IS_END_YEAR", "2020")), 12, 31),
    tenor_train_str::AbstractString=lowercase(get(ENV, "TENOR_TRAIN", "1d")),
    tenor_test_str::AbstractString=lowercase(get(ENV, "TENOR_TEST", "2h")),
    max_tau_train=parse(Float64, get(ENV, "MAX_TAU_TRAIN", endswith(tenor_train_str, "d") ? "2.0" : "0.5")),
    max_tau_test=parse(Float64, get(ENV, "MAX_TAU_TEST", endswith(tenor_test_str, "d") ? "2.0" : "0.5")),
    entry_train::Time=Time(parse(Int, get(ENV, "ENTRY_HOUR_TRAIN", "14")), 0),
    entry_test::Time=Time(parse(Int, get(ENV, "ENTRY_HOUR_TEST", "14")), 0),
)
    mode = lowercase(mode)
    mode in ("delta", "wing", "joint", "2stage", "cross_tenor") ||
        error("mode must be delta|wing|joint|2stage|cross_tenor, got $mode")

    expiry_interval = expiry_interval === nothing ?
        parse_tenor(get(ENV, "TENOR", mode == "joint" ? "1d" : "2h")) : expiry_interval
    max_tau_days = max_tau_days === nothing ?
        parse(Float64, get(ENV, "MAX_TAU_DAYS", mode == "joint" ? "2.0" : "0.5")) : max_tau_days

    delta_combos = [(pd, cd) for pd in put_deltas for cd in call_deltas]
    run_dir = make_run_dir(output_root, "condor_rolling_$(mode)_$(symbol)")
    println("Output: $run_dir   MODE=$mode   $symbol  $start_date -> $end_date")
    println("  train=$train_days d / test=$test_days d / step=$step_days d")

    if mode == "delta"
        println("  Delta combos: $(length(delta_combos))   wing fixed: \$$selection_wing_width")
        dates, pnl, n_total, n_skip = _condor_delta_dataset(
            symbol, start_date, end_date, entry_time, expiry_interval, max_tau_days,
            delta_combos, selection_wing_width, rate, div_yield, spread_lambda)
        @printf("  %d entries -> kept %d  (skipped %d)\n", n_total, length(dates), n_skip)
        length(dates) < 50 && error("Too few entries")

        oos_pnls = Float64[]
        oos_dates = Date[]
        oos_combos = Tuple{Float64,Float64}[]
        folds = build_folds(dates; train_days=train_days, test_days=test_days,
            step_days=step_days, min_train=10, min_test=1)
        for fold in folds
            train_sharpes = [annualized_sharpe(pnl[fold.train_mask, c]) for c in 1:length(delta_combos)]
            best_c = argmax(train_sharpes)
            chosen = delta_combos[best_c]
            keep = .!isnan.(pnl[fold.test_mask, best_c])
            test_pnls = pnl[fold.test_mask, best_c][keep]
            test_dates = dates[fold.test_mask][keep]
            append!(oos_pnls, test_pnls)
            append!(oos_dates, test_dates)
            append!(oos_combos, fill(chosen, length(test_pnls)))
        end

        summary = pnl_summary(oos_pnls)
        @printf("\n  RESULT rolling-delta: trades=%d total=%+.0f Sharpe=%+.2f MaxDD=%.0f\n",
            summary.n, summary.total, summary.sharpe, summary.mdd)
        _print_choice_usage("Combo usage", oos_combos, unique(oos_combos);
            formatter=c -> @sprintf("(%.2f, %.2f)", c[1], c[2]))
        print_monthly_sharpe("rolling-delta OOS", oos_dates, oos_pnls)

        oos_idx = findall(d -> d in Set(oos_dates), dates)
        ref_idx = findfirst(==((0.20, 0.05)), delta_combos)
        ref_dates = nothing
        ref_pnls = nothing
        if ref_idx !== nothing
            keep = .!isnan.(pnl[oos_idx, ref_idx])
            ref_dates = dates[oos_idx][keep]
            ref_pnls = pnl[oos_idx, ref_idx][keep]
        end
        save_cumulative_pnl_comparison(oos_dates, oos_pnls, joinpath(run_dir, "cumulative.png");
            title="$symbol $(expiry_interval) iron condor - rolling delta selection",
            label="rolling-delta",
            reference_dates=ref_dates,
            reference_pnls=ref_pnls,
            reference_label="fixed 20p/5c")
        return (run_dir=run_dir, dates=oos_dates, pnls=oos_pnls, choices=oos_combos)
    end

    if mode == "wing"
        println("  Wing widths: $wing_widths   shorts fixed: ($put_delta, $call_delta)")
        dates, pnl, n_total, n_skip = _condor_wing_dataset(
            symbol, start_date, end_date, entry_time, expiry_interval, max_tau_days,
            put_delta, call_delta, wing_widths, rate, div_yield, spread_lambda)
        @printf("  %d entries -> kept %d  (skipped %d)\n", n_total, length(dates), n_skip)
        length(dates) < 50 && error("Too few entries")

        oos_pnls = Float64[]
        oos_dates = Date[]
        oos_wings = Float64[]
        fold_rows = NamedTuple[]
        test_start = dates[1] + Day(train_days)
        fold_idx = 0
        while test_start <= dates[end]
            test_end = test_start + Day(test_days) - Day(1)
            train_start = test_start - Day(train_days)
            train_end = test_start - Day(1)
            train_mask = (dates .>= train_start) .& (dates .<= train_end)
            test_mask = (dates .>= test_start) .& (dates .<= test_end)
            if sum(train_mask) < 10 || sum(test_mask) < 1
                test_start += Day(step_days)
                continue
            end
            train_sharpes = [annualized_sharpe(pnl[train_mask, w]; empty=0.0) for w in 1:length(wing_widths)]
            best_w = argmax(train_sharpes)
            chosen = wing_widths[best_w]
            test_pnls = pnl[test_mask, best_w]
            test_dates = dates[test_mask]
            append!(oos_pnls, test_pnls)
            append!(oos_dates, test_dates)
            append!(oos_wings, fill(chosen, length(test_pnls)))
            fold_idx += 1
            push!(fold_rows, (idx=fold_idx, test=(test_start, test_end), chosen=chosen,
                train_sharpe=train_sharpes[best_w], test_total=sum(test_pnls)))
            test_start += Day(step_days)
        end

        summary = pnl_summary(oos_pnls)
        @printf("\n  RESULT rolling-wing: trades=%d total=%+.0f AvgPnL=%+.2f Sharpe=%+.2f MaxDD=%.0f\n",
            summary.n, summary.total, summary.avg, summary.sharpe, summary.mdd)
        _print_choice_usage("Wing usage", oos_wings, wing_widths; formatter=w -> @sprintf("\$%.1f", w))
        print_monthly_sharpe("rolling-wing OOS", oos_dates, oos_pnls)
        _write_oos_series(joinpath(run_dir, "oos_series.csv"), oos_dates, oos_wings, oos_pnls; choice_name="chosen_wing")
        save_cumulative_pnl(oos_dates, oos_pnls, joinpath(run_dir, "cumulative.png");
            title="$symbol $(expiry_interval) iron condor - rolling wing selection",
            label="OOS cumulative")
        save_fold_choice_scatter([r.test[1] for r in fold_rows], [r.chosen for r in fold_rows],
            joinpath(run_dir, "wing_over_time.png");
            title="$symbol - wing chosen per test fold",
            ylabel="chosen wing width")
        return (run_dir=run_dir, dates=oos_dates, pnls=oos_pnls, choices=oos_wings)
    end

    if mode == "joint"
        dates, pnls_by_entry, n_total, n_skip = build_combo_wing_dataset(
            symbol, start_date, end_date, entry_time, expiry_interval, max_tau_days,
            delta_combos, wing_widths, rate, div_yield, spread_lambda)
        @printf("  %d total -> kept %d, skipped %d\n", n_total, length(dates), n_skip)
        length(dates) < 50 && error("Too few entries")

        sel_wi = findfirst(==(selection_wing_width), wing_widths)
        sel_wi === nothing && error("selection_wing_width=$selection_wing_width not in wing_widths")

        oos_pnls = Float64[]
        oos_dates = Date[]
        oos_combos = Tuple{Float64,Float64}[]
        oos_wings = Float64[]
        test_start = dates[1] + Day(train_days)
        while test_start <= dates[end]
            test_end = test_start + Day(test_days) - Day(1)
            train_start = test_start - Day(train_days)
            train_end = test_start - Day(1)
            train_idx = findall(d -> train_start <= d <= train_end, dates)
            test_idx = findall(d -> test_start <= d <= test_end, dates)
            if length(train_idx) < 10 || isempty(test_idx)
                test_start += Day(step_days)
                continue
            end
            combo_sharpes = [annualized_sharpe([pnls_by_entry[i][ci, sel_wi] for i in train_idx]) for ci in 1:length(delta_combos)]
            best_ci = argmax(combo_sharpes)
            wing_sharpes = [annualized_sharpe([pnls_by_entry[i][best_ci, wi] for i in train_idx]) for wi in 1:length(wing_widths)]
            best_wi = argmax(wing_sharpes)
            for i in test_idx
                value = pnls_by_entry[i][best_ci, best_wi]
                isnan(value) && continue
                push!(oos_pnls, value)
                push!(oos_dates, dates[i])
                push!(oos_combos, delta_combos[best_ci])
                push!(oos_wings, wing_widths[best_wi])
            end
            test_start += Day(step_days)
        end

        summary = pnl_summary(oos_pnls)
        @printf("\n  RESULT joint rolling: trades=%d total=%+.0f AvgPnL=%+.2f Sharpe=%+.2f MaxDD=%.0f\n",
            summary.n, summary.total, summary.avg, summary.sharpe, summary.mdd)
        _print_choice_usage("Delta combo usage", oos_combos, unique(oos_combos);
            formatter=c -> @sprintf("(%.2f, %.2f)", c[1], c[2]))
        _print_choice_usage("Wing usage", oos_wings, wing_widths; formatter=w -> @sprintf("\$%.1f", w))
        print_monthly_sharpe("joint OOS", oos_dates, oos_pnls)
        save_cumulative_pnl(oos_dates, oos_pnls, joinpath(run_dir, "cumulative.png");
            title="$symbol $(expiry_interval) - rolling delta then rolling wing",
            label="rolling delta + rolling wing")
        return (run_dir=run_dir, dates=oos_dates, pnls=oos_pnls, combos=oos_combos, wings=oos_wings)
    end

    if mode == "2stage"
        pd_grid = collect(0.050:0.025:0.300)
        cd_grid = collect(0.025:0.025:0.225)
        combos = [(pd, cd) for pd in pd_grid for cd in cd_grid]
        dates, pnls_by_entry, n_total, n_skip = build_combo_wing_dataset(
            symbol, start_date, end_date, entry_time, expiry_interval, max_tau_days,
            combos, wing_widths, rate, div_yield, spread_lambda)
        @printf("  %d total -> kept %d, skipped %d\n", n_total, length(dates), n_skip)
        length(dates) < 50 && error("Too few entries")

        is_idx = findall(<=(in_sample_end), dates)
        oos_idx = findall(>(in_sample_end), dates)
        sel_wi = findfirst(==(selection_wing_width), wing_widths)
        combo_sharpes = [annualized_sharpe([pnls_by_entry[i][ci, sel_wi] for i in is_idx]) for ci in 1:length(combos)]
        best_ci = argmax(combo_sharpes)
        chosen_pd, chosen_cd = combos[best_ci]
        println("  Stage 1 selected put=$chosen_pd call=$chosen_cd")

        oos_dates_full = dates[oos_idx]
        pnl_oos = Matrix{Float64}(undef, length(oos_idx), length(wing_widths))
        for (i, idx) in enumerate(oos_idx)
            pnl_oos[i, :] = pnls_by_entry[idx][best_ci, :]
        end

        oos_pnls = Float64[]
        oos_dates = Date[]
        oos_wings = Float64[]
        test_start = oos_dates_full[1] + Day(train_days)
        while test_start <= oos_dates_full[end]
            test_end = test_start + Day(test_days) - Day(1)
            train_start = test_start - Day(train_days)
            train_end = test_start - Day(1)
            train_mask = (oos_dates_full .>= train_start) .& (oos_dates_full .<= train_end)
            test_mask = (oos_dates_full .>= test_start) .& (oos_dates_full .<= test_end)
            if sum(train_mask) < 10 || sum(test_mask) < 1
                test_start += Day(step_days)
                continue
            end
            train_sharpes = [annualized_sharpe(pnl_oos[train_mask, w]) for w in 1:length(wing_widths)]
            best_w = argmax(train_sharpes)
            keep = .!isnan.(pnl_oos[test_mask, best_w])
            append!(oos_pnls, pnl_oos[test_mask, best_w][keep])
            append!(oos_dates, oos_dates_full[test_mask][keep])
            append!(oos_wings, fill(wing_widths[best_w], sum(keep)))
            test_start += Day(step_days)
        end

        summary = pnl_summary(oos_pnls)
        @printf("\n  RESULT 2stage: trades=%d total=%+.0f AvgPnL=%+.2f Sharpe=%+.2f MaxDD=%.0f\n",
            summary.n, summary.total, summary.avg, summary.sharpe, summary.mdd)
        _print_choice_usage("Wing usage", oos_wings, wing_widths; formatter=w -> @sprintf("\$%.1f", w))
        print_monthly_sharpe("2stage OOS", oos_dates, oos_pnls)
        save_cumulative_pnl(oos_dates, oos_pnls, joinpath(run_dir, "cumulative.png");
            title="$symbol $(expiry_interval) - Stage 2 rolling wing",
            label="Stage 2")
        return (run_dir=run_dir, dates=oos_dates, pnls=oos_pnls, wings=oos_wings, chosen_combo=(chosen_pd, chosen_cd))
    end

    tenor_train = parse_tenor(tenor_train_str)
    tenor_test = parse_tenor(tenor_test_str)
    dates_train, pnls_train, train_total, train_skip = build_combo_wing_dataset(
        symbol, start_date, end_date, entry_train, tenor_train, max_tau_train,
        delta_combos, wing_widths, rate, div_yield, spread_lambda)
    dates_apply, pnls_apply, test_total, test_skip = build_combo_wing_dataset(
        symbol, start_date, end_date, entry_test, tenor_test, max_tau_test,
        delta_combos, wing_widths, rate, div_yield, spread_lambda)
    @printf("  train: %d total, kept %d, skipped %d\n", train_total, length(dates_train), train_skip)
    @printf("  test:  %d total, kept %d, skipped %d\n", test_total, length(dates_apply), test_skip)

    oos_pnls = Float64[]
    oos_dates = Date[]
    oos_combos = Tuple{Float64,Float64}[]
    oos_wings = Float64[]
    test_start = max(dates_train[1], dates_apply[1]) + Day(train_days)
    last_date = min(dates_train[end], dates_apply[end])
    while test_start <= last_date
        test_end = test_start + Day(test_days) - Day(1)
        train_start = test_start - Day(train_days)
        train_end = test_start - Day(1)
        train_idx = findall(d -> train_start <= d <= train_end, dates_train)
        apply_idx = findall(d -> test_start <= d <= test_end, dates_apply)
        if length(train_idx) < 10 || isempty(apply_idx)
            test_start += Day(step_days)
            continue
        end
        best_sh = -Inf
        best_ci = 1
        best_wi = 1
        for ci in 1:length(delta_combos), wi in 1:length(wing_widths)
            sh = annualized_sharpe([pnls_train[i][ci, wi] for i in train_idx])
            if sh > best_sh
                best_sh = sh
                best_ci = ci
                best_wi = wi
            end
        end
        for i in apply_idx
            value = pnls_apply[i][best_ci, best_wi]
            isnan(value) && continue
            push!(oos_pnls, value)
            push!(oos_dates, dates_apply[i])
            push!(oos_combos, delta_combos[best_ci])
            push!(oos_wings, wing_widths[best_wi])
        end
        test_start += Day(step_days)
    end

    summary = pnl_summary(oos_pnls)
    @printf("\n  RESULT cross-tenor: trades=%d total=%+.0f AvgPnL=%+.2f Sharpe=%+.2f MaxDD=%.0f\n",
        summary.n, summary.total, summary.avg, summary.sharpe, summary.mdd)
    _print_choice_usage("Delta combo usage", oos_combos, unique(oos_combos);
        formatter=c -> @sprintf("(%.2f, %.2f)", c[1], c[2]))
    _print_choice_usage("Wing usage", oos_wings, wing_widths; formatter=w -> @sprintf("\$%.1f", w))
    print_monthly_sharpe("cross-tenor OOS", oos_dates, oos_pnls)
    save_cumulative_pnl(oos_dates, oos_pnls, joinpath(run_dir, "cumulative.png");
        title="$symbol - train (delta, wing) on $tenor_train_str, apply on $tenor_test_str",
        label="cross-tenor")
    return (run_dir=run_dir, dates=oos_dates, pnls=oos_pnls, combos=oos_combos, wings=oos_wings)
end
