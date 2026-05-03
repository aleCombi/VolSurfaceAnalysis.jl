# Thin workflow wrapper for the rolling short-strangle experiment.

using Dates

function run_strangle_rolling_experiment(;
    output_root::AbstractString,
    symbol::AbstractString=get(ENV, "SYM", "SPY"),
    start_date::Date=Date(get(ENV, "START_DATE", "2014-06-02")),
    end_date::Date=Date(get(ENV, "END_DATE", "2026-03-27")),
    entry_time::Time=Time(parse(Int, get(ENV, "ENTRY_HOUR", "14")), 0),
    expiry_interval=Day(parse(Int, get(ENV, "EXPIRY_DAYS", "1"))),
    max_tau_days::Real=parse(Float64, get(ENV, "MAX_TAU_DAYS", "2.0")),
    spread_lambda::Real=parse(Float64, get(ENV, "SPREAD_LAMBDA", "0.7")),
    rate::Real=parse(Float64, get(ENV, "RATE", "0.045")),
    div_yield::Real=parse(Float64, get(ENV, "DIV", "0.013")),
    put_deltas=collect(0.05:0.05:0.40),
    call_deltas=collect(0.05:0.05:0.40),
    train_days::Integer=parse(Int, get(ENV, "TRAIN_DAYS", "90")),
    test_days::Integer=parse(Int, get(ENV, "TEST_DAYS", "30")),
    step_days::Integer=parse(Int, get(ENV, "STEP_DAYS", "30")),
    z_values=parse.(Float64, strip.(split(get(ENV, "Z_VALUES", "0,0.1,0.3,1,3"), ","))),
    cvar_alpha::Real=parse(Float64, get(ENV, "CVAR_ALPHA", "0.05")),
    baseline_combo=(0.20, 0.05),
)
    mode_tag = length(z_values) == 1 ? "diag" : "reg"
    run_dir = make_run_dir(output_root, "strangle_rolling_$(mode_tag)_$(symbol)")
    println("Output: $run_dir   mode=$(mode_tag == "diag" ? "diagnostic" : "regularized sweep")")
    println("\n  $symbol  $start_date -> $end_date   strangle (no wings)")
    println("  grid: $(length(put_deltas))x$(length(call_deltas)) combos   train=$train_days d / test=$test_days d / step=$step_days d")
    println("  z values: $z_values   alpha=$cvar_alpha   baseline=$baseline_combo")

    println("\nLoading $symbol ...")
    (; source, sched) = polygon_parquet_source(symbol;
        start_date=start_date,
        end_date=end_date,
        entry_time=entry_time,
        rate=rate,
        div_yield=div_yield,
        spread_lambda=spread_lambda,
    )

    println("\nRunning rolling ensemble ...")
    ensemble = run_strangle_rolling_ensemble(source, sched, expiry_interval;
        put_deltas=put_deltas,
        call_deltas=call_deltas,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        rate=rate,
        div_yield=div_yield,
        max_tau_days=max_tau_days,
        z_values=z_values,
        cvar_alpha=cvar_alpha,
        baseline_combo=baseline_combo,
    )

    report_strangle_rolling(ensemble;
        run_dir=run_dir,
        baseline_combo=baseline_combo,
        title_prefix="$symbol strangle",
    )
    return (run_dir=run_dir, ensemble=ensemble)
end
