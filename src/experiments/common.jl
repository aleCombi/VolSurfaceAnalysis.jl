# Shared experiment utilities used by orchestration scripts.

using Dates
using Printf
using Statistics

const Fold = @NamedTuple begin
    idx::Int
    train_start::Date
    train_end::Date
    test_start::Date
    test_end::Date
    train_mask::BitVector
    test_mask::BitVector
end

function make_run_dir(base_dir::AbstractString, prefix::AbstractString; timestamp=now())
    run_ts = Dates.format(timestamp, "yyyymmdd_HHMMSS")
    run_dir = joinpath(base_dir, "runs", "$(prefix)_$run_ts")
    mkpath(run_dir)
    return run_dir
end

function parse_tenor(s::AbstractString)
    text = lowercase(strip(s))
    if endswith(text, "d")
        return Day(parse(Int, text[1:end-1]))
    end
    if endswith(text, "h")
        return Hour(parse(Int, text[1:end-1]))
    end
    error("Unknown tenor: $s")
end

function finite_values(values)::Vector{Float64}
    out = Float64[]
    for v in values
        (v === missing || v === nothing) && continue
        fv = Float64(v)
        isfinite(fv) || continue
        push!(out, fv)
    end
    return out
end

function annualized_sharpe(values; annualization::Real=252, empty=-Inf, zero_std=0.0)
    vals = finite_values(values)
    isempty(vals) && return empty
    s = std(vals)
    return s > 0 ? mean(vals) / s * sqrt(annualization) : zero_std
end

function max_drawdown(values)
    vals = finite_values(values)
    isempty(vals) && return 0.0
    cum = cumsum(vals)
    running_max = accumulate(max, cum)
    return max(0.0, -minimum(cum .- running_max))
end

function pnl_summary(values; annualization::Real=252)
    vals = finite_values(values)
    isempty(vals) && return (n=0, total=0.0, sharpe=0.0, mdd=0.0, avg=0.0)
    return (
        n=length(vals),
        total=sum(vals),
        sharpe=annualized_sharpe(vals; annualization=annualization, empty=0.0),
        mdd=max_drawdown(vals),
        avg=mean(vals),
    )
end

function cvar_left(values; alpha::Real=0.05)
    vals = sort(finite_values(values))
    isempty(vals) && return 0.0
    n_tail = max(1, floor(Int, alpha * length(vals)))
    return mean(vals[1:n_tail])
end

function pnl_distribution_stats(values)
    vals = finite_values(values)
    n = length(vals)
    if n == 0
        return (
            trades=0, win_rate=0.0, mean_pnl=0.0, std_pnl=0.0,
            min_pnl=0.0, p1_pnl=0.0, p5_pnl=0.0, p10_pnl=0.0, p25_pnl=0.0,
            median_pnl=0.0, p75_pnl=0.0, p90_pnl=0.0, p95_pnl=0.0,
            p99_pnl=0.0, max_pnl=0.0, total_pnl=0.0,
            avg_win=0.0, avg_loss=0.0, skewness=0.0,
        )
    end

    wins = filter(>(0), vals)
    losses = filter(<=(0), vals)
    avg = mean(vals)
    s = n > 1 ? std(vals) : 0.0
    skewness = s > 0 ? mean((vals .- avg).^3) / s^3 : 0.0

    return (
        trades=n,
        win_rate=length(wins) / n,
        mean_pnl=avg,
        std_pnl=s,
        min_pnl=minimum(vals),
        p1_pnl=quantile(vals, 0.01),
        p5_pnl=quantile(vals, 0.05),
        p10_pnl=quantile(vals, 0.10),
        p25_pnl=quantile(vals, 0.25),
        median_pnl=quantile(vals, 0.50),
        p75_pnl=quantile(vals, 0.75),
        p90_pnl=quantile(vals, 0.90),
        p95_pnl=quantile(vals, 0.95),
        p99_pnl=quantile(vals, 0.99),
        max_pnl=maximum(vals),
        total_pnl=sum(vals),
        avg_win=isempty(wins) ? 0.0 : mean(wins),
        avg_loss=isempty(losses) ? 0.0 : mean(losses),
        skewness=skewness,
    )
end

function build_folds(
    dates::AbstractVector{Date};
    train_days::Integer,
    test_days::Integer,
    step_days::Integer,
    min_train::Integer=0,
    min_test::Integer=0,
)
    folds = Fold[]
    isempty(dates) && return folds

    test_start = dates[1] + Day(train_days)
    last_d = dates[end]
    idx = 0
    while test_start <= last_d
        test_end = test_start + Day(test_days) - Day(1)
        train_start = test_start - Day(train_days)
        train_end = test_start - Day(1)
        train_mask = (dates .>= train_start) .& (dates .<= train_end)
        test_mask = (dates .>= test_start) .& (dates .<= test_end)

        if sum(train_mask) >= min_train && sum(test_mask) >= min_test
            idx += 1
            push!(folds, (
                idx=idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_mask=train_mask,
                test_mask=test_mask,
            ))
        end
        test_start += Day(step_days)
    end
    return folds
end

function _csv_cell(v)
    if v === missing
        return ""
    elseif v isa AbstractString
        escaped = replace(v, "\"" => "\"\"")
        return any(occursin(c, escaped) for c in (",", "\"", "\n", "\r")) ? "\"$escaped\"" : escaped
    elseif v isa AbstractFloat
        return isfinite(v) ? @sprintf("%.10g", v) : string(v)
    else
        return string(v)
    end
end

function write_namedtuple_csv(path::AbstractString, rows; columns=nothing)
    mkpath(dirname(path))
    if columns === nothing
        columns = isempty(rows) ? Symbol[] : collect(propertynames(first(rows)))
    else
        columns = collect(columns)
    end

    open(path, "w") do io
        println(io, join(string.(columns), ","))
        for row in rows
            println(io, join((_csv_cell(getfield(row, col)) for col in columns), ","))
        end
    end
    return path
end
