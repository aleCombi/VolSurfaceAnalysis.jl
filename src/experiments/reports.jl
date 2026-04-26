# Shared reporting and visualization helpers for experiment outputs.

using DataFrames
using Dates
using Plots
using Printf
using Statistics

_rows(rows) = rows isa AbstractDataFrame ? eachrow(rows) : rows
_field(row, field::Symbol) = row isa NamedTuple ? getfield(row, field) : row[field]

function lookup_grid_row(rows, row_key::Symbol, row_value, col_key::Symbol, col_value; filter_fn=Returns(true))
    for row in _rows(rows)
        filter_fn(row) || continue
        _field(row, row_key) == row_value || continue
        _field(row, col_key) == col_value || continue
        return row
    end
    return nothing
end

function format_grid_value(v; digits::Int=3, pct::Bool=false, signed::Bool=true)
    if pct
        return @sprintf("%6.1f%%", Float64(v))
    elseif digits == 0
        return @sprintf("%7.0f", Float64(v))
    elseif signed
        return @sprintf("%+7.*f", digits, Float64(v))
    else
        return @sprintf("%7.*f", digits, Float64(v))
    end
end

function print_metric_grid(
    title::AbstractString,
    rows,
    row_values,
    col_values;
    row_key::Symbol,
    col_key::Symbol,
    value_key::Symbol,
    row_label::AbstractString="",
    col_label_fn::Function=string,
    row_label_fn::Function=string,
    digits::Int=3,
    pct::Bool=false,
    signed::Bool=true,
    value_fn::Function=identity,
    filter_fn::Function=Returns(true),
)
    println("\n  -- $title --")
    @printf("  %8s", row_label)
    for cv in col_values
        @printf("  %7s", col_label_fn(cv))
    end
    println()
    @printf("  %8s", "-"^8)
    for _ in col_values
        @printf("  %7s", "-"^7)
    end
    println()

    for rv in row_values
        @printf("  %8s", row_label_fn(rv))
        for cv in col_values
            row = lookup_grid_row(rows, row_key, rv, col_key, cv; filter_fn=filter_fn)
            if row === nothing
                print("      n/a")
            else
                print("  ", format_grid_value(value_fn(_field(row, value_key)); digits=digits, pct=pct, signed=signed))
            end
        end
        println()
    end
end

function print_delta_grid(title, grid, put_deltas, call_deltas, field::Symbol; digits::Int=3, pct::Bool=false)
    print_metric_grid(
        title,
        grid,
        put_deltas,
        call_deltas;
        row_key=:PutDelta,
        col_key=:CallDelta,
        value_key=field,
        row_label="P\\C",
        col_label_fn=x -> @sprintf("%.2f", x),
        row_label_fn=x -> @sprintf("%.2f", x),
        digits=digits,
        pct=pct,
    )
end

function grid_matrix(rows, row_values, col_values; row_key::Symbol, col_key::Symbol, value_key::Symbol)
    mat = fill(NaN, length(row_values), length(col_values))
    for (i, rv) in enumerate(row_values), (j, cv) in enumerate(col_values)
        row = lookup_grid_row(rows, row_key, rv, col_key, cv)
        row === nothing && continue
        mat[i, j] = Float64(_field(row, value_key))
    end
    return mat
end

function save_delta_heatmap(
    grid,
    put_deltas,
    call_deltas,
    field::Symbol,
    title::AbstractString,
    path::AbstractString;
    color=:RdYlGn,
    annotation_digits::Int=2,
)
    mat = grid_matrix(grid, put_deltas, call_deltas;
        row_key=:PutDelta, col_key=:CallDelta, value_key=field)
    annotations = [
        (j, i, text(isfinite(mat[i, j]) ? round(mat[i, j], digits=annotation_digits) : "n/a", 8, :black))
        for i in 1:length(put_deltas), j in 1:length(call_deltas)
    ]

    p = heatmap(
        string.(call_deltas),
        string.(put_deltas),
        mat;
        title=title,
        xlabel="Call delta",
        ylabel="Put delta",
        color=color,
        size=(600, 500),
        annotate=vec(annotations),
    )
    mkpath(dirname(path))
    savefig(p, path)
    return path
end

function print_monthly_sharpe(label::AbstractString, dates, pnls; min_count::Int=3)
    println("\n  -- Per-month Sharpe ($label) --")
    @printf("  %-9s  %5s  %+10s  %+8s\n", "month", "n", "totalPnL", "Sharpe")
    println("  " * "-"^46)

    yms = yearmonth.(dates)
    for ym in sort(unique(yms))
        mask = [x == ym for x in yms]
        vals = finite_values(pnls[mask])
        n = length(vals)
        n < min_count && continue
        @printf("  %4d-%02d   %5d  %+10.0f  %+8.2f\n",
            ym[1], ym[2], n, sum(vals), annualized_sharpe(vals; empty=0.0))
    end
end

function save_cumulative_pnl(
    dates,
    pnls,
    path::AbstractString;
    title::AbstractString,
    label::AbstractString="cumulative",
    color=:steelblue,
    size=(1100, 500),
)
    xs, ys = profit_curve(dates, pnls)
    isempty(xs) && return path
    p = plot(xs, cumsum(ys);
        xlabel="date",
        ylabel="cumulative PnL (USD)",
        title=title,
        label=label,
        lw=2,
        color=color,
        size=size)
    hline!(p, [0]; color=:gray, ls=:dash, label="")
    mkpath(dirname(path))
    savefig(p, path)
    return path
end

function save_cumulative_pnl_comparison(
    dates,
    pnls,
    path::AbstractString;
    title::AbstractString,
    label::AbstractString,
    reference_dates=nothing,
    reference_pnls=nothing,
    reference_label::AbstractString="reference",
    color=:steelblue,
    reference_color=:darkorange,
    reference_style=:dash,
    size=(1100, 500),
)
    xs, ys = profit_curve(dates, pnls)
    isempty(xs) && return path
    p = plot(xs, cumsum(ys);
        xlabel="date",
        ylabel="cumulative PnL (USD)",
        title=title,
        label=label,
        lw=2,
        color=color,
        size=size)

    if reference_dates !== nothing && reference_pnls !== nothing
        rx, ry = profit_curve(reference_dates, reference_pnls)
        if !isempty(rx)
            plot!(p, rx, cumsum(ry); label=reference_label, lw=2, color=reference_color, ls=reference_style)
        end
    end

    hline!(p, [0]; color=:gray, ls=:dash, label="")
    mkpath(dirname(path))
    savefig(p, path)
    return path
end

function save_fold_choice_scatter(
    dates,
    choices,
    path::AbstractString;
    title::AbstractString,
    ylabel::AbstractString,
    color=:darkorange,
    size=(1100, 350),
)
    p = scatter(dates, choices;
        xlabel="test-window start",
        ylabel=ylabel,
        title=title,
        ms=8,
        color=color,
        label="",
        size=size)
    mkpath(dirname(path))
    savefig(p, path)
    return path
end
