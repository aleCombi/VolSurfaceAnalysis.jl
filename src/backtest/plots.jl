# Backtest plotting helpers
# Save plots to disk (no interactive display)

using Dates
using Plots
using Statistics

function _clean_pnls(pnls)::Vector{Float64}
    vals = Float64[]
    for v in pnls
        v === missing && continue
        fv = Float64(v)
        isfinite(fv) || continue
        push!(vals, fv)
    end
    return vals
end

function _paired_series(dates, pnls)
    xs, ys = profit_curve(dates, pnls)
    return collect(zip(xs, ys))
end

"""
    save_pnl_distribution(pnls, path; bins=30, title="P&L Distribution")

Save a histogram of P&L values to `path`.
"""
function save_pnl_distribution(
    pnls,
    path::AbstractString;
    bins::Int=30,
    title::AbstractString="P&L Distribution"
)::AbstractString
    vals = _clean_pnls(pnls)
    isempty(vals) && return path

    p = histogram(
        vals,
        bins=bins,
        label="",
        title=title,
        xlabel="P&L (USD)",
        ylabel="Frequency",
        color=:steelblue,
        alpha=0.7
    )
    vline!(p, [0], color=:red, linestyle=:dash, label="Breakeven", linewidth=2)

    mkpath(dirname(path))
    savefig(p, path)
    return path
end

"""
    save_equity_curve(dates, pnls, path; title="Cumulative P&L")

Save an equity curve (cumulative P&L) to `path`.
"""
function save_equity_curve(
    dates,
    pnls,
    path::AbstractString;
    title::AbstractString="Cumulative P&L"
)::AbstractString
    xs, vals = profit_curve(dates, pnls)
    isempty(xs) && return path
    ys = cumsum(vals)

    p = plot(
        xs,
        ys,
        label="Equity Curve",
        title=title,
        xlabel="Date",
        ylabel="Total P&L (USD)",
        linewidth=2,
        color=:green,
        legend=:topleft
    )

    mkpath(dirname(path))
    savefig(p, path)
    return path
end

"""
    save_profit_curve(dates, pnls, path; title="Profit per Trade")

Save a per-trade profit curve (non-cumulative) to `path`.
"""
function save_profit_curve(
    dates,
    pnls,
    path::AbstractString;
    title::AbstractString="Profit per Trade"
)::AbstractString
    xs, ys = profit_curve(dates, pnls)
    isempty(xs) && return path

    p = plot(
        xs,
        ys,
        label="Profit",
        title=title,
        xlabel="Date",
        ylabel="P&L (USD)",
        linewidth=1.5,
        color=:darkorange,
        legend=:topleft
    )
    hline!(p, [0], color=:black, linestyle=:dash, label="", linewidth=1)

    mkpath(dirname(path))
    savefig(p, path)
    return path
end

"""
    save_spot_curve(spots, path; title="Spot Curve")

Save a spot price curve to `path`. `spots` is a Dict{DateTime,Float64}.
"""
function save_spot_curve(
    spots::AbstractDict{DateTime,Float64},
    path::AbstractString;
    title::AbstractString="Spot Curve"
)::AbstractString
    isempty(spots) && return path

    pairs = sort(collect(spots); by=first)
    xs = [Date(p[1]) for p in pairs]
    ys = [p[2] for p in pairs]

    p = plot(
        xs,
        ys,
        label="Spot",
        title=title,
        xlabel="Date",
        ylabel="Spot (USD)",
        linewidth=2,
        color=:navy,
        legend=:topleft
    )

    mkpath(dirname(path))
    savefig(p, path)
    return path
end

"""
    save_pnl_and_equity_curve(dates, pnls, path; title_prefix="")

Save a combined equity curve + P&L distribution figure to `path`.
"""
function save_pnl_and_equity_curve(
    dates,
    pnls,
    path::AbstractString;
    title_prefix::AbstractString=""
)::AbstractString
    xs, vals = profit_curve(dates, pnls)
    isempty(xs) && return path
    ys = cumsum(vals)

    title_eq = isempty(title_prefix) ? "Cumulative P&L" : "$title_prefix - Cumulative P&L"
    title_dist = isempty(title_prefix) ? "P&L Distribution" : "$title_prefix - P&L Distribution"

    p_eq = plot(
        xs,
        ys,
        label="Equity Curve",
        title=title_eq,
        xlabel="Date",
        ylabel="Total P&L (USD)",
        linewidth=2,
        color=:green,
        legend=:topleft
    )

    p_dist = histogram(
        vals,
        bins=30,
        label="",
        title=title_dist,
        xlabel="P&L (USD)",
        ylabel="Frequency",
        color=:steelblue,
        alpha=0.7
    )
    vline!(p_dist, [0], color=:red, linestyle=:dash, label="Breakeven", linewidth=2)

    p = plot(p_eq, p_dist, layout=(2, 1), size=(800, 800), margin=5Plots.mm)

    mkpath(dirname(path))
    savefig(p, path)
    return path
end
