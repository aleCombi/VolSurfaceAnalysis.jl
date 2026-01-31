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
    pairs = Tuple{Date,Float64}[]
    for (d, v) in zip(dates, pnls)
        d === missing && continue
        v === missing && continue
        fv = Float64(v)
        isfinite(fv) || continue
        push!(pairs, (Date(d), fv))
    end
    sort!(pairs, by=first)
    return pairs
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
    pairs = _paired_series(dates, pnls)
    isempty(pairs) && return path

    xs = [p[1] for p in pairs]
    ys = cumsum([p[2] for p in pairs])

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
    save_pnl_and_equity_curve(dates, pnls, path; title_prefix="")

Save a combined equity curve + P&L distribution figure to `path`.
"""
function save_pnl_and_equity_curve(
    dates,
    pnls,
    path::AbstractString;
    title_prefix::AbstractString=""
)::AbstractString
    pairs = _paired_series(dates, pnls)
    isempty(pairs) && return path

    xs = [p[1] for p in pairs]
    vals = [p[2] for p in pairs]
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
