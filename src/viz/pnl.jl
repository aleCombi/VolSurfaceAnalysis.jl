using RecipesBase

@recipe function f(series::PnLSeries)
    isempty(series.pnl) &&
        throw(ArgumentError("cannot plot empty PnL series"))
    title  --> "Equity curve"
    xlabel --> "Time"
    ylabel --> "Cumulative PnL (USD)"
    legend --> false
    series.timestamps, equity_curve(series)
end
