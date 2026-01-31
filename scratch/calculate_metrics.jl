using VolSurfaceAnalysis
using CSV, DataFrames

# Load results
csv_path = joinpath(@__DIR__, "polygon_short_strangle_pnl.csv")
df = CSV.read(csv_path, DataFrame)

# Margin Assumption (Reg T Strangle on SPY)
MARGIN_PER_STRANGLE = 12000.0

positions = Position[]
pnls = Union{Missing,Float64}[]

for row in eachrow(df)
    trade = Trade(Underlying("SPY"), 0.0, DateTime(row.ExpiryDate), Call; direction=1, quantity=1.0)
    pos = Position(trade, 0.0, 0.0, DateTime(row.EntryTime))
    push!(positions, pos)
    push!(pnls, Float64(row.PnL))
end

metrics = performance_metrics(positions, pnls; margin_per_trade=MARGIN_PER_STRANGLE)

println("--- Performance Metrics ---")
println("Margin Assumption: \$$(round(MARGIN_PER_STRANGLE, digits=0)) per strangle")
println("Total P&L: \$$(round(metrics.total_pnl, digits=2))")
if metrics.total_roi !== missing
    println("Total ROI (on fixed capital): $(round(metrics.total_roi * 100, digits=2))%")
end
if metrics.duration_days !== missing && metrics.duration_years !== missing
    println("Duration: $(metrics.duration_days) days ($(round(metrics.duration_years, digits=3)) years)")
end
if metrics.annualized_roi_simple !== missing
    println("Annualized ROI (Simple): $(round(metrics.annualized_roi_simple * 100, digits=2))%")
end
if metrics.annualized_roi_cagr !== missing
    println("Annualized ROI (CAGR):   $(round(metrics.annualized_roi_cagr * 100, digits=2))%")
end
if metrics.avg_return !== missing && metrics.volatility !== missing
    println("Avg Daily Return: $(round(metrics.avg_return * 100, digits=3))%")
    println("Daily Volatility: $(round(metrics.volatility * 100, digits=3))%")
end
if metrics.sharpe !== missing
    println("Sharpe Ratio: $(round(metrics.sharpe, digits=2))")
end
if metrics.win_rate !== missing
    println("Win Rate: $(round(metrics.win_rate * 100, digits=1))%")
end
if metrics.sortino !== missing
    println("Sortino Ratio: $(round(metrics.sortino, digits=2))")
end
