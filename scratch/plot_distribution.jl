using CSV, DataFrames, Plots, Statistics, Dates

# Load data
csv_path = joinpath(@__DIR__, "polygon_short_strangle_pnl.csv")
df = CSV.read(csv_path, DataFrame)

# Calculate statistics
pnl = df.PnL
mean_pnl = mean(pnl)
median_pnl = median(pnl)
min_pnl = minimum(pnl)
max_pnl = maximum(pnl)
win_rate = count(x -> x > 0, pnl) / length(pnl)
total_pnl = sum(pnl)
count_trades = length(pnl)

# Create text for annotation
stats_text = """
Count: $count_trades
Total P&L: \$$(round(total_pnl, digits=2))
Mean: \$$(round(mean_pnl, digits=2))
Median: \$$(round(median_pnl, digits=2))
Win Rate: $(round(win_rate * 100, digits=1))%
Min: \$$(round(min_pnl, digits=2))
Max: \$$(round(max_pnl, digits=2))
"""

# Plotting
p = histogram(pnl, bins=40, label="Frequency", 
              title="P&L Distribution per Strangle",
              xlabel="P&L (USD)", ylabel="Count",
              color=:steelblue, alpha=0.7, linecolor=:white,
              legend=:topright, size=(800, 500), margin=5Plots.mm)

# Add reference lines
vline!(p, [0], color=:red, linestyle=:dash, linewidth=2, label="Breakeven (\$0)")
vline!(p, [mean_pnl], color=:green, linewidth=2, label="Mean (\$$(round(mean_pnl, digits=2)))")
vline!(p, [median_pnl], color=:orange, linestyle=:dot, linewidth=2, label="Median (\$$(round(median_pnl, digits=2)))")

# Add annotation
# Position text in top-left or relative to data range
x_pos = min_pnl + (max_pnl - min_pnl) * 0.05
y_max = 0 # will determine from histogram count... hard to know without computing.
# Text annotation syntax: annotate!(x, y, text(string, size, alignment))
# A clearer way is to put it in the title or outside. Or just rely on legend.
# Let's put it in the top-left area. 
# We need to estimate y-scale. Assuming max frequency ~30-50 based on 260 trades / 40 bins = 6 per bin avg... peak might be 20-30.
# Safest is to rely on the legend for the key stats, and maybe print the detailed block in console.
# Or use `annotate!(:topleft, ...)` if supported. Plots.jl annotation is coordinate based.

# Let's try to pass the stats text as a separate label in the legend? No.
# We'll stick to lines in legend.

title!(p, "P&L Distribution (Short Strangle)\nN=$count_trades | Win Rate=$(round(win_rate*100, digits=1))% | Avg=$(round(mean_pnl, digits=2))")

# Save
output_path = joinpath(@__DIR__, "plots", "strangle_pnl_distribution_enhanced.png")
mkpath(dirname(output_path))
savefig(p, output_path)
println("Enhanced plot saved to: $output_path")
