# scripts/compare_rolling_implementations.jl
#
# Compare ground-truth `condor_rolling_wing.jl` output against the
# `IronCondorStrategy + RollingWingCondorSelector + backtest_strategy` output.

using Pkg; Pkg.activate(@__DIR__)
using DataFrames, Printf, Statistics, Plots, Dates

function read_csv(path)
    lines = readlines(path)
    header = split(lines[1], ',')
    n = length(lines) - 1
    dates = Vector{Date}(undef, n)
    wings = Vector{Float64}(undef, n)
    pnls  = Vector{Float64}(undef, n)
    for i in 1:n
        parts = split(lines[i+1], ',')
        dates[i] = Date(parts[1])
        wings[i] = parse(Float64, parts[2])
        pnls[i]  = parse(Float64, parts[3])
    end
    return DataFrame(date=dates, chosen_wing=wings, oos_pnl_usd=pnls)
end

# Pass via env vars or CLI args
GROUND_CSV = ARGS[1]
NEW_CSV    = ARGS[2]

println("Ground truth: $GROUND_CSV")
println("Via backtest: $NEW_CSV")

a = read_csv(GROUND_CSV)
b = read_csv(NEW_CSV)

println("\nGround truth: $(nrow(a)) rows, dates $(minimum(a.date)) → $(maximum(a.date))")
println("Via backtest: $(nrow(b)) rows, dates $(minimum(b.date)) → $(maximum(b.date))")

# Inner join on date
joined = innerjoin(a, b; on=:date, makeunique=true)
rename!(joined, :chosen_wing => :wing_a, :oos_pnl_usd => :pnl_a,
                :chosen_wing_1 => :wing_b, :oos_pnl_usd_1 => :pnl_b)

println("\nJoined: $(nrow(joined)) rows  (A had $(nrow(a)) - B had $(nrow(b)))")
in_a_only = setdiff(a.date, b.date)
in_b_only = setdiff(b.date, a.date)
println("  Dates in A but not B: $(length(in_a_only))   in B but not A: $(length(in_b_only))")
if !isempty(in_a_only)
    println("    A-only sample: ", first(sort(collect(in_a_only)), min(5, length(in_a_only))))
end
if !isempty(in_b_only)
    println("    B-only sample: ", first(sort(collect(in_b_only)), min(5, length(in_b_only))))
end

# Wing-choice agreement
wing_match = sum(joined.wing_a .== joined.wing_b)
println("\nWing-choice agreement: $wing_match / $(nrow(joined))   ($(round(100*wing_match/nrow(joined), digits=1))%)")

# PnL deltas
joined.pnl_diff = joined.pnl_b .- joined.pnl_a
total_a = sum(joined.pnl_a); total_b = sum(joined.pnl_b)
@printf "\n  Total PnL  A: %+.2f   B: %+.2f   diff: %+.2f  (%+.2f%%)\n" total_a total_b (total_b-total_a) 100*(total_b-total_a)/abs(total_a)
@printf "  Per-trade max|diff|: %.4f   RMSE: %.4f   median|diff|: %.4f\n" maximum(abs.(joined.pnl_diff)) sqrt(mean(joined.pnl_diff .^ 2)) median(abs.(joined.pnl_diff))

n_close   = sum(abs.(joined.pnl_diff) .< 0.01)
n_within1 = sum(abs.(joined.pnl_diff) .< 1.00)
@printf "  Rows with |diff| < 0.01 USD: %d (%.1f%%)\n" n_close 100*n_close/nrow(joined)
@printf "  Rows with |diff| < 1.00 USD: %d (%.1f%%)\n" n_within1 100*n_within1/nrow(joined)

# Sharpe of each
sharpe_of(v) = std(v) > 0 ? mean(v)/std(v)*sqrt(252) : 0.0
sh_a = sharpe_of(joined.pnl_a); sh_b = sharpe_of(joined.pnl_b)
@printf "\n  Sharpe   A: %+.3f   B: %+.3f\n" sh_a sh_b

# Show worst rows
sort!(joined, :pnl_diff; by=abs, rev=true)
println("\n  Top 10 worst per-row diffs:")
@printf "    %-10s  %5s  %5s  %+10s  %+10s  %+10s\n" "date" "wA" "wB" "pnl_A" "pnl_B" "diff"
for i in 1:min(10, nrow(joined))
    r = joined[i, :]
    @printf "    %-10s  %5.1f  %5.1f  %+10.3f  %+10.3f  %+10.3f\n" r.date r.wing_a r.wing_b r.pnl_a r.pnl_b r.pnl_diff
end

# Cumulative-PnL plot
sort!(joined, :date)
p = plot(joined.date, cumsum(joined.pnl_a);
    label="A (ground truth)", lw=2, color=:steelblue,
    xlabel="date", ylabel="cumulative PnL (USD)",
    title="Comparison — rolling-wing OOS cumulative PnL",
    size=(1100, 500),
)
plot!(p, joined.date, cumsum(joined.pnl_b); label="B (via backtest_strategy)", lw=2, color=:darkorange, ls=:dash)
hline!(p, [0]; color=:gray, ls=:dash, label="")
out_dir = dirname(NEW_CSV)
savefig(p, joinpath(out_dir, "comparison_cumulative.png"))
println("\n  Saved comparison plot: $(joinpath(out_dir, \"comparison_cumulative.png\"))")
