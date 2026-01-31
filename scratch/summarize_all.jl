using CSV, DataFrames, Statistics, Dates, Printf

# Summary Script
results_dir = @__DIR__

files = readdir(results_dir)
pnl_files = filter(f -> startswith(f, "results_") && endswith(f, "_pnl.csv"), files)

if isempty(pnl_files)
    println("No result files found in $results_dir")
    exit(0)
end

println("=" ^ 80)
println(rpad("SYMBOL", 10) * rpad("TOTAL P&L", 15) * rpad("WIN RATE", 12) * "NOTES")
println("=" ^ 80)

summary_data = []

for file in pnl_files
    # Parse symbol from "results_SYMBOL_pnl.csv"
    m = match(r"results_(.*)_pnl.csv", file)
    symbol = m === nothing ? "UNKNOWN" : m[1]
    
    df = CSV.read(joinpath(results_dir, file), DataFrame)
    
    if nrow(df) == 0
        push!(summary_data, (Symbol=symbol, PnL=0.0, WinRate=0.0))
        continue
    end
    
    total_pnl = sum(df.PnL)
    win_rate = count(x -> x > 0, df.PnL) / nrow(df)
    
    push!(summary_data, (Symbol=symbol, PnL=total_pnl, WinRate=win_rate))
    
    @printf "%-10s \$%-14.2f %-11.1f%%\n" symbol total_pnl (win_rate*100)
end

println("=" ^ 80)

# Total Portfolio
total_portfolio_pnl = sum(d.PnL for d in summary_data)
println("TOTAL PORTFOLIO P&L: \$$(round(total_portfolio_pnl, digits=2))")
