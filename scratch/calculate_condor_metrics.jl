using CSV, DataFrames, Statistics, Dates

# Load results
pnl_path = joinpath(@__DIR__, "polygon_iron_condor_pnl.csv")
trades_path = joinpath(@__DIR__, "polygon_iron_condor_trades.csv")

if !isfile(pnl_path) || !isfile(trades_path)
    println("Files not found yet. Run backtest first.")
    exit(1)
end

df_pnl = CSV.read(pnl_path, DataFrame)
df_trades = CSV.read(trades_path, DataFrame)

println("--- Iron Condor Performance Metrics ---")

# Calculate Dynamic Margin (Max Risk) per trade
# Filter for Long and Short legs
# Assuming 4 legs per EntryTime/Expiry pair
# We need to find the specific width for each entry.

# Group by EntryTime
grouped = groupby(df_trades, [:EntryTime, :Expiry])
margins = Float64[]

for gdf in grouped
    # Expect 4 legs: 2 Short (-1), 2 Long (1)
    # Strikes
    shorts = filter(r -> r.Direction == -1, gdf)
    longs = filter(r -> r.Direction == 1, gdf)
    
    if isempty(shorts) || isempty(longs)
        continue
    end
    
    # Put Wing Width: Short Put K - Long Put K
    # Call Wing Width: Long Call K - Short Call K
    
    short_put = filter(r -> r.Type == "Put", shorts)
    long_put = filter(r -> r.Type == "Put", longs)
    
    short_call = filter(r -> r.Type == "Call", shorts)
    long_call = filter(r -> r.Type == "Call", longs)
    
    risk_put = 0.0
    risk_call = 0.0
    
    if !isempty(short_put) && !isempty(long_put)
        # Risk = Strike Diff * 100
        # Access first row: long_put.Strike[1]
        width = short_put.Strike[1] - long_put.Strike[1]
        risk_put = max(0.0, width * 100)
    end
    
    if !isempty(short_call) && !isempty(long_call)
        width = long_call.Strike[1] - short_call.Strike[1]
        risk_call = max(0.0, width * 100)
    end
    
    # Iron Condor Risk = Max(Put Risk, Call Risk) - Credit Received?
    # Reg T Margin is roughly Max(Put Risk, Call Risk).
    # Conservative: Sum of Widths if one side challenged? No, typically one side.
    # We'll use Max(Put Width, Call Width).
    
    margin = max(risk_put, risk_call)
    push!(margins, margin)
end

avg_margin = mean(margins)
total_margin_allocation = sum(margins) # If we allocated capital freshly each time? No.
# ROI on Capital: Return / Average Required Capital?
# Or Return / Max Capital Utilized?
# Assuming 1 contract strategy, capital required is simply the margin per trade.
# Effectively "Return on Margin" per trade averaged?
# Or Total P&L / Average Margin (as base capital).

total_pnl = sum(df_pnl.PnL)
roi_on_avg_margin = total_pnl / avg_margin # Total P&L vs Capital Unit Size

# Annualized
start_date = minimum(df_pnl.EntryDate)
end_date = maximum(df_pnl.EntryDate)
days = Dates.value(end_date - start_date)
years = days / 365.25

annualized_return_pct = (total_pnl / avg_margin) / years

println("Avg Margin (Defined Risk): \$$(round(avg_margin, digits=2))")
println("Total P&L: \$$(round(total_pnl, digits=2))")
println("Total Return (on Avg Margin): $(round(roi_on_avg_margin * 100, digits=2))%")
println("Annualized Return: $(round(annualized_return_pct * 100, digits=2))%")

# Sharpe
# Daily Returns = PnL / Avg Margin
daily_returns = df_pnl.PnL ./ avg_margin
avg_ret = mean(daily_returns)
std_ret = std(daily_returns)
sharpe = (avg_ret / std_ret) * sqrt(252)

println("Sharpe Ratio: $(round(sharpe, digits=2))")
println("Win Rate: $(round(count(x->x>0, df_pnl.PnL)/nrow(df_pnl)*100, digits=1))%")
