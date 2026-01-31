using VolSurfaceAnalysis
using Dates
using Statistics
using DuckDB
using DataFrames

# Configuration
const POLYGON_ROOT = raw"C:\repos\DeribitVols\data\massive_parquet\minute_aggs"
const DATE = Date(2024, 6, 3) # A Monday

function analyze_spreads(date::Date)
    y = year(date)
    m = lpad(month(date), 2, '0')
    d = lpad(day(date), 2, '0')
    search_path = joinpath(POLYGON_ROOT, "date=$y-$m-$d", "*", "*.parquet")
    
    # Handle Windows paths for DuckDB
    path_sql = replace(search_path, "\\" => "/")
    
    println("Querying data for $date using DuckDB...")
    println("Path: $path_sql")
    
    # Connect
    con = DuckDB.DBInterface.connect(DuckDB.DB, ":memory:")
    
    # Query all SPY options for the day
    query = """
    SELECT 
        ticker, 
        open, high, low, close, 
        volume, transactions
    FROM '$path_sql'
    WHERE ticker LIKE '%SPY%'
    """
    
    try
        df = DuckDB.DBInterface.execute(con, query) |> DataFrame
        println("Loaded $(nrow(df)) rows")
        return df
    catch e
        println("Error loading data: $e")
        return DataFrame()
    end
end

df = analyze_spreads(DATE)

if isempty(df)
    println("No data found.")
    exit(1)
end

# Calculate spreads
spreads = Float64[]
prices = Float64[]

for row in eachrow(df)
    mid = (row.high + row.low) / 2
    if mid > 0.50 # Filter for relevant options (avoid penny options)
        spread = row.high - row.low
        push!(spreads, spread)
        push!(prices, mid)
    end
end

if isempty(spreads)
    println("No valid options found > \$0.50")
    exit(0)
end

avg_spread = mean(spreads)
median_spread = median(spreads)
avg_price = mean(prices)
pct_spread = avg_spread / avg_price * 100

println("-" ^ 40)
println("SYNTHETIC SPREAD ANALYSIS (SPY Options)")
println("Date: $DATE")
println("Assumption: Spread = High - Low (Minute Bar)")
println("-" ^ 40)
println("Count: $(length(spreads)) options (Prize > \$0.50)")
println("Avg Price:      \$$(round(avg_price, digits=2))")
println("Avg Range (High-Low): \$$(round(avg_spread, digits=3))")
println("Median Range:         \$$(round(median_spread, digits=3))")
println("Effective Spread Cost: $(round(pct_spread, digits=2))% of price")
println("-" ^ 40)
println("Comparison:")
println("Real SPY Spread: ~\$0.01 - \$0.05")
println("Our Assumption:  ~\$$(round(avg_spread, digits=2))")
println("Conservativeness Factor: $(round(avg_spread / 0.03, digits=1))x")
