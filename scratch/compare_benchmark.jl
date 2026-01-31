using CSV, DataFrames, Statistics, Dates, DuckDB, Printf

# Configuration
const SPOT_ROOT = raw"C:\repos\DeribitVols\data\massive_parquet\spot_1min"
const RESULTS_DIR = @__DIR__
# Dates from backtest log
const START_DATE = Date(2024, 1, 29)
const END_DATE = Date(2025, 7, 31)

const SYMBOLS = ["SPY", "QQQ", "IWM", "GLD", "SLV", "TLT", "USO", "HYG", "GDX", "XLE", "XLF"] 
# VIX excluded as it's not a spot asset you can buy directly (though VIXY/VIXM exist, let's skip for now)

# Helper: Scan available spot dates
function get_available_spot_dates()
    dates = Date[]
    if !isdir(SPOT_ROOT); return dates; end
    for d in readdir(SPOT_ROOT)
        m = match(r"date=(\d{4})-(\d{2})-(\d{2})", d)
        if m !== nothing
            push!(dates, Date(parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3])))
        end
    end
    return sort(dates)
end

const AVAILABLE_SPOT_DATES = get_available_spot_dates()

function get_nearest_date(target::Date)
    if isempty(AVAILABLE_SPOT_DATES); return nothing; end
    # Find index of first date >= target
    idx = searchsortedfirst(AVAILABLE_SPOT_DATES, target)
    if idx > length(AVAILABLE_SPOT_DATES)
        return AVAILABLE_SPOT_DATES[end]
    elseif idx == 1
        return AVAILABLE_SPOT_DATES[1]
    else
        d1 = AVAILABLE_SPOT_DATES[idx-1]
        d2 = AVAILABLE_SPOT_DATES[idx]
        if abs(target - d1) < abs(target - d2)
            return d1
        else
            return d2
        end
    end
end

function get_spot_price_robust(symbol::String, target_date::Date; is_end=false)
    date = get_nearest_date(target_date)
    if date === nothing; return missing; end
    
    date_str = Dates.format(date, "yyyy-mm-dd")
    path = joinpath(SPOT_ROOT, "date=$date_str", "symbol=$symbol", "data.parquet")
    
    # Try DuckDB
    con = DuckDB.DBInterface.connect(DuckDB.DB, ":memory:")
    path_sql = replace(path, "\\" => "/")
    
    try
        if !isfile(path); return missing; end
        
        # Sort logic: Start = Open (LIMIT 1), End = Close (DESC LIMIT 1)
        sort_order = is_end ? "DESC" : "ASC"
        col = is_end ? "close" : "open"
        
        query = "SELECT $col FROM '$path_sql' ORDER BY timestamp $sort_order LIMIT 1"
        df = DuckDB.DBInterface.execute(con, query) |> DataFrame
        if nrow(df) > 0
            return Float64(df[1, 1])
        end
    catch e
        return missing
    end
    return missing
end

results = []

println("=" ^ 80)
println(rpad("SYMBOL", 8) * rpad("STRATEGY ROI", 15) * rpad("B&H ROI", 15) * "DIFF (Alpha)")
println("=" ^ 80)

for symbol in SYMBOLS
    # 1. Calculate Strategy ROI
    trades_file = joinpath(RESULTS_DIR, "results_$(symbol)_trades.csv")
    pnl_file = joinpath(RESULTS_DIR, "results_$(symbol)_pnl.csv")
    
    if !isfile(trades_file) || !isfile(pnl_file)
        continue
    end
    
    df_trades = CSV.read(trades_file, DataFrame)
    df_pnl = CSV.read(pnl_file, DataFrame)
    
    if nrow(df_pnl) == 0; continue; end
    
    total_pnl = sum(df_pnl.PnL)
    
    # Calculate Avg Margin
    # Group by Expiry/Entry
    grouped = groupby(df_trades, [:EntryTime, :Expiry])
    margins = Float64[]
    for gdf in grouped
        shorts = filter(r -> r.Direction == -1, gdf)
        longs = filter(r -> r.Direction == 1, gdf)
        if isempty(shorts) || isempty(longs); continue; end
        
        risk = 0.0
        # Put side
        sp = filter(r -> r.Type == "Put", shorts)
        lp = filter(r -> r.Type == "Put", longs)
        if !isempty(sp) && !isempty(lp)
            risk = max(risk, (sp.Strike[1] - lp.Strike[1]) * 100)
        end
        # Call side
        sc = filter(r -> r.Type == "Call", shorts)
        lc = filter(r -> r.Type == "Call", longs)
        if !isempty(sc) && !isempty(lc)
            risk = max(risk, (lc.Strike[1] - sc.Strike[1]) * 100)
        end
        
        if risk > 0; push!(margins, risk); end
    end
    
    avg_margin = isempty(margins) ? 1.0 : mean(margins)
    strat_roi = total_pnl / avg_margin
    
    
    # 2. Calculate Benchmark ROI
    p_start = get_spot_price_robust(symbol, START_DATE; is_end=false)
    p_end = get_spot_price_robust(symbol, END_DATE; is_end=true)
    
    if ismissing(p_start) || ismissing(p_end)
        bench_roi = NaN
    else
        bench_roi = (p_end - p_start) / p_start
    end
    
    push!(results, (Symbol=symbol, StratROI=strat_roi, BenchROI=bench_roi))
    
    diff = strat_roi - bench_roi
    sign_str = diff > 0 ? "+" : ""
    
    @printf "%-8s %-14.1f%% %-14.1f%% %s%.1f%%\n" symbol (strat_roi*100) (bench_roi*100) sign_str (diff*100)
end

println("=" ^ 80)
