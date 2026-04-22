using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates, Printf, Statistics, DataFrames, Plots

# =============================================================================
# Configuration
# =============================================================================

START_DATE      = Date(2016, 3, 28)
END_DATE        = Date(2024, 1, 31)
ENTRY_TIME      = Time(12, 0)
EXPIRY_INTERVAL = Day(1)

SPREAD_LAMBDA   = 0.7
RATE            = 0.045
DIV_YIELD       = 0.013

PUT_DELTAS  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
CALL_DELTAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# =============================================================================
# Setup
# =============================================================================

run_ts = Dates.format(now(), "yyyymmdd_HHMMSS")
run_dir = joinpath(@__DIR__, "runs", "strangle_grid_$run_ts")
mkpath(run_dir)
println("Output: $run_dir")

println("\nLoading SPY data from $START_DATE to $END_DATE...")
(; source, sched) = polygon_parquet_source("SPY";
    start_date=START_DATE, end_date=END_DATE, entry_time=ENTRY_TIME,
    rate=RATE, div_yield=DIV_YIELD, spread_lambda=SPREAD_LAMBDA,
)
println("  $(length(sched)) entry timestamps")

# =============================================================================
# Strangle evaluation
# =============================================================================

# Result struct per trade
struct StrangleTrade
    date::Date
    put_delta::Float64
    call_delta::Float64
    pnl::Float64
    credit::Float64
    put_spread::Float64   # bid-ask spread on short put (relative)
    call_spread::Float64  # bid-ask spread on short call (relative)
    spot::Float64
    put_strike::Float64
    call_strike::Float64
end

println("\nRunning strangle grid: $(length(PUT_DELTAS))×$(length(CALL_DELTAS)) = $(length(PUT_DELTAS)*length(CALL_DELTAS)) cells...")
println("  (short + long strangles in single pass)")

# Pre-compute all strangles in a single pass through the data
# For each entry, evaluate all delta pairs at once (surfaces are cached)
short_trades = StrangleTrade[]
long_trades = StrangleTrade[]

n_entries = 0
each_entry(source, EXPIRY_INTERVAL, sched; clear_cache=true) do ctx, settlement
    ismissing(settlement) && return
    global n_entries += 1

    dctx = delta_context(ctx; rate=RATE, div_yield=DIV_YIELD)
    dctx === nothing && return
    spot = dctx.spot
    d = Date(ctx.surface.timestamp)

    for pd in PUT_DELTAS
        p_K = delta_strike(dctx, -abs(pd), Put)
        p_K === nothing && continue
        p_rec = find_record_at_strike(dctx.put_recs, p_K)
        p_rec === nothing && continue
        p_bid = extract_price(p_rec, :bid); p_bid === nothing && continue
        p_ask = extract_price(p_rec, :ask); p_ask === nothing && continue
        p_spread = let mid = (p_bid + p_ask) / 2
            mid > 0 ? (p_ask - p_bid) / mid : Inf
        end

        for cd in CALL_DELTAS
            c_K = delta_strike(dctx, abs(cd), Call)
            c_K === nothing && continue
            c_rec = find_record_at_strike(dctx.call_recs, c_K)
            c_rec === nothing && continue
            c_bid = extract_price(c_rec, :bid); c_bid === nothing && continue
            c_ask = extract_price(c_rec, :ask); c_ask === nothing && continue
            c_spread = let mid = (c_bid + c_ask) / 2
                mid > 0 ? (c_ask - c_bid) / mid : Inf
            end

            # Short strangle (sell at bid)
            short_pos = open_strangle_positions(ctx, p_K, c_K; direction=-1)
            if length(short_pos) == 2
                pnl = settle(short_pos, Float64(settlement))
                credit = -sum(entry_cost(p) for p in short_pos)
                push!(short_trades, StrangleTrade(
                    d, pd, cd, pnl, credit, p_spread, c_spread, spot, p_K, c_K))
            end

            # Long strangle (buy at ask)
            long_pos = open_strangle_positions(ctx, p_K, c_K; direction=+1)
            if length(long_pos) == 2
                pnl = settle(long_pos, Float64(settlement))
                debit = sum(entry_cost(p) for p in long_pos)
                push!(long_trades, StrangleTrade(
                    d, pd, cd, pnl, debit, p_spread, c_spread, spot, p_K, c_K))
            end
        end
    end

    if n_entries % 200 == 0
        @printf("  %d entries processed, %d short + %d long trades\r",
            n_entries, length(short_trades), length(long_trades))
    end
end

println("  $(n_entries) entries → $(length(short_trades)) short + $(length(long_trades)) long trades")
all_trades = short_trades  # keep backward compat for existing analysis

# =============================================================================
# Grid metrics
# =============================================================================

# Convert to DataFrame for easy grouping
df = DataFrame(
    Date = [t.date for t in all_trades],
    PutDelta = [t.put_delta for t in all_trades],
    CallDelta = [t.call_delta for t in all_trades],
    PnL = [t.pnl for t in all_trades],
    Credit = [t.credit for t in all_trades],
    PutSpread = [t.put_spread for t in all_trades],
    CallSpread = [t.call_spread for t in all_trades],
    Spot = [t.spot for t in all_trades],
    PutStrike = [t.put_strike for t in all_trades],
    CallStrike = [t.call_strike for t in all_trades],
)
df.Year = year.(df.Date)
df.Month = Dates.format.(df.Date, "yyyy-mm")
# PnL in USD (multiply by spot)
df.PnL_USD = df.PnL .* df.Spot
df.Credit_USD = df.Credit .* df.Spot

# Compute per-cell metrics
grid = combine(groupby(df, [:PutDelta, :CallDelta]),
    :PnL_USD => length => :Count,
    :PnL_USD => mean => :AvgPnL,
    :PnL_USD => sum => :TotalPnL,
    :PnL_USD => std => :StdPnL,
    :PnL_USD => (x -> count(>(0), x) / length(x) * 100) => :WinRate,
    :Credit_USD => mean => :AvgCredit,
    :PutSpread => (x -> mean(filter(isfinite, x))) => :AvgPutSpread,
    :CallSpread => (x -> mean(filter(isfinite, x))) => :AvgCallSpread,
    :PnL_USD => (x -> begin
        s = sort(x); n5 = max(1, floor(Int, 0.05*length(s)))
        mean(s[1:n5])
    end) => :CVaR5,
)
grid.Sharpe = [r.StdPnL > 0 ? r.AvgPnL / r.StdPnL * sqrt(252) : 0.0 for r in eachrow(grid)]
grid.AvgSpread = (grid.AvgPutSpread .+ grid.AvgCallSpread) ./ 2

sort!(grid, [:PutDelta, :CallDelta])

# =============================================================================
# Print grid tables
# =============================================================================

function print_grid(title, field; digits=3, pct=false)
    println("\n  ── $title ──")
    @printf("  %8s", "P\\C Δ")
    for cd in CALL_DELTAS
        @printf("  %7.2f", cd)
    end
    println()
    @printf("  %8s", "-"^8)
    for _ in CALL_DELTAS
        @printf("  %7s", "-"^7)
    end
    println()

    for pd in PUT_DELTAS
        @printf("  %8.2f", pd)
        for cd in CALL_DELTAS
            row = filter(r -> r.PutDelta == pd && r.CallDelta == cd, grid)
            if nrow(row) == 0
                print("      n/a")
            else
                val = row[1, field]
                s = pct ? @sprintf("%6.1f%%", val) : @sprintf("%+7.*f", digits, val)
                print("  ", s)
            end
        end
        println()
    end
end

println("\n", "=" ^ 70)
println("  STRANGLE P&L GRID — SPY $(START_DATE) to $(END_DATE)")
println("  Entry @ $(ENTRY_TIME), expiry = +1 day, spread_lambda=$(SPREAD_LAMBDA)")
println("=" ^ 70)

print_grid("Avg PnL (USD per trade)", :AvgPnL; digits=3)
print_grid("Win Rate (%)", :WinRate; digits=1, pct=true)
print_grid("Sharpe (annualized)", :Sharpe; digits=2)
print_grid("Avg Credit (USD)", :AvgCredit; digits=3)
print_grid("CVaR 5% (USD)", :CVaR5; digits=2)
print_grid("Avg Bid-Ask Spread (rel)", :AvgSpread; digits=3)
print_grid("Trade Count", :Count; digits=0)

# =============================================================================
# Best and worst cells
# =============================================================================

println("\n  ── Top 5 Cells by Sharpe ──")
top = sort(grid, :Sharpe, rev=true)[1:min(5, nrow(grid)), :]
@printf("  %-6s  %-6s  %8s  %8s  %8s  %8s  %8s\n",
    "PutΔ", "CallΔ", "Sharpe", "AvgPnL", "WinRate", "CVaR5", "Count")
for r in eachrow(top)
    @printf("  %-6.2f  %-6.2f  %+8.2f  %+8.3f  %7.1f%%  %+8.2f  %8d\n",
        r.PutDelta, r.CallDelta, r.Sharpe, r.AvgPnL, r.WinRate, r.CVaR5, r.Count)
end

println("\n  ── Bottom 5 Cells by Sharpe ──")
bot = sort(grid, :Sharpe)[1:min(5, nrow(grid)), :]
@printf("  %-6s  %-6s  %8s  %8s  %8s  %8s  %8s\n",
    "PutΔ", "CallΔ", "Sharpe", "AvgPnL", "WinRate", "CVaR5", "Count")
for r in eachrow(bot)
    @printf("  %-6.2f  %-6.2f  %+8.2f  %+8.3f  %7.1f%%  %+8.2f  %8d\n",
        r.PutDelta, r.CallDelta, r.Sharpe, r.AvgPnL, r.WinRate, r.CVaR5, r.Count)
end

# =============================================================================
# Year-by-year breakdown for top cells
# =============================================================================

println("\n  ── Year-by-Year for Top 3 Cells ──")
top3 = sort(grid, :Sharpe, rev=true)[1:min(3, nrow(grid)), :]
for r in eachrow(top3)
    pd, cd = r.PutDelta, r.CallDelta
    cell_df = filter(row -> row.PutDelta == pd && row.CallDelta == cd, df)
    println("\n  PutΔ=$(pd)  CallΔ=$(cd)  (Sharpe=$(round(r.Sharpe, digits=2)))")
    @printf("  %-6s  %6s  %10s  %8s  %8s\n", "Year", "Trades", "PnL", "AvgPnL", "WinRate")
    @printf("  %-6s  %6s  %10s  %8s  %8s\n", "-"^6, "-"^6, "-"^10, "-"^8, "-"^8)
    for yr in sort(unique(cell_df.Year))
        ydf = filter(row -> row.Year == yr, cell_df)
        nt = nrow(ydf)
        tp = sum(ydf.PnL_USD)
        ap = mean(ydf.PnL_USD)
        wr = count(>(0), ydf.PnL_USD) / nt * 100
        @printf("  %-6d  %6d  %+10.2f  %+8.3f  %7.1f%%\n", yr, nt, tp, ap, wr)
    end
end

# =============================================================================
# Heatmap plots
# =============================================================================

println("\n  Saving heatmaps...")

function save_heatmap(field, title, path; color=:RdYlGn)
    mat = zeros(length(PUT_DELTAS), length(CALL_DELTAS))
    for (i, pd) in enumerate(PUT_DELTAS)
        for (j, cd) in enumerate(CALL_DELTAS)
            row = filter(r -> r.PutDelta == pd && r.CallDelta == cd, grid)
            mat[i, j] = nrow(row) > 0 ? row[1, field] : NaN
        end
    end
    p = heatmap(
        string.(CALL_DELTAS), string.(PUT_DELTAS), mat,
        title=title, xlabel="Call Δ", ylabel="Put Δ",
        color=color, size=(600, 500),
        annotate=[(j, i, text(round(mat[i,j], digits=2), 8, :black))
                  for i in 1:length(PUT_DELTAS) for j in 1:length(CALL_DELTAS)])
    savefig(p, path)
    println("    $(basename(path))")
end

save_heatmap(:Sharpe, "Sharpe (annualized)", joinpath(run_dir, "sharpe_heatmap.png"))
save_heatmap(:AvgPnL, "Avg PnL (USD)", joinpath(run_dir, "avgpnl_heatmap.png"))
save_heatmap(:WinRate, "Win Rate (%)", joinpath(run_dir, "winrate_heatmap.png"); color=:YlGn)
save_heatmap(:AvgCredit, "Avg Credit (USD)", joinpath(run_dir, "credit_heatmap.png"); color=:YlOrRd)
save_heatmap(:CVaR5, "CVaR 5% (USD)", joinpath(run_dir, "cvar_heatmap.png"); color=:RdYlGn)
save_heatmap(:AvgSpread, "Avg Bid-Ask Spread", joinpath(run_dir, "spread_heatmap.png"); color=:YlOrRd)

# =============================================================================
# Historical evolution for selected cells
# =============================================================================

SELECTED = [
    (0.05, 0.05, "5Δ/5Δ (widest)"),       # Sharpe 1.59 — far OTM both sides, highest win rate
    (0.20, 0.05, "20Δp/5Δc (asym sweet)"), # Sharpe 1.37 — sells put premium, tight call
    (0.20, 0.20, "20Δ/20Δ (symmetric)"),   # Sharpe 1.08 — balanced, moderate risk
    (0.30, 0.30, "30Δ/30Δ (tightest)"),    # Sharpe 0.86 — near ATM, highest credit but worst risk-adj
    (0.05, 0.30, "5Δp/30Δc (inverted)"),   # Sharpe 0.52 — wide put / tight call, call risk dominates
]

println("\n", "=" ^ 70)
println("  HISTORICAL EVOLUTION — SELECTED CELLS")
println("=" ^ 70)

# --- Year-by-year Sharpe table ---
println("\n  ── Year-by-Year Sharpe ──")
all_years = sort(unique(df.Year))
@printf("  %-22s", "Cell")
for yr in all_years
    @printf("  %6d", yr)
end
println()
@printf("  %-22s", "-"^22)
for _ in all_years
    @printf("  %6s", "-"^6)
end
println()

for (pd, cd, label) in SELECTED
    cell = filter(r -> r.PutDelta == pd && r.CallDelta == cd, df)
    @printf("  %-22s", label)
    for yr in all_years
        ydf = filter(r -> r.Year == yr, cell)
        if nrow(ydf) < 10
            @printf("  %6s", "n/a")
        else
            pnls = ydf.PnL_USD
            s = std(pnls)
            sharpe = s > 0 ? mean(pnls) / s * sqrt(252) : 0.0
            @printf("  %+6.2f", sharpe)
        end
    end
    println()
end

# --- Year-by-year Avg PnL table ---
println("\n  ── Year-by-Year Avg PnL (USD) ──")
@printf("  %-22s", "Cell")
for yr in all_years
    @printf("  %8d", yr)
end
println()
@printf("  %-22s", "-"^22)
for _ in all_years
    @printf("  %8s", "-"^8)
end
println()

for (pd, cd, label) in SELECTED
    cell = filter(r -> r.PutDelta == pd && r.CallDelta == cd, df)
    @printf("  %-22s", label)
    for yr in all_years
        ydf = filter(r -> r.Year == yr, cell)
        if nrow(ydf) < 5
            @printf("  %8s", "n/a")
        else
            @printf("  %+8.1f", mean(ydf.PnL_USD))
        end
    end
    println()
end

# --- Cumulative PnL curves (overlaid) ---
println("\n  Saving historical plots...")

colors_sel = [:blue, :red, :green, :orange, :purple]
p_cum = plot(title="Cumulative PnL — Selected Strangles",
    xlabel="Date", ylabel="Cumulative PnL (USD)",
    legend=:topleft, size=(1000, 500), linewidth=2)

for (k, (pd, cd, label)) in enumerate(SELECTED)
    cell = sort(filter(r -> r.PutDelta == pd && r.CallDelta == cd, df), :Date)
    nrow(cell) == 0 && continue
    cum = cumsum(cell.PnL_USD)
    plot!(p_cum, cell.Date, cum, label=label, color=colors_sel[k])
end
hline!(p_cum, [0], color=:black, linestyle=:dash, label="", linewidth=1)
savefig(p_cum, joinpath(run_dir, "cumulative_pnl_selected.png"))
println("    cumulative_pnl_selected.png")

# --- Rolling Sharpe (63-trade window ≈ 3 months) ---
ROLL_WINDOW = 63

p_roll = plot(title="Rolling Sharpe ($(ROLL_WINDOW)-trade window)",
    xlabel="Date", ylabel="Sharpe (annualized)",
    legend=:topleft, size=(1000, 500), linewidth=1.5)

for (k, (pd, cd, label)) in enumerate(SELECTED)
    cell = sort(filter(r -> r.PutDelta == pd && r.CallDelta == cd, df), :Date)
    nrow(cell) < ROLL_WINDOW && continue
    pnls = cell.PnL_USD
    dates = cell.Date
    roll_sharpe = Float64[]
    roll_dates = Date[]
    for i in ROLL_WINDOW:length(pnls)
        window = pnls[i-ROLL_WINDOW+1:i]
        s = std(window)
        sh = s > 0 ? mean(window) / s * sqrt(252) : 0.0
        push!(roll_sharpe, sh)
        push!(roll_dates, dates[i])
    end
    plot!(p_roll, roll_dates, roll_sharpe, label=label, color=colors_sel[k])
end
hline!(p_roll, [0], color=:black, linestyle=:dash, label="", linewidth=1)
savefig(p_roll, joinpath(run_dir, "rolling_sharpe_selected.png"))
println("    rolling_sharpe_selected.png")

# --- Monthly seasonality (avg PnL by calendar month) ---
println("\n  ── Monthly Seasonality (Avg PnL by Calendar Month) ──")
@printf("  %-22s", "Cell")
for m in 1:12
    @printf("  %5s", Dates.monthabbr(m))
end
println()
@printf("  %-22s", "-"^22)
for _ in 1:12
    @printf("  %5s", "-"^5)
end
println()

for (pd, cd, label) in SELECTED
    cell = filter(r -> r.PutDelta == pd && r.CallDelta == cd, df)
    @printf("  %-22s", label)
    for m in 1:12
        mdf = filter(r -> month(r.Date) == m, cell)
        if nrow(mdf) < 5
            @printf("  %5s", "n/a")
        else
            @printf("  %+5.1f", mean(mdf.PnL_USD))
        end
    end
    println()
end

# --- Stylized payoff diagrams ---
println("\n  Saving payoff diagrams...")

# Use median spot and avg credit/strikes from each cell
p_payoff = plot(title="Short Strangle Payoff at Expiry",
    xlabel="Spot at Expiry (% of entry spot)", ylabel="P&L (USD)",
    legend=:bottomright, size=(1000, 600), linewidth=2)

for (k, (pd, cd, label)) in enumerate(SELECTED)
    cell = filter(r -> r.PutDelta == pd && r.CallDelta == cd, df)
    nrow(cell) == 0 && continue

    avg_spot = mean(cell.Spot)
    avg_credit = mean(cell.Credit) * avg_spot  # USD
    avg_put_K = mean(cell.PutStrike)
    avg_call_K = mean(cell.CallStrike)

    # Normalize strikes to % of spot
    put_pct = avg_put_K / avg_spot * 100
    call_pct = avg_call_K / avg_spot * 100

    # Payoff curve: spot range from -15% to +15%
    spot_range = range(85, 115, length=200)
    payoffs = Float64[]
    for s_pct in spot_range
        s = s_pct / 100 * avg_spot
        # Short put payoff: credit - max(K-S, 0)
        put_payoff = -max(avg_put_K - s, 0)
        # Short call payoff: credit - max(S-K, 0)
        call_payoff = -max(s - avg_call_K, 0)
        push!(payoffs, (put_payoff + call_payoff) * avg_spot / avg_spot + avg_credit)
    end
    # Actually: payoff = credit - intrinsic. Let me redo properly.
    # PnL of short strangle = credit - max(put_K - S, 0) - max(S - call_K, 0)
    # where credit and intrinsic are in USD
    payoffs_usd = [avg_credit - max(avg_put_K - (s/100*avg_spot), 0) - max((s/100*avg_spot) - avg_call_K, 0) for s in spot_range]

    plot!(p_payoff, collect(spot_range), payoffs_usd, label="$(label) [$(round(put_pct,digits=1))%-$(round(call_pct,digits=1))%]",
        color=colors_sel[k])
end
hline!(p_payoff, [0], color=:black, linestyle=:dash, label="", linewidth=1)
vline!(p_payoff, [100], color=:gray, linestyle=:dot, label="", linewidth=1)
savefig(p_payoff, joinpath(run_dir, "payoff_diagrams.png"))
println("    payoff_diagrams.png")

# =============================================================================
# Long strangle grid
# =============================================================================

println("\n", "=" ^ 70)
println("  LONG STRANGLE GRID (buy at ask)")
println("=" ^ 70)

long_df = DataFrame(
    Date = [t.date for t in long_trades],
    PutDelta = [t.put_delta for t in long_trades],
    CallDelta = [t.call_delta for t in long_trades],
    PnL = [t.pnl for t in long_trades],
    Debit = [t.credit for t in long_trades],  # stored as debit for long
    Spot = [t.spot for t in long_trades],
)
long_df.PnL_USD = long_df.PnL .* long_df.Spot
long_df.Debit_USD = long_df.Debit .* long_df.Spot

long_grid = combine(groupby(long_df, [:PutDelta, :CallDelta]),
    :PnL_USD => length => :Count,
    :PnL_USD => mean => :AvgPnL,
    :PnL_USD => std => :StdPnL,
    :Debit_USD => mean => :AvgDebit,
)
long_grid.Sharpe = [r.StdPnL > 0 ? r.AvgPnL / r.StdPnL * sqrt(252) : 0.0 for r in eachrow(long_grid)]
sort!(long_grid, [:PutDelta, :CallDelta])

# Print long grid tables using same helper
# Temporarily swap grid for printing
_saved_grid = grid
grid = long_grid
print_grid("Long Strangle Avg PnL (USD) — cost of protection", :AvgPnL; digits=2)
print_grid("Long Strangle Avg Debit (USD)", :AvgDebit; digits=2)
grid = _saved_grid

# =============================================================================
# Condor combination: short(inner) - long(outer)
# =============================================================================

println("\n", "=" ^ 70)
println("  CONDOR COMBINATIONS: short(inner) + long(outer)")
println("  condor PnL = short strangle PnL - long strangle PnL (per trade, matched by date)")
println("=" ^ 70)

# Index short and long trades by (date, put_delta, call_delta)
short_by_key = Dict{Tuple{Date,Float64,Float64}, StrangleTrade}()
for t in short_trades
    short_by_key[(t.date, t.put_delta, t.call_delta)] = t
end
long_by_key = Dict{Tuple{Date,Float64,Float64}, StrangleTrade}()
for t in long_trades
    long_by_key[(t.date, t.put_delta, t.call_delta)] = t
end

# Short iron condor = sell inner strangle (higher Δ, closer to ATM) + buy outer strangle (lower Δ, further OTM)
# Higher delta = closer to ATM = inner (short) legs
# Lower delta = further OTM = outer (long/wing) legs
# Format: (short_put_Δ, short_call_Δ, long_put_Δ, long_call_Δ, label)
# Short deltas must be > long deltas (short legs closer to ATM than wings)
CONDOR_COMBOS = [
    # Symmetric condors
    (0.10, 0.10, 0.05, 0.05, "10/10 short, 5/5 wings (tight sym)"),
    (0.15, 0.15, 0.05, 0.05, "15/15 short, 5/5 wings (wide sym)"),
    (0.20, 0.20, 0.10, 0.10, "20/20 short, 10/10 wings (sym baseline)"),
    (0.20, 0.20, 0.05, 0.05, "20/20 short, 5/5 wings (wide wings)"),
    (0.25, 0.25, 0.10, 0.10, "25/25 short, 10/10 wings"),
    (0.30, 0.30, 0.20, 0.20, "30/30 short, 20/20 wings"),
    # Asymmetric: put-heavy short legs (exploit put-side VRP)
    (0.20, 0.10, 0.10, 0.05, "20p/10c short, 10p/5c wings (asym)"),
    (0.20, 0.05, 0.10, 0.05, "20p/5c short, 10p/5c wings (put-heavy, no call wing)"),
    (0.25, 0.10, 0.10, 0.05, "25p/10c short, 10p/5c wings (wider put)"),
    (0.25, 0.10, 0.15, 0.05, "25p/10c short, 15p/5c wings (tight put wing)"),
    (0.15, 0.10, 0.05, 0.05, "15p/10c short, 5p/5c wings (moderate)"),
    (0.20, 0.15, 0.10, 0.05, "20p/15c short, 10p/5c wings"),
]

println("\n  ── Condor Grid ──")
@printf("  %-30s  %6s  %8s  %7s  %8s  %8s  %8s\n",
    "Condor", "Trades", "AvgPnL", "WinRate", "Sharpe", "CVaR5", "AvgCred")
@printf("  %-30s  %6s  %8s  %7s  %8s  %8s  %8s\n",
    "-"^30, "-"^6, "-"^8, "-"^7, "-"^8, "-"^8, "-"^8)

condor_results = []

for (sp_d, sc_d, lp_d, lc_d, label) in CONDOR_COMBOS
    pnls = Float64[]
    credits = Float64[]
    dates_c = Date[]

    for d in sort(unique(df.Date))
        sk = (d, sp_d, sc_d)
        lk = (d, lp_d, lc_d)
        haskey(short_by_key, sk) || continue
        haskey(long_by_key, lk) || continue
        st = short_by_key[sk]
        lt = long_by_key[lk]
        spot = st.spot
        condor_pnl = (st.pnl - lt.pnl) * spot  # short - long, in USD
        condor_credit = (st.credit - lt.credit) * spot  # net credit
        push!(pnls, condor_pnl)
        push!(credits, condor_credit)
        push!(dates_c, d)
    end

    isempty(pnls) && continue
    n = length(pnls)
    avg = mean(pnls)
    wr = count(>(0), pnls) / n * 100
    s = std(pnls)
    sharpe = s > 0 ? avg / s * sqrt(252) : 0.0
    n5 = max(1, floor(Int, 0.05 * n))
    cvar = mean(sort(pnls)[1:n5])
    avg_credit = mean(credits)

    push!(condor_results, (label=label, sp=sp_d, sc=sc_d, lp=lp_d, lc=lc_d,
        n=n, avg=avg, wr=wr, sharpe=sharpe, cvar=cvar, credit=avg_credit,
        pnls=pnls, dates=dates_c))

    @printf("  %-30s  %6d  %+8.2f  %6.1f%%  %+8.2f  %+8.2f  %+8.2f\n",
        label, n, avg, wr, sharpe, cvar, avg_credit)
end

# Year-by-year for top condors
if !isempty(condor_results)
    top_condors = sort(condor_results, by=r->r.sharpe, rev=true)[1:min(3, length(condor_results))]

    println("\n  ── Year-by-Year for Top 3 Condors ──")
    for cr in top_condors
        println("\n  $(cr.label)  (Sharpe=$(round(cr.sharpe, digits=2)))")
        @printf("  %-6s  %6s  %10s  %8s  %8s\n", "Year", "Trades", "PnL", "AvgPnL", "WinRate")
        @printf("  %-6s  %6s  %10s  %8s  %8s\n", "-"^6, "-"^6, "-"^10, "-"^8, "-"^8)
        for yr in sort(unique(year.(cr.dates)))
            idx = findall(i -> year(cr.dates[i]) == yr, 1:length(cr.dates))
            yp = cr.pnls[idx]
            nt = length(yp)
            @printf("  %-6d  %6d  %+10.2f  %+8.2f  %7.1f%%\n",
                yr, nt, sum(yp), mean(yp), count(>(0), yp)/nt*100)
        end
    end
end

# =============================================================================
# Loss anatomy: when and how do losses cluster?
# =============================================================================

println("\n", "=" ^ 70)
println("  LOSS ANATOMY — WHERE DO THE LOSSES COME FROM?")
println("=" ^ 70)

# Helper: analyze loss clustering for a PnL series
function loss_anatomy(label, dates::Vector{Date}, pnls::Vector{Float64})
    n = length(pnls)
    losses = findall(<(0), pnls)
    n_loss = length(losses)
    n_loss == 0 && return

    loss_pnls = pnls[losses]
    loss_dates = dates[losses]

    println("\n  ── $label ($n trades, $n_loss losses) ──")

    # Loss severity buckets
    mild = count(p -> p > -50, loss_pnls)
    moderate = count(p -> -200 < p <= -50, loss_pnls)
    severe = count(p -> -500 < p <= -200, loss_pnls)
    extreme = count(p -> p <= -500, loss_pnls)
    @printf("    Severity: mild(>-50)=%d  moderate(-200..-50)=%d  severe(-500..-200)=%d  extreme(<-500)=%d\n",
        mild, moderate, severe, extreme)
    @printf("    Avg loss: %.1f  Worst: %.1f  Loss PnL total: %.1f\n",
        mean(loss_pnls), minimum(loss_pnls), sum(loss_pnls))

    # Clustering: consecutive loss days
    streaks = Int[]
    streak = 1
    for i in 2:n_loss
        if Dates.value(loss_dates[i] - loss_dates[i-1]) <= 3  # within 3 calendar days = consecutive trading
            streak += 1
        else
            push!(streaks, streak)
            streak = 1
        end
    end
    push!(streaks, streak)
    @printf("    Loss streaks: %d clusters, max streak=%d, avg=%.1f\n",
        length(streaks), maximum(streaks), mean(streaks))

    # Worst 5 loss clusters (consecutive runs)
    # Rebuild clusters with dates and total PnL
    clusters = Tuple{Date,Date,Float64,Int}[]  # (start, end, total_pnl, count)
    c_start = 1
    for i in 2:n_loss+1
        if i > n_loss || Dates.value(loss_dates[i] - loss_dates[i-1]) > 3
            c_pnl = sum(loss_pnls[c_start:i-1])
            c_n = i - c_start
            push!(clusters, (loss_dates[c_start], loss_dates[i-1], c_pnl, c_n))
            if i <= n_loss
                c_start = i
            end
        end
    end
    sort!(clusters, by=c->c[3])  # worst first

    println("    Worst 5 loss clusters:")
    for (i, (ds, de, cp, cn)) in enumerate(clusters[1:min(5, length(clusters))])
        @printf("      %d. %s → %s  %d losses  total=\$%+.1f\n", i, ds, de, cn, cp)
    end

    # Monthly loss frequency heatmap
    loss_by_month = Dict{Int,Int}()
    loss_pnl_by_month = Dict{Int,Float64}()
    total_by_month = Dict{Int,Int}()
    for i in 1:n
        m = month(dates[i])
        total_by_month[m] = get(total_by_month, m, 0) + 1
    end
    for i in 1:n_loss
        m = month(loss_dates[i])
        loss_by_month[m] = get(loss_by_month, m, 0) + 1
        loss_pnl_by_month[m] = get(loss_pnl_by_month, m, 0.0) + loss_pnls[i]
    end
    print("    Monthly loss rate: ")
    for m in 1:12
        total = get(total_by_month, m, 0)
        nloss = get(loss_by_month, m, 0)
        rate = total > 0 ? nloss / total * 100 : 0.0
        @printf(" %s=%4.1f%%", Dates.monthabbr(m), rate)
    end
    println()
    print("    Monthly loss PnL:  ")
    for m in 1:12
        lp = get(loss_pnl_by_month, m, 0.0)
        @printf(" %s=%+5.0f", Dates.monthabbr(m), lp)
    end
    println()

    # Year-by-year loss count and severity
    println("    Year breakdown:")
    @printf("    %-6s  %5s  %5s  %10s  %10s\n", "Year", "Total", "Loss", "LossPnL", "AvgLoss")
    for yr in sort(unique(year.(dates)))
        yr_idx = findall(i -> year(dates[i]) == yr, 1:n)
        yr_loss_idx = findall(i -> year(loss_dates[i]) == yr, 1:n_loss)
        yr_loss_pnl = isempty(yr_loss_idx) ? 0.0 : sum(loss_pnls[yr_loss_idx])
        yr_avg_loss = isempty(yr_loss_idx) ? 0.0 : mean(loss_pnls[yr_loss_idx])
        @printf("    %-6d  %5d  %5d  %+10.1f  %+10.1f\n",
            yr, length(yr_idx), length(yr_loss_idx), yr_loss_pnl, yr_avg_loss)
    end
end

# Naked strangles
LOSS_STRANGLES = [
    (0.05, 0.05, "Short 5Δ/5Δ"),
    (0.10, 0.05, "Short 10Δp/5Δc"),
    (0.20, 0.05, "Short 20Δp/5Δc"),
]

for (pd, cd, label) in LOSS_STRANGLES
    cell = sort(filter(r -> r.PutDelta == pd && r.CallDelta == cd, df), :Date)
    nrow(cell) == 0 && continue
    loss_anatomy(label, cell.Date, cell.PnL_USD)
end

# Condors (short inner, long outer — higher Δ = closer to ATM = short legs)
LOSS_CONDORS = [
    (0.10, 0.10, 0.05, 0.05, "Condor 10/10 short, 5/5 wings"),
    (0.20, 0.10, 0.10, 0.05, "Condor 20p/10c short, 10p/5c wings"),
    (0.20, 0.20, 0.10, 0.10, "Condor 20/20 short, 10/10 wings"),
]

for (sp, sc, lp, lc, label) in LOSS_CONDORS
    pnls = Float64[]
    dates_c = Date[]
    for d in sort(unique(df.Date))
        sk = (d, sp, sc)
        lk = (d, lp, lc)
        haskey(short_by_key, sk) || continue
        haskey(long_by_key, lk) || continue
        st = short_by_key[sk]
        lt = long_by_key[lk]
        push!(pnls, (st.pnl - lt.pnl) * st.spot)
        push!(dates_c, d)
    end
    isempty(pnls) && continue
    loss_anatomy(label, dates_c, pnls)
end

println("\nOutput: $run_dir")
println("Done.")
