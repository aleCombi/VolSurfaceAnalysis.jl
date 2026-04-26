# Short/long strangle experiment helpers and reports.

using DataFrames
using Dates
using Plots
using Printf
using Statistics

struct StrangleTrade
    date::Date
    put_delta::Float64
    call_delta::Float64
    pnl::Float64
    credit::Float64
    put_spread::Float64
    call_spread::Float64
    spot::Float64
    put_strike::Float64
    call_strike::Float64
end

function _relative_spread(bid, ask)
    mid = (bid + ask) / 2
    return mid > 0 ? (ask - bid) / mid : Inf
end

function run_strangle_grid(
    source,
    sched,
    expiry_interval,
    put_deltas,
    call_deltas;
    rate::Real,
    div_yield::Real,
    progress_every::Int=200,
    io=stdout,
)
    short_trades = StrangleTrade[]
    long_trades = StrangleTrade[]
    n_entries = 0

    each_entry(source, expiry_interval, sched; clear_cache=true) do ctx, settlement
        ismissing(settlement) && return
        n_entries += 1

        dctx = delta_context(ctx; rate=rate, div_yield=div_yield)
        dctx === nothing && return
        spot = dctx.spot
        entry_date = Date(ctx.surface.timestamp)

        for pd in put_deltas
            p_K = delta_strike(dctx, -abs(pd), Put)
            p_K === nothing && continue
            p_rec = find_record_at_strike(dctx.put_recs, p_K)
            p_rec === nothing && continue
            p_bid = extract_price(p_rec, :bid)
            p_ask = extract_price(p_rec, :ask)
            (p_bid === nothing || p_ask === nothing) && continue
            p_spread = _relative_spread(p_bid, p_ask)

            for cd in call_deltas
                c_K = delta_strike(dctx, abs(cd), Call)
                c_K === nothing && continue
                c_rec = find_record_at_strike(dctx.call_recs, c_K)
                c_rec === nothing && continue
                c_bid = extract_price(c_rec, :bid)
                c_ask = extract_price(c_rec, :ask)
                (c_bid === nothing || c_ask === nothing) && continue
                c_spread = _relative_spread(c_bid, c_ask)

                short_pos = open_strangle_positions(ctx, p_K, c_K; direction=-1)
                if length(short_pos) == 2
                    pnl = settle(short_pos, Float64(settlement))
                    credit = -sum(entry_cost(p) for p in short_pos)
                    push!(short_trades, StrangleTrade(
                        entry_date, pd, cd, pnl, credit, p_spread, c_spread, spot, p_K, c_K))
                end

                long_pos = open_strangle_positions(ctx, p_K, c_K; direction=+1)
                if length(long_pos) == 2
                    pnl = settle(long_pos, Float64(settlement))
                    debit = sum(entry_cost(p) for p in long_pos)
                    push!(long_trades, StrangleTrade(
                        entry_date, pd, cd, pnl, debit, p_spread, c_spread, spot, p_K, c_K))
                end
            end
        end

        if progress_every > 0 && n_entries % progress_every == 0
            @printf(io, "  %d entries processed, %d short + %d long trades\r",
                n_entries, length(short_trades), length(long_trades))
            flush(io)
        end
    end

    return (short_trades=short_trades, long_trades=long_trades, entries=n_entries)
end

function strangle_trades_dataframe(trades::AbstractVector{StrangleTrade}; premium_col::Symbol=:Credit)
    df = DataFrame(
        Date=[t.date for t in trades],
        PutDelta=[t.put_delta for t in trades],
        CallDelta=[t.call_delta for t in trades],
        PnL=[t.pnl for t in trades],
        PutSpread=[t.put_spread for t in trades],
        CallSpread=[t.call_spread for t in trades],
        Spot=[t.spot for t in trades],
        PutStrike=[t.put_strike for t in trades],
        CallStrike=[t.call_strike for t in trades],
    )
    df[!, premium_col] = [t.credit for t in trades]
    df.Year = year.(df.Date)
    df.Month = Dates.format.(df.Date, "yyyy-mm")
    df.PnL_USD = df.PnL .* df.Spot
    df[!, Symbol(string(premium_col), "_USD")] = df[!, premium_col] .* df.Spot
    return df
end

_mean_finite(xs) = (vals = finite_values(xs); isempty(vals) ? NaN : mean(vals))

function strangle_grid_metrics(df::DataFrame)
    isempty(df) && return DataFrame()
    grid = combine(groupby(df, [:PutDelta, :CallDelta]),
        :PnL_USD => length => :Count,
        :PnL_USD => mean => :AvgPnL,
        :PnL_USD => sum => :TotalPnL,
        :PnL_USD => std => :StdPnL,
        :PnL_USD => (x -> count(>(0), x) / length(x) * 100) => :WinRate,
        :Credit_USD => mean => :AvgCredit,
        :PutSpread => _mean_finite => :AvgPutSpread,
        :CallSpread => _mean_finite => :AvgCallSpread,
        :PnL_USD => (x -> cvar_left(x; alpha=0.05)) => :CVaR5,
    )
    grid.Sharpe = [r.StdPnL > 0 ? r.AvgPnL / r.StdPnL * sqrt(252) : 0.0 for r in eachrow(grid)]
    grid.AvgSpread = (grid.AvgPutSpread .+ grid.AvgCallSpread) ./ 2
    sort!(grid, [:PutDelta, :CallDelta])
    return grid
end

function long_strangle_grid_metrics(df::DataFrame)
    isempty(df) && return DataFrame()
    grid = combine(groupby(df, [:PutDelta, :CallDelta]),
        :PnL_USD => length => :Count,
        :PnL_USD => mean => :AvgPnL,
        :PnL_USD => std => :StdPnL,
        :Debit_USD => mean => :AvgDebit,
    )
    grid.Sharpe = [r.StdPnL > 0 ? r.AvgPnL / r.StdPnL * sqrt(252) : 0.0 for r in eachrow(grid)]
    sort!(grid, [:PutDelta, :CallDelta])
    return grid
end

function print_top_bottom_delta_cells(grid::DataFrame; sort_field::Symbol=:Sharpe, n::Int=5)
    isempty(grid) && return

    println("\n  -- Top $(min(n, nrow(grid))) Cells by $sort_field --")
    top = sort(grid, sort_field, rev=true)[1:min(n, nrow(grid)), :]
    @printf("  %-6s  %-6s  %8s  %8s  %8s  %8s  %8s\n",
        "Put", "Call", "Sharpe", "AvgPnL", "WinRate", "CVaR5", "Count")
    for row in eachrow(top)
        @printf("  %-6.2f  %-6.2f  %+8.2f  %+8.3f  %7.1f%%  %+8.2f  %8d\n",
            row.PutDelta, row.CallDelta, row.Sharpe, row.AvgPnL, row.WinRate, row.CVaR5, row.Count)
    end

    println("\n  -- Bottom $(min(n, nrow(grid))) Cells by $sort_field --")
    bottom = sort(grid, sort_field)[1:min(n, nrow(grid)), :]
    @printf("  %-6s  %-6s  %8s  %8s  %8s  %8s  %8s\n",
        "Put", "Call", "Sharpe", "AvgPnL", "WinRate", "CVaR5", "Count")
    for row in eachrow(bottom)
        @printf("  %-6.2f  %-6.2f  %+8.2f  %+8.3f  %7.1f%%  %+8.2f  %8d\n",
            row.PutDelta, row.CallDelta, row.Sharpe, row.AvgPnL, row.WinRate, row.CVaR5, row.Count)
    end
end

function print_yearly_for_top_cells(grid::DataFrame, df::DataFrame; n::Int=3)
    isempty(grid) && return
    println("\n  -- Year-by-Year for Top $(min(n, nrow(grid))) Cells --")
    top = sort(grid, :Sharpe, rev=true)[1:min(n, nrow(grid)), :]
    for row in eachrow(top)
        pd, cd = row.PutDelta, row.CallDelta
        cell_df = filter(r -> r.PutDelta == pd && r.CallDelta == cd, df)
        println("\n  Put=$(pd)  Call=$(cd)  (Sharpe=$(round(row.Sharpe, digits=2)))")
        @printf("  %-6s  %6s  %10s  %8s  %8s\n", "Year", "Trades", "PnL", "AvgPnL", "WinRate")
        @printf("  %-6s  %6s  %10s  %8s  %8s\n", "-"^6, "-"^6, "-"^10, "-"^8, "-"^8)
        for yr in sort(unique(cell_df.Year))
            ydf = filter(r -> r.Year == yr, cell_df)
            nt = nrow(ydf)
            @printf("  %-6d  %6d  %+10.2f  %+8.3f  %7.1f%%\n",
                yr, nt, sum(ydf.PnL_USD), mean(ydf.PnL_USD), count(>(0), ydf.PnL_USD) / nt * 100)
        end
    end
end

function print_selected_yearly_sharpe(df::DataFrame, selected)
    println("\n  -- Year-by-Year Sharpe --")
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

    for (pd, cd, label) in selected
        cell = filter(r -> r.PutDelta == pd && r.CallDelta == cd, df)
        @printf("  %-22s", label)
        for yr in all_years
            ydf = filter(r -> r.Year == yr, cell)
            if nrow(ydf) < 10
                @printf("  %6s", "n/a")
            else
                @printf("  %+6.2f", annualized_sharpe(ydf.PnL_USD; empty=0.0))
            end
        end
        println()
    end
end

function print_selected_yearly_avg_pnl(df::DataFrame, selected)
    println("\n  -- Year-by-Year Avg PnL (USD) --")
    all_years = sort(unique(df.Year))
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

    for (pd, cd, label) in selected
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
end

function save_selected_cumulative_pnl(df::DataFrame, selected, path::AbstractString; colors=[:blue, :red, :green, :orange, :purple])
    p = plot(title="Cumulative PnL - Selected Strangles",
        xlabel="Date", ylabel="Cumulative PnL (USD)",
        legend=:topleft, size=(1000, 500), linewidth=2)

    for (idx, (pd, cd, label)) in enumerate(selected)
        cell = sort(filter(r -> r.PutDelta == pd && r.CallDelta == cd, df), :Date)
        nrow(cell) == 0 && continue
        plot!(p, cell.Date, cumsum(cell.PnL_USD), label=label, color=colors[mod1(idx, length(colors))])
    end
    hline!(p, [0], color=:black, linestyle=:dash, label="", linewidth=1)
    mkpath(dirname(path))
    savefig(p, path)
    return path
end

function save_selected_rolling_sharpe(
    df::DataFrame,
    selected,
    path::AbstractString;
    roll_window::Int=63,
    colors=[:blue, :red, :green, :orange, :purple],
)
    p = plot(title="Rolling Sharpe ($(roll_window)-trade window)",
        xlabel="Date", ylabel="Sharpe (annualized)",
        legend=:topleft, size=(1000, 500), linewidth=1.5)

    for (idx, (pd, cd, label)) in enumerate(selected)
        cell = sort(filter(r -> r.PutDelta == pd && r.CallDelta == cd, df), :Date)
        nrow(cell) < roll_window && continue
        roll_sharpe = Float64[]
        roll_dates = Date[]
        for i in roll_window:nrow(cell)
            push!(roll_sharpe, annualized_sharpe(cell.PnL_USD[i-roll_window+1:i]; empty=0.0))
            push!(roll_dates, cell.Date[i])
        end
        plot!(p, roll_dates, roll_sharpe, label=label, color=colors[mod1(idx, length(colors))])
    end
    hline!(p, [0], color=:black, linestyle=:dash, label="", linewidth=1)
    mkpath(dirname(path))
    savefig(p, path)
    return path
end

function print_selected_monthly_seasonality(df::DataFrame, selected)
    println("\n  -- Monthly Seasonality (Avg PnL by Calendar Month) --")
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

    for (pd, cd, label) in selected
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
end

function save_short_strangle_payoff_diagrams(
    df::DataFrame,
    selected,
    path::AbstractString;
    colors=[:blue, :red, :green, :orange, :purple],
)
    p = plot(title="Short Strangle Payoff at Expiry",
        xlabel="Spot at Expiry (% of entry spot)", ylabel="P&L (USD)",
        legend=:bottomright, size=(1000, 600), linewidth=2)

    spot_range = range(85, 115, length=200)
    for (idx, (pd, cd, label)) in enumerate(selected)
        cell = filter(r -> r.PutDelta == pd && r.CallDelta == cd, df)
        nrow(cell) == 0 && continue

        avg_spot = mean(cell.Spot)
        avg_credit = mean(cell.Credit) * avg_spot
        avg_put_K = mean(cell.PutStrike)
        avg_call_K = mean(cell.CallStrike)
        put_pct = avg_put_K / avg_spot * 100
        call_pct = avg_call_K / avg_spot * 100
        payoffs_usd = [
            avg_credit - max(avg_put_K - (s / 100 * avg_spot), 0) - max((s / 100 * avg_spot) - avg_call_K, 0)
            for s in spot_range
        ]

        plot!(p, collect(spot_range), payoffs_usd;
            label="$(label) [$(round(put_pct, digits=1))%-$(round(call_pct, digits=1))%]",
            color=colors[mod1(idx, length(colors))])
    end
    hline!(p, [0], color=:black, linestyle=:dash, label="", linewidth=1)
    vline!(p, [100], color=:gray, linestyle=:dot, label="", linewidth=1)
    mkpath(dirname(path))
    savefig(p, path)
    return path
end

function strangle_trade_index(trades::AbstractVector{StrangleTrade})
    out = Dict{Tuple{Date,Float64,Float64},StrangleTrade}()
    for t in trades
        out[(t.date, t.put_delta, t.call_delta)] = t
    end
    return out
end

function evaluate_condor_combinations(short_trades, long_trades, combos)
    short_by_key = strangle_trade_index(short_trades)
    long_by_key = strangle_trade_index(long_trades)
    dates = sort(unique(t.date for t in short_trades))
    results = NamedTuple[]

    for (sp_d, sc_d, lp_d, lc_d, label) in combos
        pnls = Float64[]
        credits = Float64[]
        condor_dates = Date[]

        for d in dates
            sk = (d, sp_d, sc_d)
            lk = (d, lp_d, lc_d)
            haskey(short_by_key, sk) || continue
            haskey(long_by_key, lk) || continue
            st = short_by_key[sk]
            lt = long_by_key[lk]
            push!(pnls, (st.pnl - lt.pnl) * st.spot)
            push!(credits, (st.credit - lt.credit) * st.spot)
            push!(condor_dates, d)
        end

        isempty(pnls) && continue
        n = length(pnls)
        avg = mean(pnls)
        s = std(pnls)
        push!(results, (
            label=label,
            sp=sp_d,
            sc=sc_d,
            lp=lp_d,
            lc=lc_d,
            n=n,
            avg=avg,
            wr=count(>(0), pnls) / n * 100,
            sharpe=s > 0 ? avg / s * sqrt(252) : 0.0,
            cvar=cvar_left(pnls; alpha=0.05),
            credit=mean(credits),
            pnls=pnls,
            dates=condor_dates,
        ))
    end
    return results
end

function print_condor_results(results)
    println("\n  -- Condor Grid --")
    @printf("  %-30s  %6s  %8s  %7s  %8s  %8s  %8s\n",
        "Condor", "Trades", "AvgPnL", "WinRate", "Sharpe", "CVaR5", "AvgCred")
    @printf("  %-30s  %6s  %8s  %7s  %8s  %8s  %8s\n",
        "-"^30, "-"^6, "-"^8, "-"^7, "-"^8, "-"^8, "-"^8)
    for row in results
        @printf("  %-30s  %6d  %+8.2f  %6.1f%%  %+8.2f  %+8.2f  %+8.2f\n",
            row.label, row.n, row.avg, row.wr, row.sharpe, row.cvar, row.credit)
    end
end

function print_top_condor_yearly(results; n::Int=3)
    isempty(results) && return
    top_condors = sort(results, by=r -> r.sharpe, rev=true)[1:min(n, length(results))]
    println("\n  -- Year-by-Year for Top $(length(top_condors)) Condors --")
    for row in top_condors
        println("\n  $(row.label)  (Sharpe=$(round(row.sharpe, digits=2)))")
        @printf("  %-6s  %6s  %10s  %8s  %8s\n", "Year", "Trades", "PnL", "AvgPnL", "WinRate")
        @printf("  %-6s  %6s  %10s  %8s  %8s\n", "-"^6, "-"^6, "-"^10, "-"^8, "-"^8)
        for yr in sort(unique(year.(row.dates)))
            idx = findall(i -> year(row.dates[i]) == yr, 1:length(row.dates))
            yp = row.pnls[idx]
            nt = length(yp)
            @printf("  %-6d  %6d  %+10.2f  %+8.2f  %7.1f%%\n",
                yr, nt, sum(yp), mean(yp), count(>(0), yp) / nt * 100)
        end
    end
end

function print_loss_anatomy(label, dates::AbstractVector{Date}, pnls::AbstractVector{<:Real})
    n = length(pnls)
    losses = findall(<(0), pnls)
    n_loss = length(losses)
    n_loss == 0 && return

    loss_pnls = Float64.(pnls[losses])
    loss_dates = dates[losses]

    println("\n  -- $label ($n trades, $n_loss losses) --")

    mild = count(p -> p > -50, loss_pnls)
    moderate = count(p -> -200 < p <= -50, loss_pnls)
    severe = count(p -> -500 < p <= -200, loss_pnls)
    extreme = count(p -> p <= -500, loss_pnls)
    @printf("    Severity: mild(>-50)=%d  moderate(-200..-50)=%d  severe(-500..-200)=%d  extreme(<-500)=%d\n",
        mild, moderate, severe, extreme)
    @printf("    Avg loss: %.1f  Worst: %.1f  Loss PnL total: %.1f\n",
        mean(loss_pnls), minimum(loss_pnls), sum(loss_pnls))

    streaks = Int[]
    streak = 1
    for i in 2:n_loss
        if Dates.value(loss_dates[i] - loss_dates[i-1]) <= 3
            streak += 1
        else
            push!(streaks, streak)
            streak = 1
        end
    end
    push!(streaks, streak)
    @printf("    Loss streaks: %d clusters, max streak=%d, avg=%.1f\n",
        length(streaks), maximum(streaks), mean(streaks))

    clusters = Tuple{Date,Date,Float64,Int}[]
    c_start = 1
    for i in 2:(n_loss + 1)
        if i > n_loss || Dates.value(loss_dates[i] - loss_dates[i-1]) > 3
            c_pnl = sum(loss_pnls[c_start:i-1])
            c_n = i - c_start
            push!(clusters, (loss_dates[c_start], loss_dates[i-1], c_pnl, c_n))
            if i <= n_loss
                c_start = i
            end
        end
    end
    sort!(clusters, by=c -> c[3])

    println("    Worst 5 loss clusters:")
    for (i, (ds, de, cp, cn)) in enumerate(clusters[1:min(5, length(clusters))])
        @printf("      %d. %s -> %s  %d losses  total=\$%+.1f\n", i, ds, de, cn, cp)
    end

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
        @printf(" %s=%+5.0f", Dates.monthabbr(m), get(loss_pnl_by_month, m, 0.0))
    end
    println()

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

function run_strangle_grid_experiment(;
    output_root::AbstractString,
    symbol::AbstractString="SPY",
    start_date::Date=Date(2016, 3, 28),
    end_date::Date=Date(2024, 1, 31),
    entry_time::Time=Time(12, 0),
    expiry_interval=Day(1),
    spread_lambda::Real=0.7,
    rate::Real=0.045,
    div_yield::Real=0.013,
    put_deltas=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    call_deltas=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    selected=[
        (0.05, 0.05, "5d/5d (widest)"),
        (0.20, 0.05, "20p/5c (asym sweet)"),
        (0.20, 0.20, "20d/20d (symmetric)"),
        (0.30, 0.30, "30d/30d (tightest)"),
        (0.05, 0.30, "5p/30c (inverted)"),
    ],
    condor_combos=[
        (0.10, 0.10, 0.05, 0.05, "10/10 short, 5/5 wings"),
        (0.15, 0.15, 0.05, 0.05, "15/15 short, 5/5 wings"),
        (0.20, 0.20, 0.10, 0.10, "20/20 short, 10/10 wings"),
        (0.20, 0.20, 0.05, 0.05, "20/20 short, 5/5 wings"),
        (0.25, 0.25, 0.10, 0.10, "25/25 short, 10/10 wings"),
        (0.30, 0.30, 0.20, 0.20, "30/30 short, 20/20 wings"),
        (0.20, 0.10, 0.10, 0.05, "20p/10c short, 10p/5c wings"),
        (0.20, 0.05, 0.10, 0.05, "20p/5c short, 10p/5c wings"),
        (0.25, 0.10, 0.10, 0.05, "25p/10c short, 10p/5c wings"),
        (0.25, 0.10, 0.15, 0.05, "25p/10c short, 15p/5c wings"),
        (0.15, 0.10, 0.05, 0.05, "15p/10c short, 5p/5c wings"),
        (0.20, 0.15, 0.10, 0.05, "20p/15c short, 10p/5c wings"),
    ],
    loss_strangles=[
        (0.05, 0.05, "Short 5d/5d"),
        (0.10, 0.05, "Short 10p/5c"),
        (0.20, 0.05, "Short 20p/5c"),
    ],
    loss_condors=[
        (0.10, 0.10, 0.05, 0.05, "Condor 10/10 short, 5/5 wings"),
        (0.20, 0.10, 0.10, 0.05, "Condor 20p/10c short, 10p/5c wings"),
        (0.20, 0.20, 0.10, 0.10, "Condor 20/20 short, 10/10 wings"),
    ],
)
    run_dir = make_run_dir(output_root, "strangle_grid")
    println("Output: $run_dir")

    println("\nLoading $symbol data from $start_date to $end_date...")
    (; source, sched) = polygon_parquet_source(symbol;
        start_date=start_date,
        end_date=end_date,
        entry_time=entry_time,
        rate=rate,
        div_yield=div_yield,
        spread_lambda=spread_lambda,
    )
    println("  $(length(sched)) entry timestamps")

    println("\nRunning strangle grid: $(length(put_deltas))x$(length(call_deltas)) = $(length(put_deltas) * length(call_deltas)) cells...")
    println("  (short + long strangles in single pass)")
    run = run_strangle_grid(source, sched, expiry_interval, put_deltas, call_deltas;
        rate=rate,
        div_yield=div_yield,
    )
    short_trades = run.short_trades
    long_trades = run.long_trades
    println("  $(run.entries) entries -> $(length(short_trades)) short + $(length(long_trades)) long trades")

    df = strangle_trades_dataframe(short_trades; premium_col=:Credit)
    grid = strangle_grid_metrics(df)

    println("\n", "="^70)
    println("  STRANGLE P&L GRID - $symbol $(start_date) to $(end_date)")
    println("  Entry @ $(entry_time), expiry = +1 day, spread_lambda=$(spread_lambda)")
    println("="^70)

    print_delta_grid("Avg PnL (USD per trade)", grid, put_deltas, call_deltas, :AvgPnL; digits=3)
    print_delta_grid("Win Rate (%)", grid, put_deltas, call_deltas, :WinRate; digits=1, pct=true)
    print_delta_grid("Sharpe (annualized)", grid, put_deltas, call_deltas, :Sharpe; digits=2)
    print_delta_grid("Avg Credit (USD)", grid, put_deltas, call_deltas, :AvgCredit; digits=3)
    print_delta_grid("CVaR 5% (USD)", grid, put_deltas, call_deltas, :CVaR5; digits=2)
    print_delta_grid("Avg Bid-Ask Spread (rel)", grid, put_deltas, call_deltas, :AvgSpread; digits=3)
    print_delta_grid("Trade Count", grid, put_deltas, call_deltas, :Count; digits=0)

    print_top_bottom_delta_cells(grid; sort_field=:Sharpe, n=5)
    print_yearly_for_top_cells(grid, df; n=3)

    println("\n  Saving heatmaps...")
    for (field, title, file, color) in [
        (:Sharpe, "Sharpe (annualized)", "sharpe_heatmap.png", :RdYlGn),
        (:AvgPnL, "Avg PnL (USD)", "avgpnl_heatmap.png", :RdYlGn),
        (:WinRate, "Win Rate (%)", "winrate_heatmap.png", :YlGn),
        (:AvgCredit, "Avg Credit (USD)", "credit_heatmap.png", :YlOrRd),
        (:CVaR5, "CVaR 5% (USD)", "cvar_heatmap.png", :RdYlGn),
        (:AvgSpread, "Avg Bid-Ask Spread", "spread_heatmap.png", :YlOrRd),
    ]
        save_delta_heatmap(grid, put_deltas, call_deltas, field, title, joinpath(run_dir, file); color=color)
        println("    $file")
    end

    println("\n", "="^70)
    println("  HISTORICAL EVOLUTION - SELECTED CELLS")
    println("="^70)
    print_selected_yearly_sharpe(df, selected)
    print_selected_yearly_avg_pnl(df, selected)

    println("\n  Saving historical plots...")
    save_selected_cumulative_pnl(df, selected, joinpath(run_dir, "cumulative_pnl_selected.png"))
    println("    cumulative_pnl_selected.png")
    save_selected_rolling_sharpe(df, selected, joinpath(run_dir, "rolling_sharpe_selected.png"); roll_window=63)
    println("    rolling_sharpe_selected.png")

    print_selected_monthly_seasonality(df, selected)

    println("\n  Saving payoff diagrams...")
    save_short_strangle_payoff_diagrams(df, selected, joinpath(run_dir, "payoff_diagrams.png"))
    println("    payoff_diagrams.png")

    println("\n", "="^70)
    println("  LONG STRANGLE GRID (buy at ask)")
    println("="^70)

    long_df = strangle_trades_dataframe(long_trades; premium_col=:Debit)
    long_grid = long_strangle_grid_metrics(long_df)
    print_delta_grid("Long Strangle Avg PnL (USD) - cost of protection", long_grid, put_deltas, call_deltas, :AvgPnL; digits=2)
    print_delta_grid("Long Strangle Avg Debit (USD)", long_grid, put_deltas, call_deltas, :AvgDebit; digits=2)

    println("\n", "="^70)
    println("  CONDOR COMBINATIONS: short(inner) + long(outer)")
    println("  condor PnL = short strangle PnL - long strangle PnL (matched by date)")
    println("="^70)

    condor_results = evaluate_condor_combinations(short_trades, long_trades, condor_combos)
    print_condor_results(condor_results)
    print_top_condor_yearly(condor_results; n=3)

    println("\n", "="^70)
    println("  LOSS ANATOMY - WHERE DO THE LOSSES COME FROM?")
    println("="^70)

    for (pd, cd, label) in loss_strangles
        cell = sort(filter(row -> row.PutDelta == pd && row.CallDelta == cd, df), :Date)
        isempty(cell.Date) && continue
        print_loss_anatomy(label, cell.Date, cell.PnL_USD)
    end

    for row in evaluate_condor_combinations(short_trades, long_trades, loss_condors)
        print_loss_anatomy(row.label, row.dates, row.pnls)
    end

    println("\nOutput: $run_dir")
    println("Done.")
    return (
        run_dir=run_dir,
        short_trades=short_trades,
        long_trades=long_trades,
        grid=grid,
        long_grid=long_grid,
        condor_results=condor_results,
    )
end
