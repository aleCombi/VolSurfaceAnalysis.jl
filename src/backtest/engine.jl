# Backtest Engine
# Event-driven simulation over historical vol surfaces

using Dates

# ============================================================================
# Strategy Interface
# ============================================================================

"""
    Strategy

Abstract base type for trading strategies.

Implement `on_snapshot` to define your strategy's logic.
"""
abstract type Strategy end

"""
    on_snapshot(strategy, surface, portfolio) -> Vector{Order}

Called at each snapshot during backtesting.
Returns a vector of orders to execute.

# Arguments
- `strategy::Strategy`: Your strategy implementation
- `surface::VolatilitySurface`: Current market snapshot
- `portfolio::Portfolio`: Current portfolio state

# Returns
- Vector of `Order` objects to execute
"""
function on_snapshot(strategy::Strategy, surface::VolatilitySurface, 
                     portfolio::Portfolio)::Vector{Order}
    error("on_snapshot not implemented for $(typeof(strategy))")
end

"""
    on_expiry(strategy, portfolio, expiry_date, delivery_price) -> Nothing

Called when options expire. Override for custom expiry handling.
Default behavior: positions are auto-settled.
"""
function on_expiry(strategy::Strategy, portfolio::Portfolio, 
                   expiry_date::Date, delivery_price::Float64)
    # Default: do nothing (auto-settlement handled by engine)
    return nothing
end

# ============================================================================
# Orders
# ============================================================================

"""
    Order

An order to open or close a position.

# Fields
- `underlying::Underlying`: BTC or ETH
- `strike::Float64`: Strike price
- `expiry::DateTime`: Expiration (normalized to 08:00 UTC)
- `option_type::OptionType`: Call or Put
- `direction::Int`: +1 buy, -1 sell
- `quantity::Float64`: Number of contracts
"""
struct Order
    underlying::Underlying
    strike::Float64
    expiry::DateTime
    option_type::OptionType
    direction::Int
    quantity::Float64
end

"""
    Order(underlying, strike, expiry, option_type; direction=1, quantity=1.0)

Convenience constructor with defaults.
"""
function Order(underlying::Underlying, strike::Float64, expiry::DateTime,
               option_type::OptionType; direction::Int=1, quantity::Float64=1.0)
    Order(underlying, strike, expiry, option_type, direction, quantity)
end

# ============================================================================
# Performance Metrics
# ============================================================================

"""
    PerformanceMetrics

Summary statistics from a backtest.

# Fields
- `total_pnl::Float64`: Final P&L
- `sharpe_ratio::Float64`: Annualized Sharpe (assumes 0 risk-free rate)
- `max_drawdown::Float64`: Maximum peak-to-trough decline
- `max_drawdown_pct::Float64`: Max drawdown as percentage
- `win_rate::Float64`: Fraction of profitable trades
- `total_trades::Int`: Number of round-trip trades
- `avg_trade_pnl::Float64`: Average P&L per trade
- `profit_factor::Float64`: Gross profit / gross loss
"""
struct PerformanceMetrics
    total_pnl::Float64
    sharpe_ratio::Float64
    max_drawdown::Float64
    max_drawdown_pct::Float64
    win_rate::Float64
    total_trades::Int
    avg_trade_pnl::Float64
    profit_factor::Float64
end

"""
    compute_metrics(snapshots, trade_log) -> PerformanceMetrics

Compute performance metrics from backtest history.
"""
function compute_metrics(snapshots::Vector{PortfolioSnapshot}, 
                         trade_log::Vector{TradeRecord})::PerformanceMetrics
    isempty(snapshots) && return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 1.0)
    
    # P&L series
    pnl_series = [s.realized_pnl + s.unrealized_pnl for s in snapshots]
    total_pnl = last(pnl_series)
    
    # Returns for Sharpe calculation
    returns = Float64[]
    for i in 2:length(pnl_series)
        push!(returns, pnl_series[i] - pnl_series[i-1])
    end
    
    # Sharpe ratio (annualized, assuming hourly snapshots)
    sharpe = if !isempty(returns) && std(returns) > 0
        mean(returns) / std(returns) * sqrt(365.25 * 24)
    else
        0.0
    end
    
    # Max drawdown
    peak = -Inf
    max_dd = 0.0
    max_dd_pct = 0.0
    for pnl in pnl_series
        peak = max(peak, pnl)
        dd = peak - pnl
        if dd > max_dd
            max_dd = dd
            max_dd_pct = peak > 0 ? dd / peak : 0.0
        end
    end
    
    # Trade statistics
    closes = filter(t -> t.action == :close, trade_log)
    n_trades = length(closes)
    
    if n_trades > 0
        trade_pnls = [t.pnl for t in closes]
        wins = count(p -> p > 0, trade_pnls)
        win_rate = wins / n_trades
        avg_pnl = mean(trade_pnls)
        
        gross_profit = sum(filter(p -> p > 0, trade_pnls), init=0.0)
        gross_loss = abs(sum(filter(p -> p < 0, trade_pnls), init=0.0))
        profit_factor = gross_loss > 0 ? gross_profit / gross_loss : Inf
    else
        win_rate = 0.0
        avg_pnl = 0.0
        profit_factor = 1.0
    end
    
    return PerformanceMetrics(
        total_pnl, sharpe, max_dd, max_dd_pct,
        win_rate, n_trades, avg_pnl, profit_factor
    )
end

# ============================================================================
# Backtest Result
# ============================================================================

"""
    BacktestResult

Complete results from a backtest run.

# Fields
- `snapshots::Vector{PortfolioSnapshot}`: Portfolio state at each snapshot
- `trade_log::Vector{TradeRecord}`: All executed trades
- `metrics::PerformanceMetrics`: Summary statistics
- `start_date::Date`: Backtest start
- `end_date::Date`: Backtest end
- `underlying::Underlying`: Asset tested
"""
struct BacktestResult
    snapshots::Vector{PortfolioSnapshot}
    trade_log::Vector{TradeRecord}
    metrics::PerformanceMetrics
    start_date::Date
    end_date::Date
    underlying::Underlying
end

# ============================================================================
# Main Backtest Engine
# ============================================================================

"""
    run_backtest(strategy, store, underlying, start_date, end_date; kwargs...) -> BacktestResult

Run a backtest simulation over historical data.

# Arguments
- `strategy::Strategy`: Your strategy implementation
- `store::LocalDataStore`: Data source
- `underlying::Underlying`: BTC or ETH
- `start_date::Date`: Start of backtest period
- `end_date::Date`: End of backtest period

# Keyword Arguments
- `resolution::Period`: Snapshot frequency (default: `Hour(1)`)
- `initial_cash::Float64`: Starting cash (default: `0.0`)
- `verbose::Bool`: Print progress (default: `false`)

# Returns
- `BacktestResult` with full history and metrics

# Example
```julia
struct MyStrategy <: Strategy end

function on_snapshot(::MyStrategy, surface, portfolio)
    # Your logic here
    return Order[]
end

store = LocalDataStore("data/")
result = run_backtest(MyStrategy(), store, BTC, Date(2024,1,1), Date(2024,1,31))
println("Total P&L: \$(result.metrics.total_pnl)")
```
"""
function run_backtest(strategy::Strategy,
                      store::LocalDataStore,
                      underlying::Underlying,
                      start_date::Date,
                      end_date::Date;
                      resolution::Period=Hour(1),
                      initial_cash::Float64=0.0,
                      verbose::Bool=false)::BacktestResult
    
    # Create iterator and portfolio
    iter = SurfaceIterator(store, underlying; 
                           start_date=start_date, 
                           end_date=end_date,
                           resolution=resolution)
    
    portfolio = Portfolio(initial_cash=initial_cash)
    
    verbose && println("Backtest: $(length(iter)) snapshots from $start_date to $end_date")
    
    # Track expiries we've processed
    processed_expiries = Set{Date}()
    
    # Main loop
    for (i, surface) in enumerate(iter)
        current_date = Date(surface.timestamp)
        
        # Check for expired positions
        expiring_positions = positions_expiring(portfolio, current_date)
        if !isempty(expiring_positions) && !(current_date in processed_expiries)
            # Auto-settle expired positions at spot (simplified)
            for pos in expiring_positions
                close_position!(portfolio, pos, surface)
            end
            push!(processed_expiries, current_date)
        end
        
        # Get strategy signals
        orders = on_snapshot(strategy, surface, portfolio)
        
        # Execute orders
        for order in orders
            try
                execute_order!(portfolio, order, surface)
            catch e
                verbose && @warn "Order failed" order exception=e
            end
        end
        
        # Record snapshot
        record_snapshot!(portfolio, surface)
        
        if verbose && i % 100 == 0
            snap = portfolio.history[end]
            println("  [$i/$(length(iter))] $(surface.timestamp): MTM=$(round(snap.mtm_value, digits=4))")
        end
    end
    
    # Compute metrics
    metrics = compute_metrics(portfolio.history, portfolio.trade_log)
    
    verbose && println("Backtest complete: $(metrics.total_trades) trades, P&L=$(round(metrics.total_pnl, digits=4))")
    
    return BacktestResult(
        portfolio.history,
        portfolio.trade_log,
        metrics,
        start_date,
        end_date,
        underlying
    )
end

"""
    execute_order!(portfolio, order, surface)

Execute an order on the portfolio.
"""
function execute_order!(portfolio::Portfolio, order::Order, 
                        surface::VolatilitySurface)
    # Create trade from order
    trade = Trade(
        order.underlying,
        order.strike,
        order.expiry,
        order.option_type,
        surface.timestamp;
        direction=order.direction,
        quantity=order.quantity
    )
    
    add_position!(portfolio, trade, surface)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    pnl_series(result::BacktestResult) -> Tuple{Vector{DateTime}, Vector{Float64}}

Extract P&L time series from backtest result.
"""
function pnl_series(result::BacktestResult)::Tuple{Vector{DateTime}, Vector{Float64}}
    times = [s.timestamp for s in result.snapshots]
    pnl = [s.realized_pnl + s.unrealized_pnl for s in result.snapshots]
    return (times, pnl)
end

"""
    equity_curve(result::BacktestResult) -> Tuple{Vector{DateTime}, Vector{Float64}}

Extract equity curve (MTM value over time) from backtest result.
"""
function equity_curve(result::BacktestResult)::Tuple{Vector{DateTime}, Vector{Float64}}
    times = [s.timestamp for s in result.snapshots]
    equity = [s.mtm_value for s in result.snapshots]
    return (times, equity)
end

"""
    trades_summary(result::BacktestResult) -> DataFrame

Get a DataFrame summary of all trades.
"""
function trades_summary(result::BacktestResult)
    log = result.trade_log
    return DataFrame(
        position_id = [t.position_id for t in log],
        action = [t.action for t in log],
        timestamp = [t.timestamp for t in log],
        underlying = [t.trade.underlying for t in log],
        strike = [t.trade.strike for t in log],
        expiry = [t.trade.expiry for t in log],
        option_type = [t.trade.option_type for t in log],
        direction = [t.trade.direction for t in log],
        quantity = [t.trade.quantity for t in log],
        price = [t.price for t in log],
        vol = [t.vol for t in log],
        pnl = [t.pnl for t in log]
    )
end

# Import Statistics for mean/std
using Statistics: mean, std
