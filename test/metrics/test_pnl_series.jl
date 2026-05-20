# Tests for the PnLSeries intermediate and equity_curve helper.

const _PS_UND = Underlying("SPY")
const _PS_EXPIRY = DateTime(2024, 2, 16, 21, 0)

# Construct a Position directly (skipping the open_position fill path,
# which needs a full OptionQuote). The metrics layer treats Positions as
# given inputs, so building them by-hand here is the cleanest fixture.
function _ps_pos(strike, otype, direction, qty, entry_price, ts)
    trd = Trade(_PS_UND, strike, _PS_EXPIRY, otype; direction=direction, quantity=qty)
    Position(trd, Float64(entry_price), 480.0, missing, missing, ts)
end

@testset "pnl_series: empty ledger -> empty series, zero counts" begin
    s = pnl_series(Position[], 500.0)
    @test isempty(s.timestamps)
    @test isempty(s.pnl)
    @test s.settlement_spot == 500.0
    @test s.n_opens == 0
    @test s.n_closes == 0
end

@testset "pnl_series: single open, no close -> residual marked to spot" begin
    ts1 = DateTime(2024, 1, 15, 15, 30)
    open = _ps_pos(480.0, Call, +1, 1.0, 5.0, ts1)   # long call @ 5.0
    s = pnl_series([open], 490.0)                    # spot 490 at settlement
    @test length(s.pnl) == 1
    # payoff = max(490 - 480, 0) * +1 = 10; entry_cost = 5 * +1 = 5; pnl = 5
    @test s.pnl[1] ≈ 5.0
    @test s.timestamps[1] == _PS_EXPIRY              # default ts = expiry
    @test s.n_opens == 1
    @test s.n_closes == 0
end

@testset "pnl_series: settlement_timestamp overrides default expiry ts" begin
    ts1 = DateTime(2024, 1, 15, 15, 30)
    open = _ps_pos(480.0, Call, +1, 1.0, 5.0, ts1)
    mark_ts = DateTime(2024, 1, 20, 16, 0)
    s = pnl_series([open], 490.0; settlement_timestamp=mark_ts)
    @test s.timestamps[1] == mark_ts
end

@testset "pnl_series: open+close pair -> realized PnL independent of settle spot" begin
    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts2 = DateTime(2024, 1, 15, 16, 30)
    open  = _ps_pos(480.0, Call, +1, 1.0, 5.0, ts1)   # buy @ 5.0
    close = _ps_pos(480.0, Call, -1, 1.0, 5.5, ts2)   # sell @ 5.5
    s_low  = pnl_series([open, close], 400.0)
    s_high = pnl_series([open, close], 600.0)
    @test length(s_low.pnl) == 1
    @test s_low.pnl[1] ≈ 0.5                          # 5.5 - 5.0
    @test s_high.pnl[1] == s_low.pnl[1]               # closed: spot irrelevant
    @test s_low.timestamps[1] == ts2                  # ts at close fill
    @test s_low.n_opens == 1
    @test s_low.n_closes == 1
end

@testset "pnl_series: partial close splits into matched + residual entries" begin
    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts2 = DateTime(2024, 1, 15, 16, 30)
    open  = _ps_pos(480.0, Call, +1, 3.0, 5.0, ts1)   # buy 3 @ 5.0
    close = _ps_pos(480.0, Call, -1, 1.0, 5.5, ts2)   # sell 1 @ 5.5
    s = pnl_series([open, close], 490.0)              # spot 490 at settle
    @test length(s.pnl) == 2
    # matched chunk of qty 1 (ts2): (5.5 - 5.0) * 1 = 0.5
    # residual chunk of qty 2 (expiry): payoff_unit = 10, cost_unit = 5; (10-5)*2 = 10
    # Both round trips must appear; order by timestamp puts close first.
    @test s.timestamps == [ts2, _PS_EXPIRY]
    @test s.pnl[1] ≈ 0.5
    @test s.pnl[2] ≈ 10.0
    @test s.n_opens == 1
    @test s.n_closes == 1
end

@testset "pnl_series: FIFO -- first open matched first" begin
    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts2 = DateTime(2024, 1, 15, 15, 45)
    ts3 = DateTime(2024, 1, 15, 16, 30)
    open_a = _ps_pos(480.0, Call, +1, 1.0, 5.0, ts1)  # first lot
    open_b = _ps_pos(480.0, Call, +1, 1.0, 7.0, ts2)  # second lot
    close  = _ps_pos(480.0, Call, -1, 1.0, 6.0, ts3)  # closes one lot
    s = pnl_series([open_a, open_b, close], 480.0)    # ATM => zero payoff residual
    # FIFO closes open_a (cost 5.0 -> pnl +1.0); open_b is residual (cost 7.0 -> pnl -7.0)
    @test length(s.pnl) == 2
    # Order by timestamp: close at ts3, residual at expiry (later)
    @test s.timestamps == [ts3, _PS_EXPIRY]
    @test s.pnl[1] ≈ 1.0
    @test s.pnl[2] ≈ -7.0
end

@testset "pnl_series: multiple contracts do not cross-net" begin
    ts1 = DateTime(2024, 1, 15, 15, 30)
    call_open  = _ps_pos(480.0, Call, +1, 1.0, 5.0, ts1)
    put_open   = _ps_pos(480.0, Put,  +1, 1.0, 4.0, ts1)
    s = pnl_series([call_open, put_open], 490.0)
    # Two contracts, both still open: two residual entries.
    # call: payoff=10, cost=5 -> pnl=+5;   put: payoff=0, cost=4 -> pnl=-4
    @test length(s.pnl) == 2
    @test sort(s.pnl) ≈ [-4.0, 5.0]
    @test s.n_opens == 2
    @test s.n_closes == 0
end

@testset "pnl_series: bare short fill becomes an open lot (permissive)" begin
    ts1 = DateTime(2024, 1, 15, 16, 30)
    # No prior long to close: the -1 fill is treated as opening a short.
    short = _ps_pos(480.0, Call, -1, 1.0, 5.5, ts1)
    s = pnl_series([short], 490.0)
    @test length(s.pnl) == 1
    # short call: payoff_unit = max(490-480,0)*-1 = -10; cost_unit = 5.5*-1 = -5.5
    # pnl = (-10 - (-5.5)) * 1 = -4.5
    @test s.pnl[1] ≈ -4.5
    @test s.n_opens == 1
    @test s.n_closes == 0
end

@testset "pnl_series: flip-over (close exceeds open) leaves residual on the other side" begin
    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts2 = DateTime(2024, 1, 15, 16, 30)
    open = _ps_pos(480.0, Call, +1, 1.0, 5.0, ts1)   # buy 1 @ 5
    flip = _ps_pos(480.0, Call, -1, 3.0, 6.0, ts2)   # sell 3 @ 6 -> closes 1, shorts 2
    s = pnl_series([open, flip], 480.0)              # ATM spot at settle
    # Closed leg: (6 - 5) * 1 = 1.0 at ts2
    # Residual short qty 2 at expiry: payoff_unit=0, cost_unit=-6 -> (0 - (-6))*2 = 12
    @test length(s.pnl) == 2
    @test s.timestamps == [ts2, _PS_EXPIRY]
    @test s.pnl[1] ≈ 1.0
    @test s.pnl[2] ≈ 12.0
end

@testset "equity_curve: cumsum of pnl, empty stays empty" begin
    s = pnl_series(Position[], 500.0)
    @test isempty(equity_curve(s))

    ts1 = DateTime(2024, 1, 15, 15, 30)
    ts2 = DateTime(2024, 1, 15, 16, 30)
    open  = _ps_pos(480.0, Call, +1, 1.0, 5.0, ts1)
    close = _ps_pos(480.0, Call, -1, 1.0, 6.0, ts2)
    s = pnl_series([open, close], 500.0)
    @test equity_curve(s) ≈ cumsum(s.pnl)
end
