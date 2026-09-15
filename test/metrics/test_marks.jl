# `mark_price` and the `marked_curve` builder: the one place `metrics`
# reads market data. Scenarios reuse the ledger fixtures (`_lg_`), so the
# cash literals are the ones test/ledger works out by hand.
#
# The fixture week is Tue 2024-01-16 to Fri 2024-01-19 (Mon the 15th is
# a holiday, so it is not a session at all). ET is UTC-5 in January, so a
# session runs 14:30 to 21:00 UTC and its close is the 21:00 print.

const _MK_UND    = Underlying("SPY")
const _MK_FROM   = DateTime(2024, 1, 16)
const _MK_TO     = DateTime(2024, 1, 19, 23, 59, 59)
const _MK_CLOSES = [DateTime(2024, 1, d, 21, 0) for d in 16:19]

_mk_quote(ts, contract, bid, ask) = OptionQuote(
    "X", contract.underlying, contract.expiry, contract.strike, contract.option_type,
    bid, ask, ismissing(bid) || ismissing(ask) ? missing : (bid + ask) / 2,
    missing, missing, missing, ts)

# Two spot prints a session: the open and the close. Only the last one in
# the 09:30-16:00 ET window is the session close.
function _mk_spots()
    out = SpotPrice[]
    for d in 16:19
        push!(out, SpotPrice(_MK_UND, 480.0, DateTime(2024, 1, d, 14, 30)))
        push!(out, SpotPrice(_MK_UND, 481.0, DateTime(2024, 1, d, 21, 0)))
    end
    return out
end

# The strangle's two lots quoted at each session close. `skip` drops a
# (day, contract) pair so the book cannot be marked there.
function _mk_quotes(; skip = Tuple{Int,ContractKey}[])
    mids = Dict(                                   # day => (put mid, call mid)
        16 => (0.85, 1.10), 17 => (0.65, 0.90), 18 => (1.25, 1.50), 19 => (0.01, 0.01))
    out = OptionQuote[]
    for d in 16:19, (contract, mid) in ((_LG_PUT470, mids[d][1]), (_LG_CALL490, mids[d][2]))
        (d, contract) in skip && continue
        push!(out, _mk_quote(DateTime(2024, 1, d, 21, 0), contract, mid - 0.05, mid + 0.05))
    end
    return out
end

_mk_data(; skip = Tuple{Int,ContractKey}[], extra = ()) =
    MarketData(InMemory(_mk_quotes(; skip=skip)), InMemory(_mk_spots()), extra...)

# Cash of the open strangle is 19370 cents; each lot is short 1 contract of
# 100 shares, so marked profit = 193.70 - 100 * (put mid + call mid).
_mk_expected(put_mid, call_mid) = 193.70 - 100 * (put_mid + call_mid)

@testset "mark_price: the quote mid, not the last trade" begin
    data = _mk_data()
    t = _MK_CLOSES[2]
    cut = TimeCut(data, t)
    @test mark_price(cut, _LG_PUT470, t) ≈ 0.65
    @test mark_price(cut, _LG_CALL490, t) ≈ 0.90
end

@testset "mark_price: no quote and no surface is a named failure, not a number" begin
    data = _mk_data()
    t = _MK_CLOSES[2]
    cut = TimeCut(data, t)
    err = try; mark_price(cut, _LG_PUT465B, t); nothing; catch e; e; end
    @test err isa UnpriceableLeg
    @test err.reason === :no_mark
    @test err.contract == _LG_PUT465B
    @test err.t == t
    println("  named failure: ", sprint(showerror, err))

    # A one-sided quote is not a mid either, and falls through the same way.
    one_sided = MarketData(InMemory([_mk_quote(t, _LG_PUT470, 0.60, missing)]),
                           InMemory(_mk_spots()))
    err2 = try; mark_price(TimeCut(one_sided, t), _LG_PUT470, t); nothing; catch e; e; end
    @test err2 isa UnpriceableLeg && err2.reason === :no_mark
    println("  named failure: ", sprint(showerror, err2))
end

@testset "mark_price: the surface is the fallback when the chain has no quote" begin
    t = _MK_CLOSES[2]
    tau = time_to_expiry(_LG_EXPIRY_A, t)
    surf = RawSurface(_MK_UND, t, 480.0, 0.04, 0.0,
                      [ExpirySlice(_LG_EXPIRY_A, tau, [460.0, 470.0, 490.0, 500.0],
                                   [0.22, 0.20, 0.18, 0.19])])
    # The chain is served but holds nothing for this contract -- the case the
    # surface exists to cover. A chain that serves the underlying at all is
    # what distinguishes it from a misconfigured map, which stays loud.
    data = MarketData(InMemory([_mk_quote(t, _lg_call(600.0), 0.01, 0.03)]),
                      InMemory(_mk_spots()), InMemory{VolatilitySurface}([surf]))
    cut = TimeCut(data, t)
    expected = price(surf, _LG_EXPIRY_A, 470.0, Put)
    @test mark_price(cut, _LG_PUT470, t) ≈ expected
    @test expected > 0
    # An expiry the surface has no slice for is still the named failure.
    err = try; mark_price(cut, _LG_PUT465B, t); nothing; catch e; e; end
    @test err isa UnpriceableLeg && err.reason === :no_mark
    println("  named failure: ", sprint(showerror, err))
end

@testset "mark_price: a surface from an earlier instant is stale, not a mark" begin
    # A surface carries its own spot and its slices their own cached tau, so
    # pricing off one built earlier values the contract at *that* instant.
    # Marking session after session from one such surface would carry a price
    # forward while `n_unmarked` stayed at zero -- the behaviour the curve
    # refuses. `asof` walks backward, so this is reachable whenever the chain
    # stops serving a contract but the surface provider still holds an older
    # build.
    built, t = _MK_CLOSES[2], _MK_CLOSES[3]
    tau  = time_to_expiry(_LG_EXPIRY_A, built)
    surf = RawSurface(_MK_UND, built, 480.0, 0.04, 0.0,
                      [ExpirySlice(_LG_EXPIRY_A, tau, [460.0, 470.0, 490.0, 500.0],
                                   [0.22, 0.20, 0.18, 0.19])])
    data = MarketData(InMemory([_mk_quote(t, _lg_call(600.0), 0.01, 0.03)]),
                      InMemory(_mk_spots()), InMemory{VolatilitySurface}([surf]))
    # At the instant it was built, the surface marks.
    @test mark_price(TimeCut(data, built), _LG_PUT470, built) ≈
          price(surf, _LG_EXPIRY_A, 470.0, Put)
    # One session later the same surface is all that `asof` can reach, and it
    # is refused rather than reused.
    err = try; mark_price(TimeCut(data, t), _LG_PUT470, t); nothing; catch e; e; end
    @test err isa UnpriceableLeg && err.reason === :no_mark
    println("  refused a stale surface: ", sprint(showerror, err))
end

@testset "mark_price: a negative quote mid is corrupt input, not a cheap contract" begin
    # `OptionQuote` does not validate, so this is where it is caught. A long
    # option is never a liability and a short one never an asset; accepting
    # the mid would flip a short lot's contribution with nothing counted.
    t = _MK_CLOSES[2]
    data = MarketData(InMemory([_mk_quote(t, _LG_PUT470, -2.0, -1.0)]),
                      InMemory(_mk_spots()))
    cut = TimeCut(data, t)
    err = try; mark_price(cut, _LG_PUT470, t); nothing; catch e; e; end
    @test err isa UnpriceableLeg && err.reason === :no_mark
    println("  refused a negative mid: ", sprint(showerror, err))
    # Zero is a real price for a worthless contract and still marks.
    ok = MarketData(InMemory([_mk_quote(t, _LG_PUT470, 0.0, 0.0)]), InMemory(_mk_spots()))
    @test mark_price(TimeCut(ok, t), _LG_PUT470, t) == 0.0
end

@testset "marked_curve: one point per session close, cash plus the marked book" begin
    L, _ = _lg_case_strangle_order()
    data = _mk_data()
    c = marked_curve(L, data, _MK_UND, _MK_FROM, _MK_TO).curve
    @test c.timestamps == _MK_CLOSES
    @test n_marked(c) == 4 && n_unmarked(c) == 0
    # At the opening marks the whole position is worth what it cost, so the
    # marked profit is exactly the commission -- cash alone would report the
    # 195.00 of premium received as profit.
    @test c.profit[1] ≈ _mk_expected(0.85, 1.10) ≈ -1.30
    @test c.profit ≈ [_mk_expected(0.85, 1.10), _mk_expected(0.65, 0.90),
                      _mk_expected(1.25, 1.50), _mk_expected(0.01, 0.01)]
    @test session_changes(c) ≈ [40.0, -120.0, 273.0]
end

@testset "marked_curve: a flat book needs no market data and is the realised total" begin
    L, _ = _lg_case_strangle_closed()      # closed at _LG_T_CLOSE = 2024-01-18T20:00
    c = marked_curve(L, _mk_data(), _MK_UND, _MK_FROM, _MK_TO).curve
    @test n_unmarked(c) == 0
    trades = trade_pnl(L)
    # Once flat, the marked profit is the ledger's realised total and stops
    # moving -- the last two sessions are one flat step.
    @test c.profit[3] ≈ total_pnl(trades)
    @test c.profit[4] ≈ total_pnl(trades)
    @test session_changes(c)[3] ≈ 0.0
end

@testset "marked_curve: an unmarkable lot costs the session, never a partial sum" begin
    L, _ = _lg_case_strangle_order()
    data = _mk_data(; skip=[(18, _LG_CALL490)])
    out = (@test_logs (:warn,) match_mode=:any marked_curve(L, data, _MK_UND, _MK_FROM, _MK_TO))
    c = out.curve
    @test n_unmarked(c) == 1
    @test c.unmarked_at == [_MK_CLOSES[3]]
    @test c.unmarked_reason == [:no_mark]
    # The point carries no value anywhere: not NaN, not the put's half of it.
    @test c.timestamps == [_MK_CLOSES[1], _MK_CLOSES[2], _MK_CLOSES[4]]
    @test !any(isnan, c.profit)
    @test !(_mk_expected(1.25, 0.0) in c.profit)
    # And the change across it is dropped rather than scaled over two days.
    @test session_changes(c) ≈ [40.0]
    # The builder is the *producer* of what persistence later keeps: the
    # failure is retained here, naming the lot that could not be priced,
    # or there is nothing for a store to write down.
    f = only(out.failures)
    @test f.stage === :mark && f.at == _MK_CLOSES[3] && f.reason === :no_mark
    @test occursin("490.0C", f.subject) && occursin("lot@", f.subject)
    @test f.reason in c.unmarked_reason          # what load_run checks on the way back
end

@testset "marked_curve: two unpriceable lots are two failures and one lost session" begin
    # The old builder stopped at the first lot it could not price, so a
    # session with two blind legs reported one reason and lost the other
    # question entirely. Both are retained now, and the session is still
    # counted once -- which is exactly the shape the store's curve/failure
    # agreement is checked against.
    L, _ = _lg_case_strangle_order()
    data = _mk_data(; skip=[(18, _LG_CALL490), (18, _LG_PUT470)])
    out = (@test_logs (:warn,) match_mode=:any marked_curve(L, data, _MK_UND, _MK_FROM, _MK_TO))
    @test n_unmarked(out.curve) == 1
    @test out.curve.unmarked_at == [_MK_CLOSES[3]]
    @test length(out.failures) == 2
    @test all(f -> f.stage === :mark && f.at == _MK_CLOSES[3], out.failures)
    @test all(f -> f.reason === :no_mark, out.failures)
    # Two lots of one session are two distinguishable questions.
    subjects = sort([f.subject for f in out.failures])
    @test length(unique(subjects)) == 2
    @test any(s -> occursin("470.0P", s), subjects) && any(s -> occursin("490.0C", s), subjects)
end

@testset "marked_curve: a printless session the calendar calls open is a counted gap" begin
    L, _ = _lg_case_strangle_order()
    spots = filter(p -> Date(p.timestamp) != Date(2024, 1, 17), _mk_spots())
    data = MarketData(InMemory(_mk_quotes()), InMemory(spots))
    out = (@test_logs (:warn,) match_mode=:any marked_curve(L, data, _MK_UND, _MK_FROM, _MK_TO))
    c = out.curve
    @test n_unmarked(c) == 1
    @test c.unmarked_reason == [:unexpected_gap]
    @test c.unmarked_at == [DateTime(2024, 1, 17, 21, 0)]   # the nominal 16:00 ET close
    @test c.timestamps == [_MK_CLOSES[1], _MK_CLOSES[3], _MK_CLOSES[4]]
    # A session that never printed names the underlying, not a lot, and the
    # reason it retains is the one the curve carries.
    f = only(out.failures)
    @test f.stage === :mark && f.at == DateTime(2024, 1, 17, 21, 0)
    @test f.reason === :unexpected_gap && f.subject == "SPY"
end

@testset "marked_curve: an empty ledger is a flat zero curve, not an empty one" begin
    c = marked_curve(Ledger(), _mk_data(), _MK_UND, _MK_FROM, _MK_TO).curve
    @test c.timestamps == _MK_CLOSES
    @test all(iszero, c.profit)
    @test n_unmarked(c) == 0
end

@testset "marked_curve: identical trade profits, different holding periods, different ratios" begin
    # Two ledgers whose realised trade profit is the same 45.00, held for
    # one session and for three. The session grid, not the trade count, sets
    # the observations, so the annualised path ratios must differ.
    function _mk_held(open_at, close_at)
        L = Ledger()
        g = mint_group!(L)
        _lg_fill!(L, _LG_PUT470, Short, Open,  1, 0.85, g; at=open_at,  leg_id=1)
        _lg_fill!(L, _LG_PUT470, Long,  Close, 1, 0.40, g; at=close_at, leg_id=2)
        return L
    end
    quick = _mk_held(DateTime(2024, 1, 16, 14, 35), DateTime(2024, 1, 16, 20, 0))
    slow  = _mk_held(DateTime(2024, 1, 16, 14, 35), DateTime(2024, 1, 19, 20, 0))
    @test trade_pnl(quick) ≈ [45.0]
    @test trade_pnl(slow)  ≈ [45.0]
    @test total_pnl(trade_pnl(quick)) == total_pnl(trade_pnl(slow))
    data = _mk_data()
    cq = marked_curve(quick, data, _MK_UND, _MK_FROM, _MK_TO).curve
    cs = marked_curve(slow,  data, _MK_UND, _MK_FROM, _MK_TO).curve
    # Flat at every session close: four equal levels, no dispersion at all.
    @test cq.profit ≈ [45.0, 45.0, 45.0, 45.0]
    @test session_changes(cq) ≈ [0.0, 0.0, 0.0]
    # Held across three sessions: the same realised 45.00, three real moves.
    @test cs.profit ≈ [0.0, 20.0, -40.0, 45.0]
    @test session_changes(cs) ≈ [20.0, -60.0, 85.0]
    @test volatility(Float64[], cq) == 0.0
    @test volatility(Float64[], cs) > 0.0
    @test isnan(sharpe(Float64[], cq)) && !isnan(sharpe(Float64[], cs))
    @test max_drawdown(Float64[], cq) == 0.0
    @test max_drawdown(Float64[], cs) ≈ 60.0
end
