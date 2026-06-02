using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using VolSurfaceAnalysis
using Dates
using Plots

# --- Configuration (env-overridable) ----------------------------------------
const ROOT       = get(ENV, "VSA_POLYGON_ROOT", "C:/repos/options-collector/data/massive")
const SYMBOL     = get(ENV, "VSA_DEMO_SYMBOL", "SPY")
const FROM       = DateTime(get(ENV, "VSA_FROM", "2026-02-27T00:00:00"))
const TO         = DateTime(get(ENV, "VSA_TO",   "2026-03-27T23:59:59"))
const ENTRY      = Time(get(ENV, "VSA_ENTRY",    "19:30:00"))
const PUT_DELTA  = parse(Float64, get(ENV, "VSA_PUT_DELTA",  "0.16"))
const CALL_DELTA = parse(Float64, get(ENV, "VSA_CALL_DELTA", "0.16"))
const EXPIRY_DAYS = parse(Int,    get(ENV, "VSA_EXPIRY_DAYS", "1"))
const QUANTITY   = parse(Float64, get(ENV, "VSA_QUANTITY",   "1.0"))
const RATE       = parse(Float64, get(ENV, "VSA_RATE",       "0.045"))
const DIV        = parse(Float64, get(ENV, "VSA_DIV",        "0.013"))
const LAMBDA     = parse(Float64, get(ENV, "VSA_LAMBDA",     "0.7"))
const SAVE_PATH  = get(ENV, "VSA_SAVE_PATH", "C:/tmp/strangle_equity.png")

# --- Build experiment programmatically --------------------------------------
underlying = Underlying(SYMBOL)
raw_source = ParquetDataSource(SYMBOL, ROOT; synthesizer=SpreadFromOHLCV(LAMBDA))
mds = ModelDataSource(raw_source; rate=FlatCurve(RATE), div=FlatCurve(DIV))

policy = DailyShortStrangle(;
    underlying      = underlying,
    entry_time      = ENTRY,
    expiry_interval = Day(EXPIRY_DAYS),
    put_delta       = PUT_DELTA,
    call_delta      = CALL_DELTA,
    quantity        = QUANTITY,
)

exp = Experiment(
    name    = "strangle_$(SYMBOL)_$(round(Int, 100*PUT_DELTA))d_$(EXPIRY_DAYS)dte",
    agent   = StaticAgent(policy),
    source  = mds,
    from    = FROM,
    to      = TO,
    metrics = [:sharpe, :sortino, :max_drawdown, :volatility, :profit_factor],
)

@info "running experiment" name=exp.name from=FROM to=TO put_delta=PUT_DELTA call_delta=CALL_DELTA entry=ENTRY expiry_days=EXPIRY_DAYS

# --- Run + time -------------------------------------------------------------
t0 = time()
res = try
    run_experiment(exp)
finally
    close(raw_source)
end
elapsed = time() - t0

# --- Report -----------------------------------------------------------------
println()
show(stdout, MIME"text/plain"(), res)
println()
@info "completed" elapsed_seconds=round(elapsed; digits=2) n_positions=length(res.positions) n_round_trips=length(res.pnl_series.pnl)

# --- Plot equity curve ------------------------------------------------------
if isempty(res.pnl_series.pnl)
    @warn "no round trips produced -- skipping plot"
else
    p = plot(res.pnl_series;
             title="$(SYMBOL) short strangle  ($(round(Int, 100*PUT_DELTA))Δp / $(round(Int, 100*CALL_DELTA))Δc, $(EXPIRY_DAYS)-DTE)",
             linewidth=1.5,
             size=(1000, 500),
             left_margin=Plots.Measures.Length(:mm, 5),
             bottom_margin=Plots.Measures.Length(:mm, 5))
    hline!(p, [0.0]; color=:gray, linestyle=:dot, linewidth=1, label=false)
    savefig(p, SAVE_PATH)
    @info "saved equity curve" SAVE_PATH
end
