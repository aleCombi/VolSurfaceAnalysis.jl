using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using VolSurfaceAnalysis
using Dates
using Plots
using Plots.Measures

const ROOT = get(ENV, "VSA_POLYGON_ROOT", "C:/repos/options-collector/data/massive")
const SYMBOL = get(ENV, "VSA_DEMO_SYMBOL", "SPY")
const DEMO_DATE = Date(get(ENV, "VSA_DEMO_DATE", "2024-06-03"))
const N_SLICES = parse(Int, get(ENV, "VSA_DEMO_N_SLICES", "4"))
const DELTA_TARGETS = [0.10, 0.20, 0.30]

const RATE = 0.045
const DIV  = 0.013

# The same map a config would build: bars and spots from the parquet
# trees, quotes synthesized from bars, flat curves, the surface provider.
const DATA = build_market_data(Dict{String,Any}(
    "option_bar"   => Dict{String,Any}("type" => "parquet_option_bars", "root" => joinpath(ROOT, "options_1min")),
    "option_quote" => Dict{String,Any}("type" => "from_bars",
                                       "synthesizer" => Dict{String,Any}("type" => "ohlcv_spread", "lambda" => 0.7)),
    "spot_price"   => Dict{String,Any}("type" => "parquet_spots", "root" => joinpath(ROOT, "spots_1min")),
    "rate_curve"   => Dict{String,Any}("type" => "constant", "currency" => "USD", "value" => RATE),
    "div_curve"    => Dict{String,Any}("type" => "constant", "underlying" => SYMBOL, "value" => DIV),
    "vol_surface"  => Dict{String,Any}("type" => "surface_from", "currency" => "USD"),
))

function pick_midday_ts(d, u::Underlying, date::Date)
    ts = timestamps(d, OptionQuote, u, DateTime(date), DateTime(date, Time(23, 59, 59)))
    isempty(ts) && error("no chain timestamps on $date for $u")
    target = DateTime(date, Time(16, 30))  # ~12:30 ET
    _, i = findmin(t -> abs(Dates.value(t - target)), ts)
    return ts[i]
end

with_data(DATA) do d
    u = Underlying(SYMBOL)
    ts = pick_midday_ts(d, u, DEMO_DATE)
    surf = only_or_missing(at(d, VolatilitySurface, u, ts))
    ismissing(surf) && error("no surface built at $ts")

    spot = surf.spot
    all_exps = expiries(surf)
    n = min(N_SLICES, length(all_exps))
    exps = all_exps[1:n]
    @info "surface built" ts spot n_expiries=length(all_exps) showing=n rate=RATE div=DIV

    iv_panel = plot(;
        title="$SYMBOL @ $(ts) UTC  (S=$(round(spot, digits=2)))",
        ylabel="Implied vol",
        legend=:topright,
        bottom_margin=2mm)

    delta_panel = plot(;
        xlabel="Strike",
        ylabel="|delta|",
        legend=false,
        bottom_margin=6mm)

    colors = palette(:tab10)

    for (i, e) in enumerate(exps)
        sl = get_slice(surf, e)
        sl === nothing && continue
        dte = round(sl.tau * 365.25, digits=1)
        label = "$(Date(e))  ($(dte)d)"
        c = colors[mod1(i, length(colors))]

        plot!(iv_panel, sl.strikes, sl.ivs;
              label=label, color=c, marker=:circle, markersize=2, linewidth=1.5)

        # Fine K grid across the slice's observed strike range for a smooth
        # delta curve. Puts to the left of spot, calls to the right.
        K_grid = range(sl.strikes[1], sl.strikes[end]; length=200)
        absdeltas = map(K_grid) do K
            otype = K >= spot ? Call : Put
            abs(delta(surf, e, K, otype))
        end
        plot!(delta_panel, K_grid, absdeltas;
              color=c, linewidth=1.5, label=label)

        # Mark each delta-target intersection on the call wing and the put wing.
        for target in DELTA_TARGETS
            K_call = invert_delta(surf, e, Call, target)
            K_put  = invert_delta(surf, e, Put,  target)
            for K in (K_call, K_put)
                K === nothing && continue
                scatter!(delta_panel, [K], [target];
                         color=c, marker=:circle, markersize=4,
                         markerstrokewidth=0, label=false)
            end
        end
    end

    vline!(iv_panel, [spot]; color=:black, linestyle=:dash, linewidth=1, label="spot")
    vline!(delta_panel, [spot]; color=:black, linestyle=:dash, linewidth=1, label=false)
    for target in DELTA_TARGETS
        hline!(delta_panel, [target]; color=:gray, linestyle=:dot, linewidth=1, label=false)
    end

    p = plot(iv_panel, delta_panel;
             layout=(2, 1), link=:x, size=(900, 700))

    save_path = get(ENV, "VSA_DEMO_SAVE", "")
    if !isempty(save_path)
        savefig(p, save_path)
        @info "saved figure" save_path
    else
        display(p)
    end
end
