# Visualization helpers.
#
# Currently exports `plot_smile_with_condors`, which produces a two-panel
# figure: top panel is the put and call smile for a given expiry annotated with
# Black-76 deltas; bottom panel is a strike-axis diagram showing one or more
# fixed-delta iron condors.

using Plots

"""
    CondorSpec

Specification of a delta-based iron condor for visualization.

# Fields
- `short_delta::Float64`  — absolute target delta of the short put and short call
- `long_delta::Float64`   — absolute target delta of the long put and long call
- `color::Symbol`         — Plots color symbol (e.g. `:firebrick`)
- `label::String`         — short label shown next to the structure (e.g. `"30Δ / 10Δ"`)
"""
struct CondorSpec
    short_delta::Float64
    long_delta::Float64
    color::Symbol
    label::String
end

"""
    plot_smile_with_condors(surface, expiry, condor_specs;
                            rate=0.0, div_yield=0.0,
                            atm_window=0.05, title="") -> Plots.Plot

Build a composite figure for a single (snapshot, expiry):

1. **Top panel** — put smile (red) and call smile (green) for strikes within
   `±atm_window` of `surface.spot`, with each market point annotated with its
   Black-76 delta computed using `(rate, div_yield)`.

2. **Bottom panel** — for each `CondorSpec`, render one row that shows the
   four strikes picked by `delta_condor_selector(short_δ, short_δ, long_δ, long_δ)`:
   - **open circles** at the long (bought) wings,
   - **filled triangles** (▼ short put, ▲ short call) at the short legs,
   - the **profit zone** between the shorts shaded in the spec's color,
   - strike values labeled above each marker.
   The bottom panel shares the strike axis with the top.

The function does not write to disk — the caller decides what to do with the
returned `Plots.Plot`.

# Arguments
- `surface`: a `VolatilitySurface` for one snapshot
- `expiry::DateTime`: which expiry to plot
- `condor_specs::AbstractVector{CondorSpec}`: condors to overlay (use `[]` to skip)
- `rate::Float64=0.0`, `div_yield::Float64=0.0`: forward & rate for delta computation
- `atm_window::Float64=0.05`: half-width as a fraction of spot
- `title::String=""`: optional title; a sensible default is generated if empty
"""
function plot_smile_with_condors(
    surface::VolatilitySurface,
    expiry::DateTime,
    condor_specs::AbstractVector{CondorSpec};
    rate::Float64 = 0.0,
    div_yield::Float64 = 0.0,
    atm_window::Float64 = 0.05,
    title::AbstractString = "",
)::Plots.Plot
    spot = surface.spot
    T = time_to_expiry(expiry, surface.timestamp)
    T > 0 || throw(ArgumentError("Expiry $expiry is at or before snapshot timestamp $(surface.timestamp)"))
    F = spot * exp((rate - div_yield) * T)

    expiry_recs = filter(r -> r.expiry == expiry, surface.records)
    isempty(expiry_recs) && throw(ArgumentError("No records for expiry $expiry on this surface"))

    K_lo = (1.0 - atm_window) * spot
    K_hi = (1.0 + atm_window) * spot

    Kp, σp, δp = _smile_for_side(filter(r -> r.option_type == Put,  expiry_recs), F, T, K_lo, K_hi, rate)
    Kc, σc, δc = _smile_for_side(filter(r -> r.option_type == Call, expiry_recs), F, T, K_lo, K_hi, rate)

    smile_title = isempty(title) ?
        "smile  $(surface.timestamp)  T=$(round(T*365.25, digits=2))d  spot=$(round(spot, digits=2))" :
        title
    smile_plt = plot(;
        xlabel = "strike K", ylabel = "IV (%)",
        title = smile_title,
        legend = :topright,
    )
    if !isempty(Kp)
        plot!(smile_plt, Kp, 100 .* σp; lw = 1.5, marker = :circle, ms = 6,
              color = :firebrick, label = "puts σ(K)")
    end
    if !isempty(Kc)
        plot!(smile_plt, Kc, 100 .* σc; lw = 1.5, marker = :circle, ms = 6,
              color = :seagreen,  label = "calls σ(K)")
    end
    vline!(smile_plt, [spot]; color = :gray, ls = :dash, label = "spot")

    all_sigs = vcat(σp, σc)
    if !isempty(all_sigs)
        y_off = 0.04 * (maximum(100 .* all_sigs) - minimum(100 .* all_sigs) + 1e-9)
        for i in eachindex(Kp)
            annotate!(smile_plt, Kp[i], 100 * σp[i] - y_off,
                      text(_fmt_delta(δp[i]), 8, :firebrick, :center))
        end
        for i in eachindex(Kc)
            annotate!(smile_plt, Kc[i], 100 * σc[i] + y_off,
                      text(_fmt_delta(δc[i]), 8, :seagreen, :center))
        end
    end

    isempty(condor_specs) && return smile_plt

    # Pick condor strikes via the production selector, against a minimal context.
    source = DictDataSource(Dict(surface.timestamp => surface), Dict{DateTime,Float64}())
    ctx = StrikeSelectionContext(surface, expiry, source)

    picked = NamedTuple[]
    for spec in condor_specs
        sel = delta_condor_selector(spec.short_delta, spec.short_delta,
                                    spec.long_delta,  spec.long_delta;
                                    rate = rate, div_yield = div_yield)
        strikes = sel(ctx)
        strikes === nothing && continue
        sp_K, sc_K, lp_K, lc_K = strikes
        push!(picked, (spec = spec, lp = lp_K, sp = sp_K, sc = sc_K, lc = lc_K))
    end

    isempty(picked) && return smile_plt

    n = length(picked)
    struct_plt = plot(;
        xlabel = "strike K", ylabel = "",
        yticks = (1:n, [p.spec.label for p in picked]),
        ylim = (0.4, n + 0.6),
        title = "iron condor structures (bought ◯, sold ▼/▲, profit zone shaded)",
        legend = false,
        xlim = xlims(smile_plt),
    )
    vline!(struct_plt, [spot]; color = :gray, ls = :dash, lw = 1.2)

    for (i, p) in enumerate(picked)
        y = Float64(i)
        c = p.spec.color
        # Profit zone (between shorts)
        plot!(struct_plt, [p.sp, p.sp, p.sc, p.sc], [y - 0.25, y + 0.25, y + 0.25, y - 0.25];
              seriestype = :shape, fillcolor = c, fillalpha = 0.18, lw = 0)
        # Spread legs
        plot!(struct_plt, [p.lp, p.sp], [y, y]; lw = 3, color = c)
        plot!(struct_plt, [p.sc, p.lc], [y, y]; lw = 3, color = c)
        # Markers: longs are open, shorts are filled triangles
        scatter!(struct_plt, [p.lp], [y]; marker = :circle,    ms = 10, mc = :white, msc = c, msw = 2)
        scatter!(struct_plt, [p.lc], [y]; marker = :circle,    ms = 10, mc = :white, msc = c, msw = 2)
        scatter!(struct_plt, [p.sp], [y]; marker = :dtriangle, ms = 12, mc = c,      msc = :black, msw = 1)
        scatter!(struct_plt, [p.sc], [y]; marker = :utriangle, ms = 12, mc = c,      msc = :black, msw = 1)
        # Strike labels
        annotate!(struct_plt, p.lp, y + 0.32, text("$(round(Int, p.lp))", 8, :black, :center))
        annotate!(struct_plt, p.sp, y + 0.32, text("$(round(Int, p.sp))", 9, c,      :center))
        annotate!(struct_plt, p.sc, y + 0.32, text("$(round(Int, p.sc))", 9, c,      :center))
        annotate!(struct_plt, p.lc, y + 0.32, text("$(round(Int, p.lc))", 8, :black, :center))
    end

    return plot(smile_plt, struct_plt; layout = grid(2, 1, heights = [0.72, 0.28]))
end

# Per-side smile extraction. Filters the records to a strike window and computes
# (K, σ, Δ) using the record's own market IV.
function _smile_for_side(side_recs, F::Float64, T::Float64,
                        K_lo::Float64, K_hi::Float64, rate::Float64)
    Ks     = Float64[]
    sigs   = Float64[]
    deltas = Float64[]
    for rec in side_recs
        K_lo <= rec.strike <= K_hi || continue
        ismissing(rec.mark_iv) && continue
        σ = Float64(rec.mark_iv) / 100.0
        (σ <= 0.0 || σ > 10.0) && continue
        d = _delta_from_record(rec, F, T, rate)
        d === missing && continue
        push!(Ks,     rec.strike)
        push!(sigs,   σ)
        push!(deltas, Float64(d))
    end
    ord = sortperm(Ks)
    return Ks[ord], sigs[ord], deltas[ord]
end

# Format a delta for in-plot annotation.
_fmt_delta(d::Real) = string("Δ=", d >= 0 ? "+" : "", round(d, digits = 2))
