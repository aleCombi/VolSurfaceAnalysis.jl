# scripts/lib/experiment.jl
#
# Shared helpers for experiment scripts.
# Requires: VolSurfaceAnalysis, Dates, Printf, Statistics

# =============================================================================
# Result type and summary printer
# =============================================================================

const ResultRow = @NamedTuple begin
    symbol::String
    features::String
    variant::String
    seed::Int
    sharpe::Float64
    sortino::Float64
    roi::Float64
    trades::Int
    win_rate::Float64
    pnl::Float64
end

function print_summary(results::Vector{ResultRow}, symbols::Vector{String})
    isempty(results) && return

    methods = Tuple{String,String}[]
    seen = Set{Tuple{String,String}}()
    for r in results
        key = (r.features, r.variant)
        if key ∉ seen; push!(seen, key); push!(methods, key); end
    end

    multi_feat = length(unique(m[1] for m in methods if m[1] != "\u2014")) > 1
    _label(f, v) = multi_feat && f != "\u2014" ? "$v ($f)" : v

    nw = max(16, maximum(length(_label(m...)) for m in methods) + 2)
    cw = 18
    total_w = nw + 2 + (cw + 2) * length(symbols)

    println("\n", "=" ^ total_w)

    for (title, field, fmt) in [
        ("Sharpe", :sharpe, (a, s) -> s > 0 ? @sprintf("%.2f \u00b1 %.2f", a, s) : @sprintf("%.2f", a)),
        ("ROI",    :roi,    (a, s) -> s > 0 ? @sprintf("%.1f%% \u00b1 %.1f%%", a*100, s*100) : @sprintf("%.1f%%", a*100)),
        ("Trades", :trades, (a, s) -> @sprintf("%.0f", a)),
    ]
        print("  ", rpad(title, nw))
        for sym in symbols; print("  ", lpad(sym, cw)); end
        println()
        println("  ", "-" ^ (nw + (cw + 2) * length(symbols)))

        for (f, v) in methods
            print("  ", rpad(_label(f, v), nw))
            for sym in symbols
                vals = [Float64(getfield(r, field)) for r in results
                        if r.symbol == sym && r.features == f && r.variant == v]
                if isempty(vals)
                    print("  ", lpad("-", cw))
                else
                    avg = mean(vals)
                    sd = length(vals) > 1 ? std(vals) : 0.0
                    print("  ", lpad(fmt(avg, sd), cw))
                end
            end
            println()
        end
        println()
    end

    println("=" ^ total_w)
end
