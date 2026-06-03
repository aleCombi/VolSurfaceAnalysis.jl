# Script-level artifact rendering. Lives here, not in the package, so the
# core stays Plots-free -- the eventual live-trading split shares the
# modelling/strategy layers without pulling in Plots. The runner `include`s
# this only when it actually needs to render (--save / --out-dir).
#
# Renderers map an artifact id (the symbols in `OutputSpec.artifacts`) to a
# function (result, dir) -> path | nothing. Regenerating the artifacts of a
# stored run is `render_artifacts(load_run(store, id), dir)`.

using VolSurfaceAnalysis
using Plots

function _render_equity_curve(result::ExperimentResult, dir::AbstractString)
    s = result.pnl_series
    isempty(s.pnl) && return nothing
    p = plot(s; title=result.experiment.name, linewidth=1.5, size=(1000, 500),
             left_margin=Plots.Measures.Length(:mm, 5),
             bottom_margin=Plots.Measures.Length(:mm, 5))
    hline!(p, [0.0]; color=:gray, linestyle=:dot, linewidth=1, label=false)
    path = joinpath(dir, "equity_curve.png")
    savefig(p, path)
    return path
end

# artifact id -> renderer. Mirrors the metric dispatch table; grow it as new
# plot/export types land.
const ARTIFACT_RENDERERS = Dict{Symbol,Function}(
    :equity_curve => _render_equity_curve,
)

"""
    render_artifacts(result, dir; artifacts=result.experiment.outputs.artifacts)

Render each declared artifact into `dir` (created if absent); return the
paths written. Renderers that produce nothing (e.g. an equity curve for an
empty series) are skipped; unknown artifact ids error.
"""
function render_artifacts(result::ExperimentResult, dir::AbstractString;
                          artifacts=result.experiment.outputs.artifacts)
    mkpath(dir)
    written = String[]
    for a in artifacts
        fn = get(ARTIFACT_RENDERERS, a) do
            error("render_artifacts: unknown artifact :$a. " *
                  "Known: $(sort(collect(keys(ARTIFACT_RENDERERS))))")
        end
        path = fn(result, dir)
        path === nothing || push!(written, path)
    end
    return written
end
