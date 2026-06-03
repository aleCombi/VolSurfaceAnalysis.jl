using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using VolSurfaceAnalysis

# Run an experiment from a TOML config and print the result.
#   --save          persist to the knowledge base (scripts/runs/, gitignored),
#                   including the default output artifacts, tagged with code provenance
#   --out-dir <dir> render the output artifacts to <dir> without persisting
#
#   julia --project=. scripts/run_experiment.jl <config.toml> [--save] [--out-dir <dir>]

function _parse_args(args)
    do_save = false
    out_dir = nothing
    positional = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--save"
            do_save = true
        elseif a == "--out-dir"
            i += 1
            i <= length(args) || error("--out-dir requires a path")
            out_dir = args[i]
        else
            push!(positional, a)
        end
        i += 1
    end
    length(positional) == 1 || error(
        "usage: julia --project=. scripts/run_experiment.jl <config.toml> [--save] [--out-dir <dir>]")
    return (config=positional[1], save=do_save, out_dir=out_dir)
end

opts = _parse_args(ARGS)

# Read the raw bytes and build from them, so the config.toml persisted with
# the run is exactly what produced it.
config_toml = read(opts.config, String)
exp = load_experiment_str(config_toml)
res = run_experiment(exp)

show(stdout, MIME"text/plain"(), res)
println()

# Materialising outputs (save or scratch render) needs Plots; load it and
# the renderers only on that path so plain runs stay light. This is its own
# top-level statement so the render calls below see the loaded methods.
need_render = opts.save || (opts.out_dir !== nothing)
need_render && include(joinpath(@__DIR__, "lib", "artifacts.jl"))

if opts.save
    sha, dirty = code_provenance()
    store_root = @__DIR__                       # -> scripts/runs/run_id=<id>/ (gitignored)
    id = with_run_store(store_root) do store
        save_run(store, res, config_toml; commit_sha=sha, dirty=dirty)
    end
    art_dir = joinpath(store_root, "runs", "run_id=" * id, "artifacts")
    paths = render_artifacts(res, art_dir)
    @info "saved run" run_id=id core_hash=core_hash(exp) dirty=dirty artifacts=length(paths) store=joinpath(store_root, "runs")
    dirty && @warn "working tree is dirty -- this run is not safe to reuse as a cache donor"
end

if opts.out_dir !== nothing
    paths = render_artifacts(res, opts.out_dir)
    @info "rendered artifacts" out_dir=opts.out_dir artifacts=length(paths)
end
