# Code provenance for the knowledge base: the git commit the running code
# is at, plus whether the working tree is dirty. Stored on every saved run
# so a run can be tied to the exact code that produced it -- the basis for
# safe cache reuse (a cached backtest is only reusable at the same commit
# with a clean tree) and for retracting every run from a flawed code era.
#
# Git is invoked in the package's own repo (via @__DIR__), not the caller's
# cwd. Any failure (no git, not a repo) yields ("", true): an unknown commit
# and a dirty flag -- i.e. "never safe to reuse."

"""
    code_provenance() -> (commit_sha::String, dirty::Bool)

Current git commit SHA of the package repo and whether its working tree
has uncommitted changes. On any git failure returns `("", true)` --
unknown commit, treated as dirty (uncacheable).
"""
function code_provenance()::Tuple{String,Bool}
    root = normpath(joinpath(@__DIR__, "..", ".."))
    sha = try
        strip(read(`git -C $root rev-parse HEAD`, String))
    catch
        return ("", true)
    end
    dirty = try
        !isempty(strip(read(`git -C $root status --porcelain`, String)))
    catch
        true
    end
    return (String(sha), dirty)
end

# The dependency environment, as Pkg itself records it. A resolved
# `Manifest.toml` beside the project is the environment record: the
# versions actually loaded, not a wish list and not an archive of the
# source behind a path dependency (see the module doc's *Conventions
# consulted*). It is copied, never re-resolved, so a saved run keeps the
# manifest that produced it rather than whatever today's registry would
# pick.

"""
    MissingManifest

The active environment has a project but no resolved `Manifest.toml`, so
the run's dependency versions cannot be recorded. `path` is where one was
looked for. Design rule 7: a save that cannot say what it ran against
stops, rather than writing an empty document that reads on disk exactly
like a run with no dependencies.
"""
struct MissingManifest <: Exception
    path::String
end

Base.showerror(io::IO, e::MissingManifest) = print(io,
    "MissingManifest: no resolved Manifest.toml at ", e.path,
    "; instantiate the environment before saving a run")

"""
    dependency_manifest() -> String

The bytes of the active environment's `Manifest.toml`, verbatim. Throws
[`MissingManifest`](@ref) when there is none -- the failure is named
before anything is written, so a run folder is never created around a
dependency record that does not exist.
"""
function dependency_manifest()::String
    project = Base.active_project()
    project === nothing && throw(MissingManifest("(no active project)"))
    path = joinpath(dirname(project), "Manifest.toml")
    isfile(path) || throw(MissingManifest(path))
    return read(path, String)
end
