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
