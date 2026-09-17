# The `VolatilitySurface` kind contract.
#
# It sits apart from `kinds.jl` because the surface type it dispatches on is a
# math object owned by `pricing`, and `kinds.jl` is included before that type
# exists: records carry math objects, and the surface builder consumes a
# record, so the two files interleave at load.

selector(s::VolatilitySurface) = s.underlying
selector_type(::Type{<:VolatilitySurface}) = Underlying
snapshot(::Type{<:VolatilitySurface}) = true
