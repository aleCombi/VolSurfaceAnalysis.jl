# The `VolatilitySurface` kind contract. Apart from `kinds.jl` because the
# type it dispatches on is defined in `pricing` after `kinds.jl` is loaded:
# records carry math objects and the surface builder consumes records, so
# the two interleave at load.

selector(s::VolatilitySurface) = s.underlying
selector_type(::Type{<:VolatilitySurface}) = Underlying
snapshot(::Type{<:VolatilitySurface}) = true
