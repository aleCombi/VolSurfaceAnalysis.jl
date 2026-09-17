# `data/protocol`: the time cut. It passes ITSELF down as the context --
# keep it that way: that is what puts a derived provider's own input reads
# under the cut.

"""
    TimeCut(inner, cutoff)

The map `inner` with every shape masked at `cutoff` (inclusive): `at`
past the cutoff is empty, `between` and `timestamps` clamp `to` to the
cutoff, `asof` clamps `ts`. `serves` is not masked.
"""
struct TimeCut{M}
    inner::M
    cutoff::DateTime
end

entry(c::TimeCut, ::Type{R}) where {R} = entry(c.inner, R)

serves(c::TimeCut, ::Type{R}, sel) where {R} = serves(entry(c, R), c, R, sel)

# The structural check runs BEFORE the mask, so a cut cannot turn a broken
# configuration into silence.
function at(c::TimeCut, ::Type{R}, sel, ts::DateTime) where {R}
    _require_served(c, R, sel)
    ts <= c.cutoff ? at(entry(c, R), c, R, sel, ts) : R[]
end
function between(c::TimeCut, ::Type{R}, sel, from::DateTime, to::DateTime) where {R}
    _require_served(c, R, sel)
    from <= c.cutoff ? between(entry(c, R), c, R, sel, from, min(to, c.cutoff)) : R[]
end
function asof(c::TimeCut, ::Type{R}, sel, ts::DateTime) where {R}
    _require_served(c, R, sel)
    asof(entry(c, R), c, R, sel, min(ts, c.cutoff))
end
function timestamps(c::TimeCut, ::Type{R}, sel, from::DateTime, to::DateTime) where {R}
    _require_served(c, R, sel)
    from <= c.cutoff ? timestamps(entry(c, R), c, R, sel, from, min(to, c.cutoff)) : DateTime[]
end
