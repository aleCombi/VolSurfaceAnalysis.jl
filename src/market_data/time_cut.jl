# `market_data` module: the time cut.
#
# `TimeCut` wraps a map and masks every shape at `cutoff`, filtering on
# visibility time only. It passes ITSELF down as the context, so a derived
# provider's input reads go through the cut too: no-lookahead through
# derived data is structural, not a call-site convention. Because
# `timestamp` is visibility time, the cut is the complete no-lookahead
# rule -- nothing knowable after the cutoff is visible, whatever its
# effective date.

"""
    TimeCut(inner, cutoff)

The map `inner` with every shape masked at `cutoff` (inclusive): `at`
past the cutoff is empty, `between` and `timestamps` clamp `to` to the
cutoff, `asof` clamps `ts`. The engine builds one per tick and hands it
to `decide`.
"""
struct TimeCut{M}
    inner::M
    cutoff::DateTime
end

entry(c::TimeCut, ::Type{R}) where {R} = entry(c.inner, R)

at(c::TimeCut, ::Type{R}, sel, ts::DateTime) where {R} =
    ts <= c.cutoff ? at(entry(c, R), c, R, sel, ts) : R[]
between(c::TimeCut, ::Type{R}, sel, from::DateTime, to::DateTime) where {R} =
    from <= c.cutoff ? between(entry(c, R), c, R, sel, from, min(to, c.cutoff)) : R[]
asof(c::TimeCut, ::Type{R}, sel, ts::DateTime) where {R} =
    asof(entry(c, R), c, R, sel, min(ts, c.cutoff))
timestamps(c::TimeCut, ::Type{R}, sel, from::DateTime, to::DateTime) where {R} =
    from <= c.cutoff ? timestamps(entry(c, R), c, R, sel, from, min(to, c.cutoff)) : DateTime[]
