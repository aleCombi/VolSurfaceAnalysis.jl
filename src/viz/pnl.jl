using RecipesBase

# The marked profit itself is the series: it is the run's true profit through
# time, realised and unrealised together. The only thing the recipe adds is
# the breaks -- a session the curve could not mark is plotted as a gap, so the
# picture cannot draw a straight line through a valuation failure and read as
# a quiet stretch. The `NaN` here is a plotting instruction meaning "start a
# new path", never a reported value; the failure itself is named and counted
# on the curve (design rule 7).
@recipe function f(curve::MarkedCurve)
    isempty(curve.profit) &&
        throw(ArgumentError("cannot plot a marked curve with no marked session"))
    ts = vcat(curve.timestamps, curve.unmarked_at)
    ys = vcat(curve.profit, fill(NaN, length(curve.unmarked_at)))
    o  = sortperm(ts)
    title  --> "Marked profit"
    xlabel --> "Time"
    ylabel --> "Marked profit (USD)"
    legend --> false
    ts[o], ys[o]
end
