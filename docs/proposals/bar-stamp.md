# Bar stamps: visibility at the end of the minute

The market data layer serves a completed minute's prices at the start
of that minute. This round fixes that one-minute lookahead before the
[marked-curve round](ledger-outputs-curve.md) lands. It is a separate
backtest change, on its own branch and in its own pull request, with
changed ledgers and changed run ids accepted as part of the correction.

## When the price becomes known

Polygon stamps a minute bar at its open. Its high, low and close are
known only when the minute ends. In `src/data/synth.jl:109–123`,
`synthesize(::SpreadFromOHLCV, bar)` builds bid and ask from those three
values, uses the close as the mark, and copies `bar.timestamp` into the
quote at line 121. The spot path does the same thing without synthesis:
`src/market_data/parquet.jl:406–416` loads each row's timestamp and close
together, and line 425 constructs a `SpotPrice` from that pair. These
locations have been checked against the current source.

A decision at 19:30 therefore sees the close and range of the minute
from 19:30 to 19:31. That minute has not finished. The defect gives
every bar-based fill and settlement price access to up to one minute of
future information; it is not a preference about how to label a chart.
The no-lookahead machinery in `TimeCut` is structurally sound, but the
record admitted through it claims to be knowable before it is. The
guarantee fails below the cut.

Both production trees, `options_1min` and `spots_1min`, contain
one-minute bars. Their canonical visibility time becomes the vendor row
timestamp plus one minute. A decision at 19:30 then reads the completed
19:29–19:30 bar. `OptionBar` must carry that honest visibility time too,
so that direct bar reads, synthesized quotes and derived surfaces all
obey the same cut. Synthesis continues to preserve the bar's timestamp;
it must not add a second minute.

Bar-end is **the convention**, fixed in code. There is no
`stamp = :bar_end` setting and no compatibility mode for bar-open
visibility. One of those settings would enable lookahead, so offering
both would invite an experiment to choose an incorrect clock. The
general rule for sources that need to declare a visibility assumption
does not turn a completed minute's availability into a choice.

## Carrying the convention through the reader

The correction belongs where vendor rows become canonical records.
The stored rows keep their vendor timestamps; consumers query in
visibility time. All four read shapes must agree on that boundary:
`at`, `between`, `asof` and `timestamps`, including the grid that drives
the engine. Updating only the returned records would leave searches,
SQL predicates and cached timestamp lists answering a different clock.
Exact and range queries must translate their bounds consistently with
the mapping, retaining inclusive endpoints and millisecond precision.

Partition dates need particular care. The collector partitions by local
date, and `market_data.md` already permits a one-day spill into the next
UTC date. A row stamped 23:59 becomes visible at 00:00 on the next date
without moving to another file. The partition walk must still find it,
including when the next date has no partition. Check the previous-day
candidates, the backward `asof` walk and the SQL range bounds together;
none may discard a row because its visibility date differs from its
storage date. The documented upper spill bound must distinguish raw
timestamps from shifted visibility times. The existing time ordering
between partitions and bounded, lazy range reads remain requirements.

This changes the explicit allowance in `data.md` and `market_data.md`:
replace their bar-open exception with the bar-end invariant in the
implementation commit. The change restores the visibility rule already
claimed by status entry 1. It does not weaken the rule that empty means
temporal absence only: an unserved selector, conflicting records or an
exhausted derivation still has its named failure.

## The result break is intentional

The ledger moves because decisions and fills now read completed bars.
The ten-year strangle's old baseline of 13,438 events, 2,240 orders and
USD 32,008.66 cash is a before value, not an acceptance assertion.
Record the new event count, order count and cash after the rerun, even
where a count happens to remain equal. Do not adjust the clock or the
policy to recover the old numbers.

`core_hash` must move, and every run id must move with it. Status entry
7 describes content-derived identity: changing code alone does not
change that projection, and recording a new `commit_sha` is not an
identity break. The implementation must explicitly distinguish the
corrected core convention in identity, without making it a user option
or placing it in `OutputSpec`. Stored runs cease to be reproducible
under the corrected code; that cost is accepted. Their ledgers cannot
be reused as results of the corrected backtest. This is a deliberate
break in comparability in service of the vision's reproducible research,
not a claim that the old and new runs represent the same experiment.

Settlement prices move for the same reason. The ordinary reference
window is 09:30 through 16:00 ET, inclusive. With bar-open visibility,
the row stamped 16:00 can win even though its close belongs to
16:00–16:01, after the session close. With bar-end visibility, the
15:59–16:00 bar is stamped 16:00 and wins; the vendor's 16:00 row becomes
visible at 16:01 and is outside the window. The settlement rule's text
and expiry bound need no change. Its input becomes honest.

This does not make minute aggregates official closing prices, or remove
the documented exposure to extended-hours prints on early-close days.
The official-close kind and provider remain separate backlog work.
Existing early-close measurements must be checked again against the
shifted inputs rather than treated as a guarantee about the tree.

## Why the curve waits

The curve round marks at the session close. In the blocking case, the
last option bar is stamped 20:59 UTC although it becomes knowable at
21:00. An exact quote lookup at the close therefore finds nothing.
Bar-end visibility puts that quote at 21:00, where the mark can read it
directly. No staleness window, offset constant or shift in the marking
code is needed.

The curve round's requirement that a fallback surface be stamped at the
mark instant exactly stays. A stale surface prices at a different
instant; relaxing that requirement would conceal the clock defect.
Fix the clock first, establish the new ledger baseline, then land the
curve round against it. Its output-side requirement that `core_hash`
and the ledger stay unchanged applies to that later round. It does not
constrain this earlier correction. Its regression must preserve the
new baseline, not the pre-correction numbers currently in its brief.

The implementation commit also updates status entries 1 and 7 and the
sequence of work, and removes **Bar-end timestamp convention as a spec
option** from the backlog as decided, rather than leaving it parked.
The curve implementation and official-close work are outside this
round, as is making the convention configurable.

## Tests

Tests belong beside the source they exercise, one test file per source
file. Small parquet fixtures must exercise the real readers; fixtures
made only from already-stamped in-memory records cannot catch this
mapping defect.

1. Give an option bar and a spot row distinct, known prices. Assert
   that their canonical records first become visible one minute after
   the raw stamp, and that synthesis preserves that instant and the
   expected bid, ask and mark. A cut before bar end must hide both the
   bar and its derived quote; a cut at bar end must admit them. Exercise
   derived surface reads through the same cut so no input escapes it.
2. Pin agreement among all four read shapes for both trees. Check exact
   reads, inclusive ranges, latest-visible reads and timestamp grids
   before, at and after bar end, including fractional-second bounds.
   A 23:59 row in the earlier partition must be found at next-day 00:00
   by every shape, with no next-day file required. Cover the existing
   after-midnight spill, gaps between partitions, sorted range output
   and spot duplicate handling under the shifted stamps.
3. Use different closes for the raw 15:59 and 16:00 spot rows. Settlement
   at 16:00 ET must select the former and exclude the latter. Cover
   early-close data and the existing intraday-expiry bound, retaining
   named failures for genuinely unanswerable settlement questions.
4. At a scheduled decision, use adjacent bars with different prices and
   assert that the fill uses the minute that just ended. At a session
   close, prove that an exact lookup finds the final completed option
   bar's quote, establishing the curve round's prerequisite without
   adding any tolerance to marking.
5. Pin the identity break against the prior core projection: the same
   resolved experiment gets a new `core_hash` and `full_hash`. Output
   variations still share one corrected core, and no config option can
   restore bar-open visibility. Rerun the ten-year strangle and record
   its new ledger totals, settlement results and run id as the baseline
   for the curve round.

The implementation needs focused tests and the full test gate on a
machine with sufficient memory, plus the data-backed regression. This
brief does not report those gates as run; no Julia process is needed to
write or review it on this 3.7 GB box.

## Done means

Every minute-bar read exposes completed information at bar end, and the
four shapes, derived reads and partition boundaries agree on that
instant. A cut cannot expose a minute's close or range during that
minute. Settlement uses the completed bar inside its existing window,
and the session-close quote is available to the next round without a
marking workaround.

The ledger baseline has been regenerated, `core_hash` and every run id
have changed, and the loss of reproduction of old runs under the new
code is recorded. The required gates pass. Module docs state the new
invariant, status records the identity break and sequencing, and the
spec-option backlog entry is gone. The change lands independently
before the curve round, whose unchanged-ledger requirement then starts
from this corrected baseline.
