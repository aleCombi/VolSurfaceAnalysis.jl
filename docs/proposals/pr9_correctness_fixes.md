# PR #9 correctness fixes

Status: **implemented** on `data-kinds`, in the ten-commit sequence of
[pr9_implementation_plan.md](pr9_implementation_plan.md) (its Contradictions
section records the six places this document disagreed with itself or with the
source, and how each was resolved). One section per confirmed correctness
finding from the review of `data-kinds` (PR #9). Cleanup and efficiency findings
are out of scope.

Red specifications live in `test/regressions/test_review_findings.jl`. They were
excluded from the gate while they were red; all six are green and the file is in
`test/runtests.jl`.

All six findings are decided. See the implementation order and the test gaps at
the end.

## Note for the implementing agent: these are one defect, not six

Five of the six findings are the same mistake. A question that cannot be
answered is reported as an ordinary empty result, and every consumer downstream
correctly concludes it has nothing to do.

The states that were collapsed into one empty vector:

| State | Finding | Decided outcome |
|---|---|---|
| nothing serves this selector | 2, 3 | error, named |
| nothing at this instant | -- | empty, the only legitimate case |
| input present, derivation failed | 4 | walk back, bounded; error past the bound |
| two rows, two answers | 5 | collapse if identical, error if conflicting |
| not yet visible at this instant | 6 | empty, honouring the stamp |

Read the sections as one change of stance rather than six repairs: **empty is
reserved for "served, and nothing at this instant". Every other unanswerable
question gets a name.** Where a section's wording and this table disagree, the
table is the intent.

This is also a candidate standing rule for `docs/design.md`. It is deliberately
not written there yet, because design rule 3 requires proposing a rule change
explicitly before applying it. Do not add it as part of implementing this plan.

---

---

## 1. Settlement follows the clock instead of the trade

### What is wrong

`run_experiment` takes one underlying from `exp.clock.sel` and uses it for three
different purposes: choosing the window-end tick, reading the window-end spot,
and settling every residual lot. Only the first is a question about the clock.

The fill side of the engine, changed in this same PR, reads
`at(cut, SpotPrice, trd.underlying, t)`. So entry is priced per trade and exit is
priced per experiment. A QQQ call bought under a SPY clock fills against QQQ and
settles against SPY.

The comment above `_build_settle` still claims settlement uses the clock
selector "the same simplification as at fill time". That claim was true before
this PR and is false after it.

### Root cause

A clock is a tick grid: a kind plus a selector, meaning step wherever records of
this kind exist for this selector. Its selector answers *when*. The orchestrator
reads it as an answer to *whose price*.

The coupling is held in place by the settle callback's signature. It is
`settle(expiry::DateTime) -> Union{Float64, Missing}`, so the one underlying
captured when the closure is built applies to every lot. The series builder is
holding the lot when it calls the callback, with the underlying one field away
from the expiry it passes, and the signature discards it.

### Decision

Both halves, as agreed:

- **The trade underlying leads.** Settlement resolves the spot per lot, from the
  lot's own trade, matching what the engine already does at fill time.
- **The clock underlying must match the trade underlying.** A single-underlying
  experiment is the real invariant of this codebase. It is asserted rather than
  assumed.

The rejected alternative was to treat the clock selector and the traded
underlying as independent axes, allowing a run to tick on one instrument's grid
while trading another. Nothing in the repo supports that today, and designing
around it would protect an accidental coupling instead of a real invariant.

### Fix

**The trade leads.** Widen the settle contract so it receives the lot's trade
rather than only its expiry, and look the spot up for that trade's own selector.

- Add `selector(t::Trade) = t.underlying` so the lookup is expressed in the
  module's existing vocabulary rather than as a bare field access. The trait is
  already defined for every record kind; the position side has no method today.
- `_build_settle` resolves the spot per lot for **both** branches. The
  in-window branch already does a lookup and simply changes its selector. The
  past-the-window branch currently returns a single `Float64` computed once from
  the clock, and must become a per-selector lookup at `window_end`.
- `PnLSeries.window_end_spot` stays as it is, a provenance record of the run's
  reference underlying. It must stop being the value that marks foreign lots.

**The match is enforced.**

- Add a trait on `Policy` reporting the underlyings a policy declares
  statically, returning empty when it declares none. `NoOpPolicy` declares none;
  `DailyShortStrangle` declares its `underlying` field.
- `load_experiment` errors when a declared underlying differs from the clock
  selector.
- Optionally the engine compares `trd.underlying` against the clock selector at
  fill time. That catches policies which choose late and cannot be checked at
  load. Treat it as a follow-up, not part of this change.

Enforcing the match is what makes the past-the-window branch safe by
construction, which no signature change reaches on its own.

### Ripple

- `docs/modules/metrics.md`: the settle callback signature is public.
- `docs/modules/experiment.md`: the settlement contract and the load-time checks.
- `docs/modules/policies.md`: the new trait.
- Delete the stale parity claim above `_build_settle`.

### Test gap

The red test covers one branch of two. Its expiry lands exactly on the window
end, so it exercises only the in-window settlement path. Fixing the callback
alone would turn it green while leaving the past-the-window branch marking
foreign lots at the clock underlying's spot. Add a case with an expiry after the
window end before treating this finding as specified.

---

## 2 and 3. Structural absence is not temporal absence

These two findings are one defect seen from two sides, and one decision settles
both.

### What is wrong

Finding 2: `build_market_data` verifies that every derived spec's input *kinds*
are present, never that the selectors line up. A `SurfaceFrom` configured for EUR
reading a `Constant(RateCurve(USD, ...))` loads cleanly, produces an empty
surface at every tick, caches the empty result, and completes the run with no
positions and no diagnostic. Every layer behaves correctly and the result is a
silent no-op.

Finding 3: `BySelector` throws `KeyError` for a selector it does not route,
which contradicts the protocol rule that empty means absent for all four shapes.
A `spot_price` entry routed only for SPX, under a SPY clock, dies mid-run with a
raw `KeyError` instead of a diagnostic.

### Root cause

The protocol collapses two different situations into one empty vector.

- **Temporal absence.** This selector has no record at this instant. An ordinary
  runtime fact that varies with the timestamp.
- **Structural absence.** Nothing in this configuration serves this selector at
  all. A static property that varies with nothing.

A consumer cannot tell "not yet" from "not ever". The surface reader asks for a
EUR rate curve, gets empty, and correctly concludes it cannot build. It has no
way to know the emptiness is permanent and the run is already dead.

`BySelector` is the one provider that does distinguish them. It is the right
observation, and today it is the only one making it.

### Decision

**Empty means served but nothing at this time. Asking for a selector nobody
serves is an error.**

Every provider can answer "do I serve this selector" cheaply:

- `Constant` holds one selector in its record.
- `BySelector` holds its routes.
- `InMemory` has the selectors present in its rows.
- The parquet readers list partitions per underlying by walking the `date=`
  directories for a file under that `symbol=`, already cached per reader. An
  empty partition list means the tree holds nothing for that underlying.
- Derived providers do not answer. They delegate, and let their input's error
  propagate, so the failure names the real cause rather than reporting a missing
  derived record.

This makes the protocol's single representation of absence honest rather than
merely stated, and makes `BySelector` consistent with everything else instead of
removing the one correct behaviour.

The rejected alternative was to keep empty ambiguous at runtime and catch the
config case at load time only. It leaves selectors chosen at runtime unchecked,
and it leaves the parquet case uncovered on the mistaken belief that a parquet
reader cannot answer.

### Fix

- Add a shape reporting whether a provider serves a selector, implemented for
  each raw provider and delegated by derived ones.
- Raise a dedicated error rather than `KeyError`. It must name the kind, the
  selector asked for, and what is served instead. A bare key error naming only
  the selector does not say enough to fix a config.
- Keep a load-time check in `build_market_data` for the closed-world providers.
  It is no longer the mechanism, only the fast path, so a mistyped currency
  fails in a second rather than after a backtest has been running.

### Ripple

- `docs/modules/market_data.md`: the protocol rule on absence, which currently
  says empty means absent for all four shapes.
- The `BySelector` docstring, which promises `KeyError` specifically.
- `docs/modules/experiment.md`: the load-time check list.

### Open question

`InMemory` is the fixture provider, used throughout the test suite. Its rows are
the whole world, so no rows for a selector does look structural, and making it
throw is the consistent choice. Some existing tests may rely on an empty result
for a selector with no rows in the window. Expect this to cost a pass through the
suite, and decide there rather than in the abstract.

---

## 4. Surface `asof` does not walk back past an unbuildable chain

### What is wrong

`asof` on the surface reader takes the latest quote timestamp at or before the
requested instant, builds at exactly that instant, and returns whatever that
produces. If the build fails there it stops, and never tries the timestamp
before it, while `timestamps` and `between` keep reporting the full quote grid.

With a `vol_surface` clock this fails late and confusingly. The backtest runs the
whole grid and fills trades, then the orchestrator asks for the window-end
surface, gets empty, and reports no clock ticks in the window.

### Root cause

A third kind of absence, distinct from the two in findings 2 and 3. The chain is
present, every input resolves, and the build still yields nothing because no
expiry survived. The input exists and the output does not.

The reader resolves the *input's* grid and assumes the derived record exists
wherever the input does. For a derived kind that assumption is wrong, and it is
wrong exactly when derivation fails.

Read strictly, `asof` promises the records at the largest visible timestamp at or
before the one requested. For a derived kind that must mean the largest timestamp
where a **surface** exists, not where a chain exists. Walking back is the
contract, not a leniency.

### How far the walk actually goes

`build_surface` yields nothing only when every expiry in the chain is dropped,
and an expiry is dropped when it has already expired or when no strike yields an
invertible implied vol. So the whole build fails only if the entire chain is
expired or the entire chain is unusable at that instant.

The benign cases are one tick wide:

- a chain of same-day contracts evaluated at the expiry instant, where time to
  expiry hits zero and every slice drops;
- a minute of bad data, where marks are missing or priced outside the range
  implied vol can invert;
- a missing spot at that timestamp, which fails earlier in the reader but breaks
  the walk the same way.

Many consecutive failures require every expiry to be unusable across many
minutes. That is not a data quirk. It is a truncated dataset, a synthetic
fixture, or a broken feed.

So the bound is not a performance knob. It is a statement about how much silent
staleness is acceptable before the run would rather be told.

### Decision

- **Bound the walk by tick count, not by elapsed time.** A tick count means "at
  most this many observations back", which behaves sensibly on any grid. A time
  window does not: five minutes is a few ticks on minute data and zero ticks on
  daily data, so the same setting silently disables the walk on a coarser grid.
  Default small, around three, since every benign case is one step.
- **Exhausting the bound throws.** Failing to find a surface within a handful of
  ticks is not absence, it is a broken feed or a truncated dataset. Returning
  empty would send a configuration-grade problem back through the same channel as
  ordinary missing data, which is the mistake findings 2 and 3 exist to correct.
- **The bound lives on the spec.** It changes which surface a policy sees, so it
  changes results, so it belongs in the identity hash and the config surface.

This combination disposes of the visibility question without extra machinery.
Silent staleness is capped at a couple of ticks, and anything worse is loud. A
counter of skipped chains on the reader is a reasonable later addition, with
precedent in `PnLSeries.n_unmarked`, but the throw covers the case that matters.

### Ripple

- `docs/modules/surfaces.md`.
- `docs/modules/market_data.md`: the `asof` wording for derived kinds.
- `docs/modules/experiment.md`: identity gains a field.

### Related, fix while in this code

The reader's `timestamps` returns the quote grid, so it reports instants where no
surface exists. Making it exact would mean building every surface in the range,
which is not acceptable. It stays an over-estimate, and that should be written
down as a property of derived kinds rather than left to be rediscovered the way
this finding was.

### Test gap

The red test compares the walked-back result against a surface it builds at the
earlier timestamp. That reference is non-empty today, but if it ever stopped
building the comparison would be empty against empty and would pass vacuously.
Assert the reference is non-empty first.

---

## 5. Spot reads do not de-duplicate by timestamp

### What is wrong

The spot reader gathers candidate partitions, concatenates them, sorts by
timestamp, and never checks for duplicates. Spots are a snapshot kind, read
everywhere through `only_or_missing`, which throws on two records. So a single
repeated row aborts the fill, the window-end lookup, settlement, and the surface
reader.

Two ways to get there without anyone writing bad code: a vendor re-delivers a
minute so one partition holds it twice, or the same after-midnight row lands in
both the spill of one partition and the body of the next, which the documented
partition convention explicitly permits.

The previous data layer took the first match and moved on, so this is a
behaviour change introduced by the rewrite.

The same reader's `timestamps` has the matching gap. It appends across partitions
and sorts without a uniqueness pass, while `InMemory`'s version applies one, so
the tick grid can contain the same instant twice.

### Decision

De-duplication alone is not the right response, because two rows at one instant
are not all the same thing.

- **Exact duplicates collapse silently.** Same timestamp, same price: there is no
  information to lose.
- **Conflicting prices throw**, naming the timestamp and both values. A data root
  that disagrees with itself about a price is worth stopping for. Taking the
  first is a silent choice between two answers, and the run continues on a number
  nobody verified. That is the failure mode findings 2, 3 and 4 exist to correct;
  the old layer's behaviour was not right merely because it was quieter.

The reader already holds both timestamps and prices when it loads a block, so the
comparison costs nothing.

The rejected alternative was to match master by keeping the first occurrence and
documenting that conflicts go undetected. Less code, never surprises anyone
mid-run, and silently wrong.

### Fix

- Apply the collapse-or-throw rule in `between`, after the sort, so it covers
  both the repeated row and the cross-partition overlap. The default `at`
  inherits it.
- Apply a uniqueness pass to `timestamps` on the same reader.

### Ripple

- `docs/modules/market_data.md`: the partition overlap convention and the
  de-duplication rule.

### Deferred

Bars are left alone deliberately. A chain has many rows per timestamp by design,
so the de-duplication key is the contract rather than the instant, and the
conflict rule needs thinking through for six fields instead of one. Record it as
a separate question rather than fixing it here.

---

## 6. `Constant` `asof` ignores its own visibility stamp

### What is wrong

Curve records carry a timestamp meaning "as of", the moment the curve became
known. `Constant` honours it in two of its three shapes: `between` returns the
record only when its stamp falls inside the range, and `timestamps` follows
`between`. But `asof` returns the record whenever the selector matches,
regardless of the instant requested.

So a curve stamped in June is visible to a January query. That goes straight
through `TimeCut`, which the module describes as the complete no-lookahead rule.
The one mechanism the design relies on to prevent seeing the future has a hole in
it for this provider.

It also shows up in identity. `to_dict(::Constant)` hashes a non-`typemin` stamp,
so two experiments differing only in when a curve became known produce different
hashes and identical results. The identity layer treats the stamp as meaningful
while the read path treats it as decoration.

### Root cause and the rule question

Documented behaviour, not an oversight: the docstring says the record is visible
from the start of time. Design rule 3 applies.

The argument for changing it is that the type already contradicts itself. Three
shapes honour the stamp, one ignores it, and identity votes with the majority.
That is not a position anyone chose, it is a gap.

### Decision

**`asof` returns the record only when the selector matches and the stamp is at or
before the requested instant.**

Safe for everything reachable from a config file today. Both two-argument curve
constructors stamp `typemin(DateTime)`, so the guard is always satisfied and
nothing built from TOML changes behaviour. Only explicitly stamped records move,
which is the intent.

Combined with the decision on findings 2 and 3, `Constant` ends up expressing all
three states cleanly and without overlap:

| Query | Result |
|---|---|
| selector it does not serve | throws (structural) |
| its selector, before the stamp | empty (temporal) |
| its selector, at or after the stamp | the record |

The rejected alternative was to drop the stamp from identity and leave the read
path alone. It keeps the look-ahead and leaves the four shapes permanently
disagreeing, trading a correctness problem for a smaller diff.

### Ripple

- The `Constant` docstring: visible from its timestamp, which defaults to the
  start of time.
- `docs/modules/market_data.md` if it repeats the wording.

---

## Implementation order

1. **6** — `Constant` honours its stamp. Local, and nothing built from TOML
   changes behaviour.
2. **5** — spot collapse-or-throw. Local, one reader.
3. **2 and 3** — structural absence throws. Touches every provider and the
   protocol doc; do it as one change, since a partial rollout leaves the
   providers disagreeing with each other.
4. **4** — bounded surface walk-back. Depends on 2 and 3 for the vocabulary that
   decides what exhausting the bound means.
5. **1** — settlement follows the trade. Largest, and touches metrics,
   experiment, policies and the engine.

Findings 2, 3, 4, 5 and 6 all move absence and conflict from silence to a named
error. Doing them before 1 means the settlement work lands on a data layer that
already fails loudly.

## Test gaps to close first

`test/regressions/test_review_findings.jl` is a starting point, not a complete
specification. Three known holes:

- **Finding 1 covers one branch of two.** Its expiry lands exactly on the window
  end, so only the in-window settlement path runs. Fixing the callback alone
  would turn it green while the past-the-window branch still marks foreign lots
  at the clock underlying's spot. Add a case with an expiry after the window end.
- **Finding 2 asserts too loosely.** It expects any `ErrorException` and its
  config points at nonexistent parquet roots, so eager root validation would make
  it pass for the wrong reason. Match the message.
- **Finding 4 can pass vacuously.** It compares against a surface built at the
  earlier timestamp; if that reference ever stopped building, the comparison
  would be empty against empty. Assert the reference is non-empty first.

Several decisions here also need tests that do not exist yet, notably the
conflict throw in finding 5, the bound-exhausted throw in finding 4, and the
structural-absence throw across every provider in findings 2 and 3.
