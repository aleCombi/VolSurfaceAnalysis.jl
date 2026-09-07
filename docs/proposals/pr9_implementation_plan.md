# PR #9 implementation plan

Status: **implemented.** All ten commits landed on `data-kinds`; the gate run
is recorded in [status.md](../status.md). Companion to
[pr9_correctness_fixes.md](pr9_correctness_fixes.md)
(the six decisions) and [pr9_remaining_findings.md](pr9_remaining_findings.md)
(the lower-confidence findings, triaged below).
This file turns the six correctness decisions into a commit sequence, triages the
four lower-confidence findings, and records the design choices the decisions left
open. Nothing here changes a decision in the correctness document; where a
decision could not be implemented as written it is flagged rather than worked
around.

---

## Commit sequence

| # | Subject | Finding | Src | Docs | Red→green | Revertible alone |
|---|---|---|---|---|---|---|
| 1 | `Record the PR #9 review: proposals, implementation plan, red specs (step 0)` | — | — | proposals, status | — (specs tightened, still red) | yes |
| 2 | `Docs: provider builder table, chain access, core_hash wording (finding D)` | D | identity docstring | experiment, surfaces | — | yes |
| 3 | `market_data: Constant asof honours its visibility stamp (finding 6)` | 6 | providers | market_data | testset 6 | yes |
| 4 | `market_data: spot reads collapse duplicate instants, throw on conflict (finding 5)` | 5 | protocol, parquet | market_data | testset 5 | yes |
| 5 | `market_data: partition convention made time-ordered; sub-second range bounds (findings A, C)` | A, C | parquet | market_data | — (new tests) | yes |
| 6 | `market_data: structural absence is an error, not an empty result (findings 2 and 3)` | 2+3 | protocol, providers, map, time_cut, by_selector, parquet, surface_from, config | market_data, experiment, surfaces | testsets 2, 3 | yes |
| 7 | `surfaces: bounded asof walk-back on SurfaceFrom (finding 4)` | 4 | protocol, surface_from, config, identity | surfaces, market_data, experiment | testset 4 | yes |
| 8 | `metrics/experiment: settlement follows the trade (finding 1)` | 1 | pnl_series, trade, policy, daily_short_strangle, agent, experiment, config | metrics, experiment, policies, agents, backtest, status | testset 1 (both branches) | yes |
| 9 | `Cleanup: unused import, unreachable cache knobs documented (review cleanup)` | cleanup | store, lifecycle | market_data, persistence | — | yes |
| 10 | `Regression suite into the gate; status and proposal closure` | — | — | status, proposals | whole file joins the gate | yes |

Ordering follows the correctness document's proposed order (6, 5, 2+3, 4, 1) with
three additions and **no** reordering of the five fixes. The additions are: a
step-0 commit that lands the untracked proposals and the tightened red specs
(they are currently uncommitted working-tree state); a docs-drift commit early so
later doc edits build on correct text; and the A/C parquet work placed
immediately after finding 5, because the remaining-findings document explicitly
asks for A to be sequenced with decision 5 and both touch the same twenty lines
of `src/market_data/parquet.jl`.

No dependency was found that breaks the proposed order. Two soft couplings are
worth knowing:

- Commits 6 and 8 both edit the "missing spot" expectations in
  `test/backtest/test_engine.jl` and `test/experiment/test_experiment.jl`. Commit
  6 changes the *error type* those tests expect; commit 8 changes the *settlement
  path* they exercise. Doing 6 first (as the document proposes) means commit 8
  lands on tests that already name the new error.
- Commit 7 depends on commit 6 only for vocabulary (what "exhausting the bound"
  means, and where the named-error section of `protocol.jl` lives). It is
  technically implementable before 6.

---

## Contradictions and things the code does not match

Flagged rather than silently resolved.

1. **The red test for finding 3 encodes the rejected alternative.** The testset
   `"BySelector returns empty for unrouted selector"`
   (`test/regressions/test_review_findings.jl:107-125`) asserts that all four
   shapes return their typed empty result for an unrouted selector. The decision
   for findings 2 and 3 is the opposite: `BySelector` was the one provider
   getting this right, and every other provider must join it in throwing. The
   test as written can only go green by implementing the rejected option. It is
   rewritten in commit 1. This is a fourth test gap; the correctness document
   lists three.
2. **Section 3's prose and section 3's decision disagree.** "What is wrong" says
   the `KeyError` "contradicts the protocol rule that empty means absent for all
   four shapes" — which reads as "make it return empty". The Decision and the
   summary table say the rule changes instead. Per the document's own
   instruction ("where a section's wording and this table disagree, the table is
   the intent"), the table governs; recorded here so a future reader does not
   re-derive the prose reading.
3. **A cleanup item does not match the code.** The remaining-findings document
   says `open_data(::MarketData)` "never forwards its `max_days`, `max_chains` or
   `max_surfaces` keyword arguments". The method is
   `open_data(m::MarketData) = MarketData(_open_all(m.entries...))` — it accepts
   no keyword arguments at all, so there is nothing to forward. The observation
   (cache bounds cannot be set from a run) is correct; the mechanism described is
   not. Commit 9 documents the fact rather than "fixing the forwarding".
4. **Finding 1's ripple list misses one doc.** The stale parity claim lives in
   two places: the comment above `_build_settle`
   (`src/experiment/experiment.jl:120-122`) and
   `docs/modules/backtest.md:110-113` ("that simplification is deliberate and
   shared with `run_experiment`"). Commit 8 updates both.
5. **Finding 1's enforcement must stay at load time, or its own red test can
   never pass.** The red testset builds an `Experiment` directly with a QQQ trade
   under a SPY clock and expects PnL `19.0`. If the clock/trade match were
   asserted inside `run_experiment` instead of `load_experiment`, that test would
   throw instead of settling. The decision already says `load_experiment`; noted
   because it is easy to "strengthen" in the wrong place.
6. **Finding 2's red test config points at `/nonexistent/...` roots on purpose.**
   The load-time check added in commit 6 must not touch the filesystem, or the
   test passes for the wrong reason. This is why the parquet specs answer
   "cannot say" rather than `false` before they are opened (see Q1 below).

---

## Design decisions this plan settles

### Q1. The "does this provider serve this selector" shape

One new protocol shape, in the module's existing two-arity style:

```julia
serves(m, ::Type{R}, sel)        -> Union{Bool,Missing}   # map level
serves(p, ctx, ::Type{R}, sel)   -> Union{Bool,Missing}   # provider level
```

No timestamp argument: the question is static by definition. The default is
permissive:

```julia
serves(::Any, ::Any, ::Type, ::Any) = missing
```

**Yes, a sentinel is required, and it is `missing`.** Two provider classes cannot
answer:

- **Parquet specs before they are opened.** The reader can answer (the partition
  list is a `readdir` walk, already cached per selector), but the *spec* cannot,
  and the spec is what `build_market_data` holds. The red test for finding 2
  deliberately configures nonexistent roots, so a spec that answered `false` by
  probing the disk would make that test pass for the wrong reason.
- **Derived providers**, by decision: "derived providers do not answer. They
  delegate, and let their input's error propagate, so the failure names the real
  cause." Returning `missing` is exactly that: the map-level check waves the
  derived read through, the derived provider reads its input through the map,
  and the input's own check throws naming the input kind and selector. A
  `SurfaceFrom` asked for SPX reports `OptionBar/SPX unserved`, not "no surface".

Per provider:

| provider | returns | how |
|---|---|---|
| `InMemory{R}` | `Bool` | a `Set` of `selector(r)` built at construction (new third field; O(1), the shapes are on the hot path) |
| `Constant{R}` | `Bool` | `selector(c.record) == sel` |
| `BySelector{R}` | `Bool`/`missing` | route on `sel`; no route implies `false`; a route implies delegate to the part, so a route to a parquet spec yields `missing` |
| `ParquetOptionBars`, `ParquetSpots` (specs) | `missing` | cannot answer without the tree |
| `ParquetBarsReader`, `ParquetSpotsReader` | `Bool` | `!isempty(_partitions(r, u))` — an empty `date=` list means the tree holds nothing for that symbol |
| `QuotesFromBars` | `missing` | explicit method with a comment, not the default |
| `SurfaceFrom`, `SurfaceReader` | `missing` | same |
| `TimeCut` | delegates | `serves(c::TimeCut, R, sel) = serves(entry(c, R), c, R, sel)` |

`Union{Bool,Missing}` is chosen partly because Julia's three-valued `&` already
does the right thing (`missing & false === false`), so any future provider that
wants to delegate conjunctively over several inputs gets Kleene semantics for
free.

**Where the throw happens.** In the four map-level shapes on `MarketData` and the
four on `TimeCut`, through one shared helper:

```julia
_require_served(m, ::Type{R}, sel) = serves(m, R, sel) === false &&
    throw(UnservedSelector(R, sel, served_description(entry(m, R))))
```

Not inside each provider's four shapes: that would be sixteen call sites and
would also fire on the internal provider-level delegation `BySelector` and
`QuotesFromBars` already do. Consequences to accept and document:

- Provider-level calls (`at(p, ctx, R, sel, ts)`) are unchecked. Tests and
  internal delegation use that arity; this is deliberate.
- On `TimeCut`, the structural check runs *before* the cutoff mask, so an
  unserved selector throws even for a query past the cutoff. Structural beats
  temporal.
- One extra `Set`/`Dict` lookup per map-level read. Measurable with
  `scripts/bench_point_vs_range.jl` if it ever matters; not expected to.
- A provider that implements no `serves` method silently opts out of the check.
  The stricter alternative (no default, plus a `has_lifecycle`-style
  `hasmethod` gate at load) is rejected for now because it breaks third-party and
  in-test providers such as `_SF_CountingBars` in `test/surfaces/test_surface_from.jl`.

**The load-time fast path needs a second, smaller trait.** `serves` answers about
a selector the caller already has; `build_market_data` has no query selector.
What it can check is the selectors a derived spec demands *statically*:

```julia
demands(::Any) = ()                                          # providers.jl, next to inputs
demands(s::SurfaceFrom) = ((RateCurve, s.currency),
                           ((SpotPrice, v) for v in values(s.spot_for))...)
```

`build_market_data` then errors when `serves(m, K, sel) === false` for any
`(K, sel)` in any spec's `demands`. `missing` is skipped — that is what keeps the
check filesystem-free. This is the mechanism that turns the finding-2 red test
green.

### Q2. How much of the suite `InMemory`-throws breaks

Fifteen assertions across six files, in ten testsets. Not a pass through the
suite's worth of surprises — the affected sites are enumerable by reading, and
every one of them is a place that was asserting the ambiguity findings 2 and 3
exist to remove.

| file | line(s) | what it asserts today | after commit 6 |
|---|---|---|---|
| `test/market_data/test_providers.jl` | 16 | `at(m, SpotPrice, QQQ, T1) == SpotPrice[]` | `@test_throws UnservedSelector` |
| | 40 | `asof(m, OptionBar, QQQ, T3) == OptionBar[]` | `@test_throws UnservedSelector` |
| | 52 | `asof(m, SpotPrice, SPX, T1) == SpotPrice[]` (`Constant`) | `@test_throws UnservedSelector` |
| | 55 | `between(m, SpotPrice, SPX, ...) == SpotPrice[]` (`Constant`) | `@test_throws UnservedSelector` |
| `test/market_data/test_by_selector.jl` | 39, 40 | `@test_throws KeyError` | `@test_throws UnservedSelector` |
| `test/market_data/test_parquet.jl` | 39 | `at(d, OptionBar, QQQ, t1a) == OptionBar[]` | throws (reader: no partitions) |
| | 103 | `timestamps(d, OptionBar, QQQ, ...) == DateTime[]` | throws |
| `test/surfaces/test_surface_from.jl` | SPX surface | `== VolatilitySurface[]` | throws, naming `OptionBar`/SPX |
| | wrong-currency rate | `== VolatilitySurface[]` | throws, naming `RateCurve`/EUR |
| | wrong-underlying div | `== VolatilitySurface[]` | throws, naming `DivCurve`/SPX |
| | `spot_for` remap without spots | `== VolatilitySurface[]` | throws, naming `SpotPrice`/SPX |
| `test/backtest/test_engine.jl` | 136-138 | clock on QQQ yields no ticks | throws |
| | 144-148 | `@test_throws ErrorException` on `InMemory(SpotPrice[])` | `@test_throws UnservedSelector` |
| `test/experiment/test_experiment.jl` | 204-207 | `@test_throws ErrorException` on `_ex_map(..., SpotPrice[])` | `@test_throws UnservedSelector` |

Two of these are the same trap and deserve calling out: **an empty `InMemory`
serves nothing**, so the two "missing window-end spot / missing fill spot"
regression tests stop exercising the missing-spot path and start exercising the
unserved path. They also fail on the *type*, because `UnservedSelector <:
Exception` and not `<: ErrorException` (Julia's `ErrorException` is concrete).
The right update is to keep both cases: change the empty-`InMemory` sites to
expect `UnservedSelector`, and add a spot fixture that serves the underlying but
has no row at the relevant instant, so the original "missing spot" error stays
covered.

Unaffected and worth noting because they look risky: `test_surface_from`'s "no
spot at ts2" case (SPY is served, one row, absent at `TS2` — temporal, still
empty); `test_clock.jl`'s `Clock{SpotPrice}(SPX)` (SPX is in `_md_spots`);
everything in `test_time_cut.jl`, `test_map.jl`, `test_policy.jl`,
`test_agent.jl`, `test_persistence`, and the viz suites.

### Q3. The widened settle callback

```julia
settle(trd::Trade) -> Union{Float64,Missing}
```

The trade replaces the expiry outright rather than being added beside it: the
expiry is one field away (`trd.expiry`), and two arguments would let a caller
pass a mismatched pair.

Call sites:

1. `src/metrics/pnl_series.jl`, the residual loop: `spot = settle(expiry)` becomes
   `spot = settle(lot.pos.trade)`; the entry is still stamped at
   `lot.pos.trade.expiry`. Both docstrings (the `PnLSeries` struct and
   `pnl_series`) state the new signature.
2. `src/experiment/experiment.jl`, `_build_settle`. New shape:

   ```julia
   function _build_settle(d::MarketData, window_end::DateTime)
       function settle(trd::Trade)::Union{Float64,Missing}
           ts = min(trd.expiry, window_end)
           s  = only_or_missing(at(d, SpotPrice, selector(trd), ts))
           return ismissing(s) ? missing : s.price
       end
   end
   ```

   The two branches collapse into one lookup at `min(expiry, window_end)`, which
   is precisely what the decision asks for ("the past-the-window branch ... must
   become a per-selector lookup at `window_end`"). Note the behaviour change this
   implies: case 1 could previously never return `missing` (it returned a
   precomputed scalar); it now can, for a lot whose underlying is served but has
   no row at `window_end`. For the clock underlying that is impossible —
   `run_experiment` has already errored if the window-end spot is missing — so
   single-underlying experiments, which is what the second half of decision 1
   makes the enforced invariant, are unaffected. `u` and `window_end_spot` drop
   out of `_build_settle`'s parameter list.
3. `run_experiment`: `settle = _build_settle(d, window_end)`. It still resolves
   `spot` at the window end, both for the existing loud error and for provenance.
4. `test/metrics/test_pnl_series.jl:17`, `_const_settle(spot) = (_::DateTime) -> ...`
   — the annotation must become `::Trade` or be dropped. This is the only test
   closure that breaks; the five `settle=_->500.0` sites in
   `test/metrics/test_core.jl` and the `settle=_ -> missing` site still compile,
   and should be left alone or updated for clarity, not correctness.
5. `src/positions/trade.jl` gains `selector(t::Trade) = t.underlying`. It goes
   there, not in `market_data/kinds.jl`, because `Trade` is defined later in the
   include order. Small tension to record in the docs: `kinds.jl` documents
   `selector` as a per-*kind* trait, and `Trade` is not a kind (no `timestamp`,
   never served by a provider). The method is still the right call — it is the
   module's vocabulary for "which parallel series is this about" — but
   `market_data.md` and `positions.md` should both say that one non-kind type
   implements it.

**`PnLSeries.window_end_spot` does not change.** Same field, same type, same
position in the struct, so the persistence layer, `manifest.parquet` and
`_load_manifest` are all untouched. What changes is its docstring and
`docs/modules/metrics.md`: it stops being described as "the case-1 mark passed
into the settle closure" and becomes purely "the spot at the run's reference
(clock) underlying at the window end, recorded for provenance". The metrics layer
already never used it in a computation; after commit 8 the orchestrator does not
either.

### Q4. The lookback bound on `SurfaceFrom`

- **Field:** `lookback_ticks::Int`, on `SurfaceFrom`, default `3`.
- **Meaning:** the number of input timestamps *examined*, not steps taken. `1`
  reproduces today's "try only the newest quote timestamp"; `3` tries the newest
  and two earlier ones. Rejected at construction if `< 1`.
- **Config key:** `lookback_ticks` in `[data.vol_surface]`, optional.

Everything that must change because it enters identity:

| place | change |
|---|---|
| `SurfaceFrom` struct | third field; the kwarg constructor gains `lookback_ticks::Int=3` |
| `Base.==` / `Base.hash` on `SurfaceFrom` | **must** include the new field — they are hand-written today because of the `Dict` field, so a new field is silently dropped otherwise. This is the single easiest thing to forget in commit 7. |
| `to_dict(s::SurfaceFrom)` | `"lookback_ticks" => s.lookback_ticks`, emitted **always**, not omitted-when-default. The identity layer's stated principle is "identity from the resolved experiment, not config bytes"; the omit-when-default trick on `Constant`'s timestamp is a documented exception, not the house style. |
| `_build_surface_from` (`config.jl`) | reads and validates the key |
| `docs/modules/experiment.md` | the `vol_surface` row of the loader table; the identity paragraph |
| `docs/modules/surfaces.md` | the `SurfaceFrom` section: the bound, the throw, and the "`timestamps` is an over-estimate for derived kinds" property the decision asks to write down |
| `docs/modules/market_data.md` | the `asof` rule for derived kinds |
| `test/experiment/test_identity.jl` | new assertion: two `SurfaceFrom`s differing only in `lookback_ticks` have different `core_hash` |

**Stored-run compatibility: this does not trip `schema_version`.** Checked
against the code rather than assumed:

- `RUN_SCHEMA_VERSION` (currently 2) versions the *manifest column set*, and
  `load_run` compares it before reading. Nothing about the manifest's shape
  changes, so no bump is needed and old runs at version 2 keep loading.
- `load_run` does **not** recompute `full_hash`; it reads `config.toml` verbatim
  and rebuilds. So every already-saved run still loads, still reports its stored
  `core_hash`, and is internally consistent.
- `save_run` *does* recompute `full_hash(config_exp)` and compare it to
  `full_hash(result.experiment)` — both computed by the same new code, so they
  agree.
- The real consequence is narrower and should be stated in the commit message:
  **rerunning an existing config that has a `[data.vol_surface]` table produces a
  new `run_id`**, so it lands in a new `runs/run_id=.../` directory beside the
  old one, and a `core_hash` comparison against a pre-commit-7 run will not
  match. That affects the gitignored `scripts/runs/` store and the backlog
  "reproducibility harness", not the tested surface. No test pins a hash literal
  (`test_identity.jl:43` and `test_store.jl:86` only check hex shape).
- Bumping `RUN_SCHEMA_VERSION` to force old runs to be regenerated would be
  *wrong* here: their manifests are still readable and their stored results are
  still what that config produced under that code.

Config strictness: `[data.<kind>]` builders silently drop unknown keys (a
cleanup item). A typo'd `lookback_ticks` would then silently take the default and
silently change identity-vs-intent. Commit 7 therefore validates the key set of
`[data.vol_surface]` specifically; the general case stays a cleanup item.

### Q5. The error type for structural absence

**One type, for structural absence.**

```julia
struct UnservedSelector <: Exception
    kind     :: Type
    selector :: Any
    served   :: String     # what the entry does serve, for the message
end
```

`served` is filled by the provider that answered `false`: `Constant` gives its
one selector, `InMemory` a sorted (truncated) list, `BySelector` its route
selectors, a parquet reader `"no date= partitions for symbol=QQQ under <root>"`.
A bare key error naming only the selector does not say enough to fix a config,
which is the whole point of the decision.

**Where:** `src/market_data/protocol.jl`, in a new "Errors" section at the bottom,
with a `Base.showerror` method. That file is included at position 2 of the
module's include list, so `map.jl`, `by_selector.jl`, `parquet.jl`,
`surfaces/surface_from.jl` and `experiment/config.jl` all see it. Everything is
one flat module (`VolSurfaceAnalysis`), so "both `market_data` and `surfaces` can
throw it" needs nothing but include order. Exported, because tests and users
catch it by name.

**Not one type for everything.** The governing stance is "every other unanswerable
question gets a name", and the summary table names three distinct states. Each
gets its own type, each added by the commit that first throws it, each in the
same section of `protocol.jl` (disjoint regions, so the commits stay
independently revertible):

| state | type | added in | carries |
|---|---|---|---|
| nothing serves this selector | `UnservedSelector` | commit 6 | kind, selector, what is served |
| two rows, two answers | `ConflictingRecords` | commit 4 | kind, selector, timestamp, the two values |
| input present, derivation failed past the bound | `DerivationExhausted` | commit 7 | kind, selector, requested instant, oldest instant tried, the bound |

The cheaper alternative — `ArgumentError` with a good message for the latter two —
is viable and would shrink commits 4 and 7 slightly. It is rejected because
`@test_throws ArgumentError` cannot distinguish "conflicting spot rows" from
"reader is closed", and both are thrown by the same reader.

### Q6. What would settle finding B (quotes re-synthesized per call)

The claim is that `at(::QuotesFromBars, ...)` rebuilds a full `OptionQuote` chain
several times per tick where the old layer built it once. The reasoning is
correct; the *impact* is not established, and one repo fact cuts against it:
`DailyShortStrangle` implements `tick_times`, so the only real config
(`configs/strangle_spy_16d_1dte.toml`) calls `decide` roughly once per day, not
once per minute. The per-tick multiplier applies to a policy that does not narrow
the grid, which today does not exist.

Measurement that would settle it, modelled on `scripts/bench_point_vs_range.jl`
(prior art in this repo: it already opens a real `ParquetOptionBars` over
`~/data/massive`, enumerates a month of minute timestamps, warms up to keep
compilation out of the numbers, and reports wall time, `@allocated`, and the
`Sys.maxrss()` delta):

1. A `scripts/bench_quote_synthesis.jl` in the same shape: open
   `MarketData(ParquetOptionBars, QuotesFromBars)` over one month of SPY minute
   data and time (a) `at(m, OptionQuote, u, t)` at every timestamp, (b) the same
   with three calls per timestamp (surface reader + `decide` + one order), and
   (c) a variant where `QuotesFromBars` opens into a reader with an
   `LRU{Tuple{Underlying,DateTime},Vector{OptionQuote}}`. Report the same three
   numbers plus the record count, as section 10 of the data-kinds proposal does.
2. A whole-run comparison: `run_experiment` on the strangle config over one
   month, before and after, since that is the only end-to-end number anyone acts
   on.

Threshold for acting: the caching variant has to move the whole-run number by
something worth a lifecycle change. `QuotesFromBars` is currently its own reader
(`open_data(::Union{InMemory,Constant,QuotesFromBars}) = s`); giving it a reader
adds a spec/reader pair, a `serves` method on the reader, and a cache whose
cut-independence argument has to be made the way `SurfaceReader`'s was.
Recommendation: **defer** until those numbers exist; the benchmark itself is
cheap and can be run whenever.

---

## The commits

### 1. `Record the PR #9 review: proposals, implementation plan, red specs (step 0)`

Lands what is currently uncommitted working-tree state, plus this file, plus the
four test-gap fixes.

**Files:** `docs/proposals/pr9_correctness_fixes.md` (new),
`docs/proposals/pr9_remaining_findings.md` (new),
`docs/proposals/pr9_implementation_plan.md` (new),
`test/regressions/test_review_findings.jl` (new, tightened),
`test/runtests.jl` (the commented-out include, with its explanation),
`docs/status.md` (an "In flight" entry naming the six findings and this plan).

**Test-gap fixes, all four, in this one commit** rather than folded into the fix
commits. The gaps are independent of the fixes, they are all in one file, and
folding them would mean each fix commit contains both "the spec I am meeting" and
"the change that meets it" — which is exactly what makes a regression commit hard
to review:

1. **Finding 1 covers one branch of two.** Add a second case to the settlement
   testset with an expiry *after* the window end, on the foreign (QQQ)
   underlying, asserting the payoff computed from QQQ's window-end spot. Without
   it, commit 8 could turn the existing case green while leaving the
   past-the-window branch marking foreign lots at the clock underlying's spot.
2. **Finding 2 asserts too loosely.** Replace `@test_throws ErrorException` with a
   capture-and-match on the message: it must mention `rate_curve`/`RateCurve`,
   `EUR`, and `USD`. This is safe to write before commit 6 because the message
   shape is fixed here (Q1/Q5). It also guards against the roots being validated
   eagerly.
3. **Finding 4 can pass vacuously.** Assert `!isempty(prior)` before comparing
   `asof(...) == prior`.
4. **Finding 3 asserts the rejected alternative** (see Contradictions #1).
   Rewrite the testset to expect `UnservedSelector` from all four shapes, with a
   comment pointing at the decision.

Docs per rule 1: none — no public surface changes. Rule 4 is satisfied by the
status.md entry. Revertible: yes, nothing else depends on it.

### 2. `Docs: provider builder table, chain access, core_hash wording (finding D)`

Three drifted descriptions, all confirmed against the code, all covered by rule 1
and none dependent on a fix.

**Files:**
- `docs/modules/experiment.md`, the "new concrete types register themselves"
  paragraph (~line 225): add `_PROVIDER_BUILDERS` to the list beside
  `_CURVE_BUILDERS`, `_SYNTHESIZER_BUILDERS`, `_POLICY_BUILDERS`,
  `_AGENT_BUILDERS`. It is the one registry a new provider spec must appear in,
  and the diff that renamed it in code dropped the old name from the doc without
  adding the new one.
- `docs/modules/surfaces.md`, "Does NOT own": "Raw chain access (that is `data`)"
  becomes `market_data`, matching `docs/modules/data.md`, which now assigns readers to
  `market_data`.
- `src/experiment/identity.jl`, the `core_hash` docstring: "source, agent,
  window" becomes "data, clock, agent, window", matching `_core_dict`.

Revertible: yes.

### 3. `market_data: Constant asof honours its visibility stamp (finding 6)`

**Src:** `src/market_data/providers.jl`.

```julia
asof(c::Constant{R}, ::Any, ::Type{R}, sel, ts::DateTime) where {R} =
    (selector(c.record) == sel && c.record.timestamp <= ts) ? R[c.record] : R[]
```

and the docstring: "one record, visible from its timestamp, which defaults to the
start of time" — replacing "visible from the start of time", which is the
documented behaviour this commit changes.

**Docs (rule 1, and rule 3 is engaged):** `docs/modules/market_data.md`, the
"Provider specs" bullet for `Constant` and the "Curve kinds" paragraph, which
both repeat the "start of time" wording. Because this changes *documented*
behaviour, the commit message must cite the decision in
`pr9_correctness_fixes.md` section 6 — the proposal is the explicit rule-change proposal
rule 3 requires; the implementing commit is the application, not the proposal.

**Tests:** regression testset "constant asof respects record timestamp" goes
green. New coverage in `test/market_data/test_providers.jl`: a stamped `Constant`
is empty before the stamp, present at and after it, and its `between`/`timestamps`
are unchanged. **No existing test breaks** — every `Constant` in the suite is
built with a two-argument constructor (`typemin(DateTime)`), verified across
`test_providers.jl`, `test_by_selector.jl`, `test_surface_from.jl`,
`test_policy.jl`, `test_experiment.jl` and `test_config.jl`.

Revertible: yes.

### 4. `market_data: spot reads collapse duplicate instants, throw on conflict (finding 5)`

**Src:**
- `src/market_data/protocol.jl`: new "Errors" section with `ConflictingRecords`
  and its `showerror`.
- `src/market_data/parquet.jl`: `between(::ParquetSpotsReader, ...)` applies
  collapse-or-throw after the sort, so it covers both the repeated vendor row and
  the cross-partition overlap; the provider-level default `at` inherits it.
  `timestamps(::ParquetSpotsReader, ...)` gains a `unique!` after its sort,
  matching `InMemory`'s.

Rule: equal timestamp **and** equal price collapses silently; equal timestamp and
different price throws, naming the timestamp and both values. The reader holds
both vectors when it loads a block, so the comparison is free.

**Docs:** `docs/modules/market_data.md`, the parquet section — the partition
overlap convention gains the de-duplication rule, and the "Bars are left alone
deliberately" carve-out is recorded (a chain has many rows per timestamp by
design, so its de-duplication key is the contract, not the instant; the six-field
conflict rule is a separate question).

**Tests:** regression testset "spot reads de-duplicate timestamps" goes green.
New in `test/market_data/test_parquet.jl`: exact duplicates within one partition
collapse; the same row present in both a partition body and the previous
partition's spill collapses; conflicting prices throw `ConflictingRecords` with
both values in the message; `timestamps` is unique across the overlap. No
existing test breaks (the shared fixture writes no duplicates).

Revertible: yes.

### 5. `market_data: partition convention made time-ordered; sub-second range bounds (findings A, C)`

Include-now items from the remaining-findings document; see the triage table for
why.

**Finding A — resolve by tightening the convention, not by changing four shapes.**
The convention today says partition `D` may hold any timestamp in
`[D 00:00, D+1 02:00)`. That permits a D-1 partition to hold a row *later* than a
row in D, which is what makes `asof` (first partition walking back) disagree with
`at`/`timestamps` (merge both candidates), and what makes the lazy `PartitionBars`
iterator emit out-of-order records and `by_timestamp` throw. Tighten it instead:
**every row in partition D-1 precedes every row in partition D.** That is what
the collector actually produces — one contiguous local-day session per partition,
with the after-midnight UTC spill belonging to the *earlier* session — and under
it both disagreements vanish by construction: the latest row at or before `ts` is always in
the newest candidate partition that has one, and concatenating D-1 then D is
already sorted.

**Src:** `src/market_data/parquet.jl` header comment. **Docs:**
`docs/modules/market_data.md` "Partition convention". **Tests:** a new
`test/market_data/test_parquet.jl` case writing its own small tree (not the
shared `_md_build_parquet_fixture`, to avoid perturbing existing assertions) that
asserts the ordering property holds for the spill layout and that
`asof == at(last(timestamps(...)))` under it.

The alternative — take the maximum over both candidate partitions in `asof` for
both readers, and merge lazily in `PartitionBars` — is more code, and the lazy
two-way merge is the awkward part. If a future feed genuinely interleaves
partitions, that is the fix; today it would be defending against data no
collector writes. Recorded in the commit message.

**Finding C — one line.** `_ts_sql` formats to whole seconds, so a `between` whose
lower bound carries sub-second precision admits the bar at the floor of that
bound while `at` and `timestamps` compare at full precision. Change the format to
`"yyyy-mm-dd HH:MM:SS.sss"`; DuckDB parses fractional seconds, and the exact
`timestamp = ...` predicate in `at` stays exact. Test: a `between` with a
sub-second lower bound excludes the bar at its floor, matching `timestamps`.

Revertible: yes, independently of commit 4 (disjoint lines).

### 6. `market_data: structural absence is an error, not an empty result (findings 2 and 3)`

The largest data-layer commit, and one commit by decision: a partial rollout
leaves providers disagreeing about what absence means.

**Src:**
- `protocol.jl`: the `serves` generic (both arities) with its three-valued
  contract and the permissive default; `UnservedSelector` + `showerror`; the
  header rule comment updated (empty no longer means absent unconditionally).
- `providers.jl`: `InMemory` gains a selector `Set` field and `serves`; `Constant`
  gains `serves`; `QuotesFromBars` gains an explicit `serves(...) = missing` with
  the delegation rationale; `demands(::Any) = ()` next to `inputs`.
- `map.jl`: `serves(m::MarketData, R, sel)`, `_require_served`, and the check at
  the head of the four map-level shapes.
- `time_cut.jl`: `serves(c::TimeCut, R, sel)` and the same check, before the
  cutoff mask.
- `by_selector.jl`: `serves` routes then delegates; `_route`'s no-match fallback
  throws `UnservedSelector` instead of `KeyError` (it is now unreachable from the
  map level, but reachable provider-level); the docstring's `KeyError` promise is
  replaced.
- `parquet.jl`: `serves` returning `missing` on the two specs and `Bool` on the
  two readers, plus the `served` description strings.
- `surfaces/surface_from.jl`: explicit `serves(...) = missing` on spec and reader;
  `demands(s::SurfaceFrom)`.
- `experiment/config.jl`: `build_market_data` gains the `demands`/`serves` load
  check (skipping `missing`, so no filesystem access), and the dead duplicate
  `has_lifecycle` check inside the input-kinds loop is deleted — it is
  unreachable in any case the following loop does not already cover, and it sits
  three lines from the new code.

**Docs (rule 1):**
- `docs/modules/market_data.md`: the protocol rules ("**Empty means absent** for
  all four shapes" becomes the structural/temporal distinction with the
  three-state table from the decision); a new `serves` row in the shapes table; the
  `BySelector` paragraph (`KeyError` becomes `UnservedSelector`); the `InMemory` and
  `Constant` spec bullets; the parquet reader bullets.
- `docs/modules/experiment.md`: the load-time check list gains the
  selector-demand check, with the note that it is a fast path over closed-world
  providers and not the mechanism.
- `docs/modules/surfaces.md`: the `SurfaceFrom` "**Failure is empty**" bullet must
  be qualified — an absent chain, spot or curve is now either an error (nothing
  serves that selector) or empty (served, nothing at this instant).

**Tests:** regression testsets 2 and 3 (as rewritten in commit 1) go green;
fifteen assertions across six files updated per Q2; new coverage that each raw
provider answers `serves` correctly and that a derived provider's failure names
the *input* kind and selector, not the derived one.

Revertible: yes, though it is the commit most likely to conflict on revert once 7
and 8 land (both add methods near it).

### 7. `surfaces: bounded asof walk-back on SurfaceFrom (finding 4)`

**Src:**
- `surfaces/surface_from.jl`: the `lookback_ticks` field, constructor,
  `==`/`hash`, docstring, and the walk:

  ```julia
  function asof(r::SurfaceReader, m, ::Type{VolatilitySurface}, u::Underlying, ts::DateTime)
      cursor = ts
      for _ in 1:r.spec.lookback_ticks
          q = asof(m, OptionQuote, u, cursor)
          isempty(q) && return VolatilitySurface[]          # temporal: nothing visible
          win = first(q).timestamp
          s = at(r, m, VolatilitySurface, u, win)
          isempty(s) || return s
          cursor = win - Millisecond(1)
      end
      throw(DerivationExhausted(VolatilitySurface, u, ts, cursor, r.spec.lookback_ticks))
  end
  ```

  Three distinct outcomes, matching the decision's three states: no chain at all
  gives empty (temporal); a chain that builds gives the surface; chains that never build
  within the bound throw. Each step is one `asof` on the input, and the empty
  results are cached by `at`, so a repeated walk is cheap.
- `protocol.jl`: `DerivationExhausted` + `showerror`; the `asof` docstring gains
  the derived-kind clause ("the largest timestamp at which a record of `R`
  *exists*, which for a derived kind is not necessarily where its input exists").
- `experiment/config.jl`: `_build_surface_from` reads and validates
  `lookback_ticks`, and validates the table's key set.
- `experiment/identity.jl`: `to_dict(::SurfaceFrom)` gains the field.

**Docs:** `surfaces.md` (the bound, the throw, and the "`timestamps` and `between`
over-estimate for derived kinds" property the decision asks to write down rather
than leave to be rediscovered), `market_data.md` (`asof` for derived kinds),
`experiment.md` (loader table row, identity paragraph).

**Tests:** regression testset 4 goes green (with the non-empty reference
assertion from commit 1). New: exhausting the bound throws
`DerivationExhausted`; `lookback_ticks = 1` still throws rather than reproducing
the old empty result (call this out — the old behaviour is deliberately not
reachable); the config key round-trips; two specs differing only in
`lookback_ticks` differ in `core_hash` and in `==`/`hash`. No existing test
breaks: `test_surface_from`'s only `asof`-returns-empty assertion is
`asof(..., _SF_TS1 - Minute(1))`, which has no visible chain at all.

Revertible: yes. Note the run-id consequence from Q4 in the commit message.

### 8. `metrics/experiment: settlement follows the trade (finding 1)`

**Src:**
- `metrics/pnl_series.jl`: `settle(trd::Trade)`, the residual-loop call site, both
  docstrings, and the `window_end_spot` field docstring (provenance only).
- `positions/trade.jl`: `selector(t::Trade) = t.underlying`.
- `experiment/experiment.jl`: `_build_settle(d, window_end)` per Q3; the stale
  parity comment above it deleted; `run_experiment`'s docstring updated (each
  residual lot settles at its own trade's underlying).
- `policies/policy.jl`: `declared_underlyings(::Policy) = ()` with its docstring
  ("the underlyings a policy declares statically; empty when it declares none,
  which means it cannot be checked at load").
- `policies/daily_short_strangle.jl`: `declared_underlyings(p) = (p.underlying,)`.
- `agents/agent.jl`: `declared_underlyings(::Agent) = ()`;
  `declared_underlyings(a::StaticAgent) = declared_underlyings(a.policy)`.
- `experiment/config.jl`: `_experiment_from_cfg` errors when
  `declared_underlyings(agent)` is non-empty and does not contain the clock
  selector, naming both.
- `VolSurfaceAnalysis.jl`: export `declared_underlyings`.

Explicitly **not** in this commit: the engine-side fill-time comparison of
`trd.underlying` against the clock selector. The decision calls it a follow-up.

**Docs (rule 1 — four module docs plus status):**
- `metrics.md`: the mermaid node (`settle DateTime to ...` becomes `Trade`), the
  `pnl_series` contract paragraph, the `window_end_spot` paragraph, and the
  "Per-leg settle closure" key-decision row.
- `experiment.md`: the "Per-leg settlement" key-decision row (it currently says
  "via the clock underlying's spot"), the load-time check list, and the failure
  modes table (a lot on a served underlying with no spot at `min(expiry,
  window_end)` counts in `n_unmarked`; on an unserved underlying it throws
  `UnservedSelector`).
- `policies.md`: the new trait, in "The abstraction" and "Owns".
- `agents.md`: the agent-level delegation.
- `backtest.md`: the "shared with `run_experiment`" claim (Contradictions #4).
- `status.md`: the step 5/6 paragraph describing `settle(expiry)`.

**Tests:** regression testset 1 goes green on both branches. Existing:
`test/metrics/test_pnl_series.jl:17` (`_const_settle`'s `::DateTime`
annotation) is the only breakage; the other seven `settle=` sites still compile.
New: `declared_underlyings` for `NoOpPolicy`, `DailyShortStrangle`, `StaticAgent`;
`load_experiment_str` rejects a config whose policy underlying differs from the
clock selector (and accepts the matching case, which every config in `configs/`
is); a past-the-window foreign lot settling at its own underlying's window-end
spot.

Revertible: yes, and it is the commit most worth being able to revert alone,
which is why the engine-side check is deliberately left out.

### 9. `Cleanup: unused import, unreachable cache knobs documented (review cleanup)`

Optional; drop it if the branch is long enough. Contents: remove `using SHA` from
`src/persistence/store.jl`; document in `docs/modules/market_data.md` that cache
bounds are `open_data` kwargs on individual specs and are **not** settable from a
run through `open_data(::MarketData)` (which takes no kwargs — see Contradictions
#3), so `max_days_cached` in a `[data.<kind>]` table is silently ignored. The dead
duplicate lifecycle check is already removed in commit 6. The two duplicated-SQL
items and the duplicated reader scaffolding are deferred (see triage).

### 10. `Regression suite into the gate; status and proposal closure`

Uncomment the `include("regressions/test_review_findings.jl")` in
`test/runtests.jl` (all six testsets are green by now), record the gate run in
`docs/status.md` the way `d636c7e` did for step 3.2, mark both proposal documents
implemented with a pointer to this plan, and move the PR-9 entry out of status
"In flight".

The regression file stays where it is rather than being dissolved into the
topical suites: the "why" is worth keeping co-located, and the file is small. The
alternative — fold each spec into `test_providers.jl` / `test_parquet.jl` /
`test_surface_from.jl` / `test_experiment.jl` as it goes green and delete
`test/regressions/` — is defensible and would need no new gate entry.

---

## Remaining findings: triage

| item | recommendation | reason |
|---|---|---|
| **A. Partition overlap inconsistent across shapes** | **include now** (commit 5), as a convention tightening plus a test, not a code change | The remaining-findings doc asks for it to be sequenced with decision 5, and it is the same twenty lines. Tightening the convention to "every row in D-1 precedes every row in D" is what the collector already produces and makes all four shapes agree by construction, where the code fix would need a lazy two-way merge for bars `between`. |
| **B. Quotes re-synthesized per call** | **defer**, pending measurement | The only real config narrows the grid with `tick_times`, so the per-tick multiplier does not apply to it. The fix is a lifecycle change (`QuotesFromBars` gains a reader and a cache whose cut-independence has to be argued). See Q6 for the exact benchmark; `scripts/bench_point_vs_range.jl` is the template. |
| **C. Sub-second range bound truncated** | **include now** (commit 5) | One-line format change plus one test; removes a disagreement between three shapes on the same bound; zero risk. |
| **D. Documentation drift (3 items)** | **include now** (commit 2), early | All three are rule-1 debts, all verified against the code, none depends on a fix, and two of the three files are edited again later — better to correct them before building on them. |
| Dead duplicate lifecycle check | **include now**, folded into commit 6 | Same function, three lines away from the new load check. |
| Duplicated SQL helpers (timestamp formatter, path quoter) | **defer** | Extraction crosses module boundaries (`market_data` and `persistence`, `market_data` and `data`/polygon) and would need a home nobody has chosen. Commit 5 edits `_ts_sql`, which makes the duplication more visible, not less; record it as its own small refactor. |
| Duplicated reader scaffolding (`ParquetBarsReader` / `ParquetSpotsReader`) | **defer, and do it after this plan, not during** | It is a genuine refactor of code that commits 4, 5 and 6 all touch. Doing it first would rebase all three; doing it inside one of them would hide the correctness change. |
| Unreachable cache knobs | **include the doc half now** (commit 9), defer the code | Stating "cache bounds are not settable from a run" is honest and free. Making `[data.<kind>]` tables reject unknown keys is a behaviour change that could fail existing local configs; commit 7 does it for `[data.vol_surface]` only, where a typo would silently change identity. |
| Refutation: "`InMemory` can hold duplicate rows for one selector and instant" | **defer, and record the reopening** | The remaining-findings caveat is right: once commit 4 makes conflicting spot rows an error in the parquet reader, the fixture provider disagreeing with it is a wart. But `InMemory` is deliberately permissive (`only_or_missing` enforces downstream) and several fixtures rely on that. Revisit as its own question after commit 4 lands. |

---

## Running this on the constrained machine

Do not run `Pkg.test()` per commit — 2 cores and 3.7 GB, and there is a warm REPL
holding state. Per commit, run only what that commit touches, from a *fresh*
`julia --project=.` process, never the warm pane:

- the regression file directly:
  `julia --project=. test/regressions/test_review_findings.jl` (nested testsets
  catch and continue, so the still-red ones do not abort the run);
- the one or two topical suites the commit edits, by including
  `test/market_data/fixtures.jl` first (the suites assume `runtests.jl`'s single
  module and its `using` lines).

The full gate runs once, in commit 10, and its result is what gets recorded in
`status.md`.
