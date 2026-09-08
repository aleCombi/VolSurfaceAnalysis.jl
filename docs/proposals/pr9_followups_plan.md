# PR #9 follow-ups: implementation plan

Status: commit 1 **implemented**; the rest proposed. Turns the two items in
[pr9_followups.md](pr9_followups.md) into work: the spot `asof` gap (small,
decidable now, and now done -- commit 1, marked ✓ below) and finding B from
[pr9_remaining_findings.md](pr9_remaining_findings.md) (quotes re-synthesized
per call — deferred pending measurement, and this plan is mostly about getting
that measurement).

The third follow-up, making `InMemory` reject conflicting rows, is **not** in
scope here: it is decided but blocked on a per-kind "one record per instant"
trait that does not exist, which is its own design question.

---

## Commit sequence

| # | Subject | Item | Src | Docs | Tests |
|---|---|---|---|---|---|
| 1 ✓ | `market_data: spot asof obeys the de-duplication rule (follow-up 2)` | asof gap | parquet | market_data | 3 new cases |
| 2 | `backtest: fetch the chain once per tick, not once per order (finding B, part 1)` | B | engine | backtest | engine suite |
| 3 | `Benchmark quote synthesis; finding B measured (finding B, part 2)` | B | — (script) | followups | — |
| 4 | *conditional* `market_data: QuotesFromBars gains a reader with a bounded chain cache (finding B, part 3)` | B | providers, lifecycle | market_data, experiment | new lifecycle + cache cases |

Commits 1 and 2 are unconditional and independent of each other. Commit 3
produces the number that decides whether commit 4 happens at all; **do not write
commit 4 before commit 3 has run.** That is the whole point of the deferral —
finding B is the one performance claim in the review, its reasoning is sound and
its impact is not established.

---

## 1. Spot `asof` obeys the de-duplication rule — landed

### The gap

Decision 5 applies collapse-or-throw in `between` (with the provider-level
default `at` inheriting it) plus a uniqueness pass on `timestamps`. That is what
landed. `asof(::ParquetSpotsReader, ...)` reached neither: it walked the partition
list backward and, on the first block holding a timestamp `<= ts`, returned
`_append_spots!(SpotPrice[], u, b, win, win)` — a direct read of one block. A
vendor row delivered twice at the winning instant came back as two records, and
`only_or_missing` threw the bare `ArgumentError` that decision 5 exists to
replace. Worse, reading one block means reading one *partition*, so a
conflict across the spill overlap made `asof` return the later partition's price
silently while `at` threw — see the measured symptom in
[pr9_followups.md](pr9_followups.md).

### Decision: route `asof` through `between`, not through a second call to the helper

Two candidates were recorded. Taking the second.

```julia
function asof(r::ParquetSpotsReader, ctx, ::Type{SpotPrice}, u::Underlying, ts::DateTime)
    _assert_open(r)
    parts = _partitions(r, u)
    for j in searchsortedlast(parts, Date(ts)):-1:1
        b = _block(r, u, parts[j])
        k = searchsortedlast(b.timestamps, ts)
        k == 0 && continue
        win = b.timestamps[k]
        return between(r, ctx, SpotPrice, u, win, win)
    end
    SpotPrice[]
end
```

The cheaper alternative — apply `_collapse_duplicates!` to the existing
single-block slice — is one line and correct for the case that exists today. It
is rejected because it leaves three call sites each remembering to apply the
rule, which is the shape of the defect being fixed. Routing through `between`
makes "every spot read obeys the rule" true by construction, and it deletes the
duplicated block-slicing rather than adding to it.

**Cost, and why it is acceptable.** The `between` call re-derives the candidate
partitions and consults both blocks instead of the one already in hand. Both are
cached (`_partitions` is a per-reader `Dict`, `_block` a bounded LRU), so the
extra work is two searchsorted pairs on vectors already in memory, on a shape
called roughly once per run in practice (`run_experiment`'s window-end lookup,
the surface reader's curve reads). If it ever shows up, the fallback is the
one-liner.

**Two behaviour changes to state, neither a regression.** `asof` can now throw
`ConflictingRecords` — which is the point, and it is the only way the conflict
across the overlap gets reported at all. And it now merges the two candidate
partitions at the winning instant rather than reading one: under the
time-ordered convention commit 5 of the previous sequence established, that
returns the same record, and where the overlap duplicates a row it collapses it
instead of returning the later partition's copy. Both follow from the rule
already documented in `market_data.md`.

### Files

- **Src:** `src/market_data/parquet.jl`, the `asof(::ParquetSpotsReader, ...)`
  method only. The bars reader is deliberately untouched — bars are outside the
  rule by decision 5's "Deferred", because a chain has many rows per timestamp
  by design.
- **Docs (rule 1):** `docs/modules/market_data.md`, the *Spot de-duplication*
  paragraph, which currently names `between` and `timestamps`. It becomes: every
  spot read obeys the rule, and `asof` obeys it by going through `between`.
- **Tests:** `test/market_data/test_parquet.jl`. Three cases in the existing
  duplicate-tree block (it already writes its own small tree): identical
  duplicates at the winning instant collapse under `asof`, in one partition and
  across the overlap; and `asof == at(last(timestamps(...)))` still holds over
  the duplicated tree. `asof` joins `at` and `between` in the same-instant
  conflict block. Plus one tree the original write-up did not foresee: the two
  candidate partitions carrying one instant at *different* prices, where the
  direct read returned the later price silently — `asof` throws there now.

Revertible alone: yes.

---

## 2. Fetch the chain once per tick, not once per order

Half of finding B's suggested direction, and the half that needs no measurement,
no lifecycle change and no cache.

### What is wrong

`run_backtest`'s per-order loop calls `resolve_quote(cut, trd, t)`, and
`resolve_quote` opens with `chain = at(cut, OptionQuote, trade.underlying, t)`.
So a tick emitting *n* orders performs *n* full-chain reads, each of which
re-synthesizes the whole `OptionQuote` chain from the cached `OptionBar` chain.
The chain is identical across those *n* calls by construction — same map, same
selector, same instant, inside one loop iteration.

### Fix

Split `resolve_quote` rather than change it:

```julia
resolve_quote(chain::AbstractVector{OptionQuote}, trade::Trade, t::DateTime) -> OptionQuote
resolve_quote(cut::TimeCut, trade::Trade, t::DateTime) =                        # unchanged
    resolve_quote(at(cut, OptionQuote, trade.underlying, t), trade, t)
```

and hoist the fetch in `run_backtest`:

```julia
isempty(orders) && continue
chain = at(cut, OptionQuote, first(orders).underlying, t)
for trd in orders
    qte = resolve_quote(chain, trd, t)
    ...
end
```

**One thing to get right.** The hoisted chain is fetched for *one* underlying,
and `resolve_quote` matches on `underlying` among its four fields, so a tick
emitting orders on two underlyings would silently find no match and error with
the existing "no matching quote in chain" message rather than the truthful one.
Today that cannot happen — commit 8 of the previous sequence made one
experiment, one underlying a load-time invariant — but the hoist turns a
would-be correct read into a confusing error, so it must group:

```julia
for (u, group) in _orders_by_underlying(orders)
    chain = at(cut, OptionQuote, u, t)
    for trd in group; ...; end
end
```

For the single-underlying case that is one group and one fetch, which is the
whole win. Anything else is a policy the load-time check already rejects, and
the grouping keeps the engine honest if that check is ever relaxed.

The existing public signature keeps working, so `test/policies/test_policy.jl:171`
and every other caller are untouched.

### Files

- **Src:** `src/backtest/engine.jl`.
- **Docs (rule 1):** `docs/modules/backtest.md` — `resolve_quote` gains an
  arity, and the tick-loop description should say the chain is read once per
  underlying per tick.
- **Tests:** `test/backtest/test_engine.jl` — the new arity resolves against a
  chain directly; a two-order tick fills both legs against one fetch (assert via
  a counting provider in the shape of `_SF_CountingBars`); the cut arity still
  behaves as before.

Revertible alone: yes, and independent of commit 4 — if the cache never lands,
this still removes the per-order multiplier.

---

## 3. Measure finding B

### The claim, and the repo fact that cuts against it

`at(::QuotesFromBars, ...)` rebuilds the entire `OptionQuote` chain from the
cached `OptionBar` chain on every call. The reasoning is correct. The impact is
not established, and one fact argues it is small: `DailyShortStrangle`
implements `tick_times`, so the only real config
(`configs/strangle_spy_16d_1dte.toml`) calls `decide` roughly once per *day*, not
once per minute. The per-tick multiplier applies to a policy that does not narrow
the grid, which today does not exist.

Per tick that does fire, the passes are: one from the surface reader, one from
`decide`'s own chain read, and one per order from `resolve_quote` — so `n + 2`,
which commit 2 reduces to `3`, and a cache would reduce to `1`.

### The script

`scripts/bench_quote_synthesis.jl`, modelled on `scripts/bench_point_vs_range.jl`
(same shape: activate the project, open a real reader over `~/data/massive`,
enumerate a month, warm up on the first day so compilation is out of the numbers,
report wall time, `@allocated`, the `Sys.maxrss()` delta and the record count,
then assert the variants agree on that count).

Arms, over one month of SPY minute data:

| arm | what it measures |
|---|---|
| **A** | `at(m, OptionQuote, u, t)` once per timestamp — the baseline synthesis cost |
| **B** | three calls per timestamp — the engine's real per-tick shape *after* commit 2 |
| **C** | `n + 2` calls per timestamp with `n = 2` — the shape *before* commit 2, so the commit's own value is measured, not assumed |
| **D** | arm B against a `QuotesFromBars` reader holding an `LRU{Tuple{Underlying,DateTime},Vector{OptionQuote}}` — the candidate fix |

`A` also gives the per-record synthesis cost, which is the number that
generalises to other configs.

### The whole-run number

The only end-to-end figure anyone acts on. `run_experiment` on the strangle
config over **one month** (not the config's ten-year window), before and after,
via a `--from` / `--to` override or a trimmed copy of the config. Report wall
time and peak RSS.

Expect the difference here to be small precisely because `tick_times` narrows
the grid to ~21 ticks in a month; that is the point of measuring rather than
reasoning. If arm D is dramatic and the whole-run number is not, the honest
conclusion is "the cache matters for a policy that does not yet exist", and
commit 4 stays deferred with the numbers recorded.

### Threshold for acting

The caching variant has to move the **whole-run** number by something worth a
lifecycle change. `QuotesFromBars` is currently its own reader
(`open_data(s::Union{InMemory,Constant,QuotesFromBars}) = s`); giving it a real
one adds a spec/reader pair, a `serves` method on the reader, an entry in the
lifecycle suite, and a cut-independence argument that has to be made the way
`SurfaceReader`'s was. A 2% whole-run improvement does not buy that; a 20% one
does. Record the actual number and the decision in
[pr9_followups.md](pr9_followups.md), whichever way it goes.

### Files

`scripts/bench_quote_synthesis.jl` (new), and the results appended to
`pr9_followups.md` as a numbered section the way section 10 of the data-kinds
proposal records its benchmarks. No `src/` change, no doc rule engaged.

---

## 4. *Conditional:* `QuotesFromBars` gains a reader

**Write this only if commit 3 says so.** Specified here so the decision in
commit 3 is made against a known cost, not an unknown one.

- `QuotesReader(spec, cache::LRU{Tuple{Underlying,DateTime},Vector{OptionQuote}})`,
  with `kind`, `inputs`, and `serves(...) = missing` (still derived, still
  delegates).
- `open_data(s::QuotesFromBars; max_chains::Int=10) = QuotesReader(...)`,
  `close_data!(::QuotesReader) = nothing`; `QuotesFromBars` drops out of the
  resource-free union in `lifecycle.jl`.
- **Only `at` is cached.** `between`, `asof` and `timestamps` stay
  pass-through. This is not an optimisation choice, it is the cut-independence
  argument: `at(r, m, OptionQuote, u, ts)` reads `at(m, OptionBar, u, ts)` at
  *exactly* `ts`, so an entry keyed `(u, ts)` is valid under any cutoff `>= ts` —
  the same invariant `SurfaceReader` relies on, and a stronger form of it
  (exact instant rather than "at or before"). `asof` resolves a *different*
  instant under a different cut and must not be keyed on the requested one; the
  parquet bars reader draws the line in exactly the same place.
- **Identity is untouched.** The cache bound is an `open_data` kwarg, not a spec
  field, so `to_dict(::QuotesFromBars)` does not change and no stored run's
  `run_id` moves. Contrast `lookback_ticks`, which *is* a spec field because it
  changes which record a policy sees; a cache size changes nothing but speed.
  This is the distinction `market_data.md` already draws.
- **Docs (rule 1):** `market_data.md` — the lifecycle opt-in list (the
  resource-free union shrinks), the derived-provider section, and the
  cache-bounds paragraph commit 9 of the previous sequence added, which says
  bounds are `open_data` kwargs on individual specs; `experiment.md` if the
  "new provider spec needs" checklist is affected.
- **Tests:** `test/market_data/test_lifecycle.jl` (`QuotesFromBars` moves from
  the resource-free list to the open/close list), a bounded-cache case in the
  shape of `test_surface_from.jl`'s `max_surfaces=1` test, and a cut-independence
  case asserting the same chain object comes back through the bare map and
  through a cut at or after its timestamp.

Revertible alone: yes, though it is the one that touches `lifecycle.jl`, so a
revert should be checked against the lifecycle suite rather than assumed.

---

## Running this on the constrained machine

Same constraints as the previous sequence: 2 cores, 3.7 GB, Julia 1.12 via
juliaup, a warm REPL in the tmux `julia` pane that must not be disturbed.

- Per commit, run only the suites it touches, from a fresh `julia --project=.`
  process: commit 1 → `test_parquet.jl`; commit 2 → `test_engine.jl` and
  `test_policy.jl`; commit 4 → `test_lifecycle.jl`, `test_providers.jl`,
  `test_parquet.jl`.
- Full `Pkg.test()` once, at the end of whichever commit is last.
- **The whole-run benchmark needs headroom.** A `--save` run of the strangle
  config peaks around 1.7 GB, and the box has ~1.3 GB free with the REPL warm.
  Run the one-month comparison without `--save`, in a pane via `ws run`, and if
  it is still tight, exit the REPL first *and say so* — it may hold expensive
  state.
- The benchmark reads `~/data/massive` (2974 date partitions present), so arm A
  over one month is real data, not a fixture.
