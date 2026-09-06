# Implementation plan: kinds, providers, readers (proposal v3 + A3 fixes)

Status: plan, produced 2026-09-06 on the `data-kinds` branch from
[data_kinds.md](data_kinds.md) v3 with Review A3's fixes folded in.
Review B3's structural asks stay declined (proposal section 9).

Facts gathered on the DevBox that shape the plan:

- Branch `data-kinds` at `61e3879`; the working tree has an unrelated
  `README.md` edit and untracked `docs/nvim.md` and
  `configs/strangle_spy_16d_1dte.local.toml`, so `code_provenance()`
  would stamp the baseline `dirty=true` unless cleaned up first.
- Data: 2974 option days (2014-06-02 to 2026-03-27), 2559 spot days
  (2016-03-28 on). One SPY option day is 102,811 rows over 390 distinct
  timestamps (281 rows at the 19:30 tick). The SPY spot file for a day
  spans 09:00 UTC to 00:59 UTC next day, i.e. partition date !=
  `Date(timestamp)` after midnight.
- `Base.between` exists but is unexported and Integer-only; none of
  `at`, `asof`, `timestamps`, `kind`, `entry`, `selector`, `Clock`,
  `Currency` exist in Base or Dates. No existing export collides.
- LRUCache.jl is not in the depot; OrderedCollections is a dep with
  `_lru_touch!`/`_lru_evict!` already tested.
- `ws run` sends to the `dev:shell` tmux window;
  `scripts/run_experiment.jl --save` writes to
  `scripts/runs/run_id=<id>/` (gitignored).

Every commit keeps the suite green and updates module docs in the same
commit (design rule 1); rule-changing commits say so in the message
(rule 3); `status.md` moves at the milestones marked below (rule 4).

Conventions used below:

- "Map-level" shape = 4 arguments, called by consumers:
  `at(m, R, sel, ts)`. "Provider-level" shape = 5 arguments, the context
  first after the provider: `at(p, ctx, R, sel, ts)`. `TimeCut` and
  `MarketData` are map-level; specs and readers are provider-level.
- `asof` returns `Vector{R}`: every record at the largest visible
  timestamp `<= ts`, empty when none (A3 fix 1). "Empty means absent"
  holds for all four shapes; `missing` never comes out of a shape.
- Module split: `src/data/` keeps *what a datum is* (records, vendor row
  mapping: `quotes.jl`, `polygon.jl`, `synth.jl`); new
  `src/market_data/` holds *how it is obtained* (protocol, specs,
  readers, map, cut, lifecycle, clock). `SurfaceFrom` lives with
  `surfaces` (proposal 6.4). Justification: it mirrors the proposal's
  own split between 2.1 and 2.2–2.8, and lets the old
  `source.jl`/`parquet_source.jl` be deleted at step 3 without moving
  any new file.
- `OptionBar`, `OptionQuote`, `SpotPrice`, `Underlying` stay exactly
  where they are (`src/data/quotes.jl`, `src/data/synth.jl`) through
  all steps. `market_data/kinds.jl` only adds `selector`/`selector_type`
  methods on them. Nothing is duplicated.
- The proposal's `VolSurface` kind is the existing abstract
  `VolatilitySurface` (concrete `RawSurface` already carries
  `underlying` and `timestamp`). No wrapper record: policies keep
  calling `invert_delta(surface, ...)` directly. The loader's kind name
  is `"vol_surface"`.

---

## Step 0 — baseline run and convention check (2 commits, no library code)

### Commit 0a: housekeeping so the baseline has clean provenance, plus the comparison script

Files:
- `.gitignore`: add `*.local.toml`.
- `docs/nvim.md`, `README.md`: commit the pending unrelated change
  (they are already written).
- New `scripts/compare_runs.jl <store_root> <run_id_a> <run_id_b>`:
  opens one DuckDB connection and compares the two run folders
  **without** `load_run` (which will refuse old-schema runs after commit
  2.3). It prints and exits non-zero on any difference in:
  `positions.parquet` (all columns except `run_id`, joined on
  `leg_idx`), `pnl_series.parquet` (`idx`, `timestamp`, `pnl` within
  `1e-9`), `metrics.parquet` (per `metric_name`, `abs(a-b) <= 1e-9`,
  NaN==NaN), and manifest `window_end_spot`, `n_unmarked`, `n_opens`,
  `n_closes`. Self-check: `compare_runs.jl scripts <id> <id>` must
  pass.

Acceptance: `git status` clean after the commit;
`julia --project=. scripts/compare_runs.jl` prints usage.

### Baseline run (no commit; artifacts are gitignored)

```
mkdir -p scratch
# 1. precompile once on 2 cores
ws run 'cd /home/ale/dev/VolSurfaceAnalysis.jl && JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"'
ws wait shell '\$ $' 3600
# 2. record the expected run id before running (identity is config-derived)
julia --project=. -e 'using VolSurfaceAnalysis; e = load_experiment("configs/strangle_spy_16d_1dte.local.toml"); println("full=", full_hash(e), " core=", core_hash(e))'
# 3. the run, detached in the tmux shell window, log kept
ws run 'cd /home/ale/dev/VolSurfaceAnalysis.jl && julia --project=. scripts/run_experiment.jl configs/strangle_spy_16d_1dte.local.toml --save 2>&1 | tee scratch/step0_baseline.log'
ws wait shell 'saved run|ERROR|Error' 28800     # poll; a 10-year daily strangle is tens of minutes on this box
ws capture shell 60
```

Expected RunStore path: `scripts/runs/run_id=<full_hash from step 2>/`
containing `config.toml`, `manifest.parquet`, `metrics.parquet`,
`positions.parquet`, `pnl_series.parquet`, `artifacts/equity_curve.png`.
Confirm the manifest:

```
julia --project=. -e 'using DuckDB; db = DuckDB.DB(":memory:"); for r in DBInterface.execute(db, "SELECT run_id, core_hash, name, from_ts, to_ts, n_positions, n_unmarked, window_end_spot, commit_sha, dirty FROM read_parquet(\"scripts/runs/*/manifest.parquet\")"); println(r); end'
```

Gate: exactly one row; `dirty == false`; `commit_sha` == HEAD of commit
0a; `run_id` equals the precomputed `full_hash`. Then
`julia --project=. scripts/compare_runs.jl scripts <id> <id>` passes.
Write `<id>`, `core_hash`, `n_positions`, `n_unmarked`,
`window_end_spot`, wall time and peak RSS (from the log /
`Sys.maxrss()` printed by the script if added) into
`scratch/step0_baseline.txt` and into section 10 (commit 0b).

Memory note: the box shows 2.6 GB of 3.7 GB in use; `--save` loads
Plots for the artifact. Close the `codex` tmux window or the REPL if the
run is OOM-killed, and rerun.

### Commit 0b: proposal text fixes (A3) and section 10

`docs/proposals/data_kinds.md`:
- 2.1: declare the bar-time visibility convention (decision in Risks:
  keep bar-open stamps as a documented one-minute allowance, not a
  shift).
- 2.2: `asof(...) -> Vector{R}` "every record at the largest visible
  timestamp `<= ts`, empty when none"; drop the "two records at the
  winning timestamp throws" rule and the `missing` return; 6.1 loses
  the `asof` exception.
- 2.4/Appendix B: `Constant` checks `selector(c.record) == sel` in
  `asof`, `between`, `timestamps`.
- 2.8/Appendix B: window end = last clock tick, obtained as the
  timestamp of `asof(data, OptionQuote-clock-kind, clock.sel, to)` (one
  partition walk, no scan); the spot is `at(data, SpotPrice, u,
  window_end)`. `Clock{R,S}` typed selector.
- Minor items: `timestamps` forwarding shown for `QuotesFromBars`,
  `SurfaceFrom`, `BySelector`; best-effort unwind in `open_data`;
  sorted `BySelector` parts and `spot_for` in `to_dict`; the
  derived-cache invariant ("a derived provider reads its inputs at or
  before the requested `ts`, so a `(sel, ts)` cache entry is valid
  under any cutoff `>= ts`"); 2.6 says "two runs", no `core_hash`
  family.
- Section 10 filled with the convention check (checklist below) and the
  baseline record.
- Status line: "v3.1, accepted; implementation in progress".

`docs/status.md`: in-flight entry says step 0 done, baseline `<id>`
saved on the DevBox.

Rule-5 checklist (each item: what to look up, what to record in section
10):
1. **Tables.jl** — `Tables.partitions(x)` (iterator of tables),
   `Tables.rows`/`Tables.columns`, and how DuckDB.jl results implement
   them. Record: our `between` is an iterator of *records*, not of
   tables, so `Tables.partitions` is not the right hook; the day-lazy
   parquet iterator mirrors its "one partition in memory at a time"
   contract; `Tables.columntable` stays the materialization path inside
   `_day_bars`.
2. **DBInterface.jl** — `DBInterface.connect(T, ...)`,
   `DBInterface.close!(conn)`, `execute`, `prepare`; and Base's
   `open(f, ...)`/`close`/`isopen`. Record: the ecosystem uses a
   project-owned verb pair with a bang on the mutating close
   (`close!`), never catch-all methods on `Base.open`/`Base.close`;
   `open_data`/`close_data!` follow that, and `with_data(f, m)` follows
   the in-repo `with_run_store`/`with_parquet_source` and Base
   `mktempdir(f)`/`redirect_stdout(f)` scoped-form precedent.
3. **Type-marker dispatch** — `Base.read(io, ::Type{T})`,
   `Base.parse(::Type{T}, s)`, `Base.rand(rng, ::Type{T})`,
   `JSON3.read(s, ::Type{T})`, `StructTypes.StructType(::Type{T})`,
   `Tables.schema`. Record: source first, `::Type{R}` second (as
   `read(io, T)`); `selector_type(::Type{R})`/`kind(p)` are
   `StructTypes`-style traits; providers have no abstract supertype
   (duck-typed protocol, as Tables.jl).
4. **As-of conventions** — TimeSeries.jl (`from`/`to`/`findwhen`, exact
   `ta[dt]` indexing), DataInterpolations.jl
   (`ConstantInterpolation(...; dir=:left)`), pandas
   `Series.asof`/`merge_asof`, Impute.jl `locf`. Record: Julia has no
   established name; `asof` is taken from pandas, `between(from, to)`
   is chosen over TimeSeries' `from`/`to` pair, and `Base.between` is
   unexported and Integer-only, so no clash (and `import Base: between`
   must never be written).
5. **Measured inference** — filled in at commit 1.3 (`@inferred entry`,
   `BySelector` routing) and revisited at 2.2 on a config-built map
   (`open_data` return type, `@allocated entry`).
6. **Naming/layout** — Julia manual "Style Guide" (no `get_` prefixes
   for accessors, bang for mutation), "Interfaces" (iteration protocol
   for `between`/`by_timestamp`), package layout (files `include`d, no
   submodules — matches existing modules). Record one line each.

---

## Step 1 — the new layer beside the old (6 commits)

Shared test scaffolding, introduced in 1.1 and grown per commit:
`test/market_data/fixtures.jl` (in-memory `OptionBar`/`SpotPrice` rows
for SPY and SPX at three timestamps; later the parquet writers ported
from `test/data/test_parquet_source.jl`). All `test/market_data/*`
constants are prefixed `_MD_` to avoid clashing with `_TC_`/`_EN_`/`_EX_`
in the old suites, since `runtests.jl` includes everything into one
module.

### Commit 1.1 — kinds, selector contract, protocol stubs, library

Files:
- `src/market_data/kinds.jl`: `struct Currency; code::String; end`
  (uppercase-normalizing constructor, `Base.show`, mirrors
  `Underlying`); `selector(r)` and `selector_type(::Type{R})` for
  `OptionBar`, `OptionQuote`, `SpotPrice` (`Underlying`). Docstring
  states the rule: `timestamp` is visibility time on every kind.
- `src/market_data/protocol.jl`: `function at end; function between
  end; function asof end; function timestamps end; function kind end`;
  the provider-level default `at(p, ctx, ::Type{R}, sel, ts) where R =
  collect(between(p, ctx, R, sel, ts, ts))`.
- `src/market_data/library.jl`: `only_or_missing(v)`; `by_timestamp(it)`
  as a lazy iterator type `ByTimestamp{I}` yielding `(ts::DateTime,
  Vector{R})` run-length groups of a sorted iterable.
- `src/VolSurfaceAnalysis.jl`: includes after `data/parquet_source.jl`;
  exports `Currency, selector, selector_type, at, between, asof,
  timestamps, kind, only_or_missing, by_timestamp`.
- `docs/modules/market_data.md` (new): kinds and the visibility rule,
  selector contract, the four shapes and their rules (sorted,
  empty=absent, `between` is an iterable valid while the reader is
  open, `asof` has no default, ranges bounded), map-level vs
  provider-level arity, library.
- Tests `test/market_data/test_kinds.jl` (`selector`/`selector_type`
  per kind, `Currency("usd") == Currency("USD")`),
  `test/market_data/test_library.jl` (`only_or_missing` empty→missing,
  one→record, two→throws; `by_timestamp` groups a mixed sorted vector
  correctly, is lazy — `Iterators.take` does not consume the rest,
  empty input yields nothing).
- `test/runtests.jl`: include the two files.

Acceptance: suite green; `names(VolSurfaceAnalysis)` shows the new
exports; no method of `Base.between` was added
(`length(methods(Base.between)) == 1`).

### Commit 1.2 — InMemory, Constant, QuotesFromBars, MarketData, TimeCut, Clock

Files:
- `src/market_data/providers.jl`:
  - `InMemory{R}(rows)`: stores `sort(rows; by = r -> r.timestamp)`
    (stable). Provider-level `between` (filter selector + range), `asof`
    (filter selector, `searchsortedlast` by timestamp, return all rows
    sharing that timestamp), `timestamps` (unique sorted in range),
    `kind`.
  - `Constant{R}(record)`: `asof` returns `[c.record]` iff
    `selector(c.record) == sel` (A3 fix 2), `between`/`timestamps`
    include it only when the selector matches and `from <=
    record.timestamp <= to`.
  - `QuotesFromBars{Q<:QuoteSynthesizer}(synthesizer)`: `kind =
    OptionQuote`; `at`, `between` (`Iterators.map`), `asof`,
    `timestamps` all read `OptionBar` through the context `m` with the
    same selector. `inputs(::QuotesFromBars) = (OptionBar,)`;
    `inputs(::Any) = ()` (used by the loader in 2.2).
- `src/market_data/map.jl`: `struct MarketData{P<:Tuple}` with an inner
  constructor rejecting empty tuples and duplicate kinds;
  `MarketData(specs...)`; `entry(m, ::Type{R})` via `@inline _entry`
  recursion with a clear `error("MarketData has no provider for $R")`;
  the four map-level shapes forwarding `entry(m, R)` with `m` as
  context.
- `src/market_data/time_cut.jl`: `struct TimeCut{M}; inner::M;
  cutoff::DateTime; end` and the four map-level shapes exactly as
  proposal 2.7 (returning `R[]`/`DateTime[]` when `ts > cutoff` or
  `from > cutoff`; `asof` clamps `min(ts, cutoff)`), passing `c` itself
  as context. `entry(c::TimeCut, R) = entry(c.inner, R)`.
- `src/market_data/clock.jl`: `struct Clock{R,S}; sel::S; end` with
  `Clock{R}(sel)` checking `sel isa selector_type(R)`; `timestamps(m,
  c::Clock{R}, from, to) = timestamps(m, R, c.sel, from, to)`.
- Exports: `InMemory, Constant, QuotesFromBars, MarketData, entry,
  TimeCut, Clock, inputs`.
- `docs/modules/market_data.md`: sections "Provider specs", "Derived
  providers", "The map", "Time cut" (structural no-lookahead through
  derived providers), "Clock".
- Tests:
  - `test/market_data/test_providers.jl`: for `InMemory{SpotPrice}` with
    SPY and SPX rows interleaved: `at` exact hit/miss, `between` sorted
    and selector-filtered, `at == collect(between(ts, ts))`, `asof`
    returns all records at the winning timestamp (fixture with two
    `OptionBar`s at one timestamp), `asof` before first → empty,
    `timestamps` bounded. `Constant{SpotPrice}`: `asof` for the right
    selector → one record, for another selector → empty (fix 2);
    `between` over a 2024 window → empty; `timestamps` → empty.
    `QuotesFromBars`: `at(m, OptionQuote, SPY, t)` synthesizes exactly
    `at(m, OptionBar, SPY, t)` (bid/ask numbers as in the existing synth
    test), `timestamps(m, OptionQuote, ...) == timestamps(m, OptionBar,
    ...)`.
  - `test/market_data/test_map.jl`: `entry` hit, missing kind errors
    with the kind name, duplicate kind rejected at construction,
    `@inferred entry(m, OptionBar)`, `@inferred at(m, OptionQuote, SPY,
    t)`.
  - `test/market_data/test_time_cut.jl`: each shape masked past the
    cutoff; `between` with `from > cutoff` → empty; `asof` clamps;
    **cut-through-derived**: with `cut = TimeCut(m, t1)`, `at(cut,
    OptionQuote, SPY, t2) == OptionQuote[]` and `asof(cut, OptionQuote,
    SPY, t3)` returns the `t1` chain — the derived provider's
    `OptionBar` read went through the cut; `timestamps(cut,
    Clock{OptionQuote}(SPY), t1, t3) == [t1]`.
  - `test/market_data/test_clock.jl`: `Clock{OptionQuote}(Currency("USD"))`
    throws; `clock.sel` is concretely typed
    (`typeof(clock).parameters[2] === Underlying`).

Acceptance: suite green; `@inferred` tests pass.

### Commit 1.3 — BySelector

Files:
- `src/market_data/by_selector.jl`: `struct BySelector{R,P<:Tuple};
  parts::P; end` with the constructor `BySelector{R}(parts::Pair...)`
  rejecting empty, mixed-kind, duplicate-selector, and wrong selector
  type (`first(p) isa selector_type(R)`) lists; `_route(sel, parts...)`
  recursion throwing `KeyError(sel)`; the four provider-level shapes
  forwarding the context untouched (including `timestamps`). Exports
  `BySelector`.
- `docs/modules/market_data.md`: "Composition: `BySelector`"
  (invariants, union-split routing).
- Tests `test/market_data/test_by_selector.jl`: constructor rejections;
  routing SPY→`InMemory{SpotPrice}` A, SPX→`InMemory{SpotPrice}` B for
  every shape; unknown selector → `KeyError`; `@inferred at(m,
  SpotPrice, SPY, t)` and `@inferred asof(...)` where the `SpotPrice`
  entry is a two-part heterogeneous `BySelector` (return type
  `Vector{SpotPrice}` regardless of branch); a
  `@code_warntype`-equivalent assertion: `Base.return_types(at,
  (typeof(m), Type{SpotPrice}, Underlying, DateTime)) ==
  [Vector{SpotPrice}]`. Record the observed `@code_warntype` output for
  `entry` and the routed `at` in proposal section 10 (this commit edits
  the proposal).

Acceptance: suite green; section 10 has the inference paragraph.

### Commit 1.4 — lifecycle

Files:
- `src/market_data/lifecycle.jl`: `function open_data end; function
  close_data! end`. Explicit opt-in methods:
  `open_data(s::Union{InMemory,Constant,QuotesFromBars}) = s`,
  `close_data!(::Union{InMemory,Constant,QuotesFromBars}) = nothing`.
  `open_data(::BySelector{R})` and `open_data(::MarketData)`
  implemented with a **recursive, type-stable tuple open**:
  ```julia
  _open_all() = ()
  function _open_all(s, rest...)
      r = open_data(s)
      tail = try _open_all(rest...) catch; _close_quietly(r); rethrow() end
      (r, tail...)
  end
  ```
  `_close_quietly(r)` swallows and `@warn`s a close error so the
  original exception always propagates (A3 minor + B3 item 4).
  `close_data!(::MarketData)` / `close_data!(::BySelector)` share
  `_close_all_best_effort(readers)`: reverse order, every close
  attempted, first error rethrown after the loop. `with_data(f, m)`:
  close quietly on an exception from `f`, close normally on success, so
  a close error never masks `f`'s error. `has_lifecycle(spec) =
  hasmethod(open_data, Tuple{typeof(spec)})` for the loader.
- Exports `open_data, close_data!, with_data`.
- `docs/modules/market_data.md`: "Lifecycle" (spec vs reader, no `Any`
  fallback, unwind, best-effort close, use-after-close is the storage's
  error).
- Tests `test/market_data/test_lifecycle.jl` with test-local specs:
  `_TrackSpec(id, log)` whose reader appends `(:open, id)`/`(:close,
  id)` to a shared log, `_FailOpenSpec` throwing in `open_data`,
  `_FailCloseSpec` throwing in `close_data!`. Assert: **open-failure
  unwind** — map `(A, B, FailOpen)` throws the FailOpen error and the
  log is `[open A, open B, close B, close A]`; unwind where B's close
  also throws still propagates the FailOpen error; `close_data!` on
  `(A, FailClose, C)` closes C and A and rethrows FailClose;
  `with_data` closes on success and on `f` throwing (the `f` error is
  what propagates); a `BySelector` of tracked parts opens and closes in
  order; a spec without `open_data` fails `has_lifecycle`.

Acceptance: suite green.

### Commit 1.5 — parquet specs and readers, ported from `parquet_source.jl`

Files:
- `src/data/polygon.jl`: receive `ContractMeta`,
  `_contract_meta_from_parsed` from `parquet_source.jl` (vendor mapping,
  storage-agnostic); `parquet_source.jl` keeps using them. Minimal
  move, old tests unchanged.
- `src/market_data/lru.jl`: `struct LRU{K,V}; d::OrderedDict{K,V};
  max::Int; end` with `Base.get!(f, ::LRU, k)`, `haskey`, `length`,
  `empty!`, wrapping the existing touch/evict logic (copied, not
  shared, so the old file stays deletable). Decision: no LRUCache.jl
  dependency (see Risks).
- `src/market_data/parquet.jl`:
  - Specs `ParquetOptionBars(root)` and `ParquetSpots(root)`: `root` is
    the *kind-specific* directory (`.../options_1min`,
    `.../spots_1min`) — one spec per storage tree, no subdir defaults
    inside the spec; `kind`, and `inputs() = ()`. Construction is pure
    (no `isdir`, no warning) so rehydrating a saved run off-machine is
    silent.
  - Readers `ParquetBarsReader` / `ParquetSpotsReader`: `spec`,
    `con::DuckDB.DB`, `partitions::Dict{Underlying,Vector{Date}}`
    filled lazily per selector by listing `date=*/symbol=<T>/data.parquet`
    once (`_partitions(root, u)`),
    `days::LRU{Tuple{Underlying,Date},PartitionMeta}` (distinct
    timestamps + column flags, ported `DayMeta`),
    `chains::LRU{Tuple{Underlying,DateTime},Vector{OptionBar}}`,
    `contracts::Dict{String,ContractMeta}`; spots reader:
    `blocks::LRU{Tuple{Underlying,Date},SpotBlock}`.
  - `open_data(s::ParquetOptionBars; max_days=200, max_chains=10)`
    throws `ArgumentError` if `!isdir(s.root)`; `close_data!` =
    `DBInterface.close!(con)`.
  - Bars: `at` = ported `_load_chain_at` returning `Vector{OptionBar}`
    (no synthesis; ticker-underlying mismatch still throws), through the
    chain LRU; `between` = `Iterators.flatten(_day_bars(r, u, d, from,
    to) for d in _candidate_days(from, to))` where `_day_bars` runs one
    `WHERE timestamp BETWEEN ... ORDER BY timestamp` query per day and
    returns a fresh vector (not cached); `asof` walks `partitions[u]`
    backward from `searchsortedlast(partitions, Date(ts))`, consulting
    `PartitionMeta.timestamps` (cached) until a day has a timestamp
    `<= ts`, then `at` on that timestamp; `timestamps` = ported
    `available_timestamps` over the partition list intersected with the
    range (no per-day `isfile` when the partition list says the day is
    absent).
  - Spots: block per partition day, `at` binary search, `between` day
    walk, `asof` backward walk, `timestamps` from blocks.
  - Partition/timestamp mapping convention (see Risks 8):
    `_candidate_days(from, to) = (Date(from) - Day(1)):Day(1):Date(to)`
    intersected with the partition list, and `at`/`asof` use the same
    rule, so `at == collect(between)` holds by construction even for
    spot rows stamped after midnight UTC.
- Exports `ParquetOptionBars, ParquetSpots`.
- `docs/modules/market_data.md`: "Parquet readers" (partition list as
  the bound for `asof`, one day in memory for `between`, chain LRU only
  for `at`, the partition-day convention, the bar-time allowance
  repeated from `data.md`); `docs/modules/data.md`: vendor mapping now
  produces `OptionBar` only; synthesis is `QuotesFromBars`.
- Tests `test/market_data/test_parquet.jl` (fixture writers moved into
  `test/market_data/fixtures.jl`; the old
  `test/data/test_parquet_source.jl` keeps its own copies until step 3
  — duplication is temporary and deliberate): every existing parquet
  assertion re-expressed on the new API (hit/miss timestamp/day, mark
  and volume, parsed_* authoritative, volume/OHLC absent → `missing`,
  ticker mismatch throws, LRU eviction on `r.chains`, single-root
  convenience is now the loader's job so it is not tested here), plus
  new: **`at == collect(between)`** for every fixture timestamp and
  `collect(between(t1a, t2a)) == vcat(at(t1a), at(t1b), at(t2a))`;
  `asof` at `t1b + 1min` → the `t1b` chain, at `t2a - 1min` (a day
  gap) → the `t1b` chain, before `t1a` → empty; `timestamps` equals the
  old `available_timestamps`; spots `at`/`asof`/`between`/`timestamps`,
  including a row at `2024-01-16T00:30` written into the
  `date=2024-01-15` partition found by `at`, `asof(2024-01-16T00:45)`
  and `between`; `open_data` on a missing root throws; use after
  `close_data!` throws; `with_data` on a `MarketData((ParquetOptionBars,
  QuotesFromBars, ParquetSpots))` yields synthesized quotes equal to the
  old `ParquetDataSource.get_chain` on the same fixture (cross-check
  against the old layer while it exists). Port the `VSA_POLYGON_ROOT`
  real-data smoke.

Acceptance: suite green; running the new-vs-old cross-check with the
real tree for one day (`VSA_POLYGON_ROOT=~/data/massive`) shows
identical quote vectors.

### Commit 1.6 — curve kinds, `SurfaceFrom`, `docs/modules/model_data.md` kept coherent

Files:
- `git mv src/model_data/curves.jl src/market_data/curves.jl`; append
  `struct RateCurve; currency::Currency; curve::Curve;
  timestamp::DateTime; end`, `struct DivCurve; underlying::Underlying;
  curve::Curve; timestamp::DateTime; end`, two-arg constructors
  defaulting `timestamp = typemin(DateTime)` (the `Constant` case),
  `selector`/`selector_type`. `git mv test/model_data/test_curves.jl
  test/market_data/test_curves.jl` plus kind tests. Include order:
  `market_data/curves.jl` replaces the old `model_data/curves.jl` line;
  `model_data/source.jl` (old) still compiles because `Curve` is
  defined earlier.
- `src/surfaces/surface_from.jl`:
  `selector(::VolatilitySurface)`/`selector_type`; `struct SurfaceFrom;
  spot_for::Dict{Underlying,Underlying}; currency::Currency; end`
  (kwarg constructor with empty `spot_for`), `kind =
  VolatilitySurface`, `inputs = (OptionQuote, SpotPrice, RateCurve,
  DivCurve)`; `struct SurfaceReader; spec;
  cache::LRU{Tuple{Underlying,DateTime},Vector{VolatilitySurface}};
  end`; `open_data(s::SurfaceFrom; max_surfaces=64)`,
  `close_data!(::SurfaceReader) = nothing`; `at` exactly as Appendix B
  but with `only_or_missing(asof(m, RateCurve, ccy, ts))` /
  `only_or_missing(asof(m, DivCurve, u, ts))` (A3 fix 1) and
  `build_surface` returning `nothing` mapped to `VolatilitySurface[]`
  (cached); `between` = `Iterators.flatten(at(...) for ts in
  timestamps(...))`; `asof` = `at` at the timestamp of `asof(m,
  OptionQuote, u, ts)` (empty if that is empty); `timestamps` forwards
  to `OptionQuote`.
- Exports `RateCurve, DivCurve, SurfaceFrom`.
- Docs: `docs/modules/surfaces.md` gains "`SurfaceFrom`: the derived
  provider" (the cache-validity invariant, `spot_for`, failure →
  empty); `docs/modules/market_data.md` "Curve kinds" and the
  snapshot/`asof` shape table; `docs/modules/model_data.md` layout
  section updated to say `Curve` types now live in
  `market_data/curves.jl` and that the module is scheduled for deletion
  at step 3 (rule 1 keeps it truthful while it exists).
- Tests `test/surfaces/test_surface_from.jl`: full in-memory map
  (`InMemory{OptionBar}` BS-priced bars as in
  `test/model_data/test_source.jl`, `QuotesFromBars`,
  `InMemory{SpotPrice}`, `Constant(RateCurve(USD, FlatCurve(0.04)))`,
  `Constant(DivCurve(SPY, FlatCurve(0.015)))`,
  `SurfaceFrom(currency=USD)`): `only_or_missing(at(d,
  VolatilitySurface, SPY, t1))` is a `RawSurface` with `iv ≈ 0.20`;
  absent chain/spot/curve → empty and cached (second call does not
  re-read: wrap the bars entry in a test-local counting provider);
  `spot_for` remap (`SPY => SPX`) uses SPX's spot; cache bounded
  (`max_surfaces=1` evicts); **cut independence**: surface at `t1` via
  `TimeCut(d, t1)` is `===` the one via `d`, and via `TimeCut(d, t2)`;
  a `PCCurve` stepped rate flows into `surface.rate` at the right `ts`;
  `timestamps(d, VolatilitySurface, SPY, ...)` equals the bar
  timestamps.
- `docs/status.md`: step 1 complete (rule 4 milestone).

Acceptance: suite green; the old suites (`test/model_data`,
`test/backtest/test_time_cut.jl`, …) untouched and still green.

---

## Step 2 — consumers switch (4 commits)

### Commit 2.1 — engine, policies, agents, experiment on `MarketData` + `Clock` (transitional loader)

Files and changes:
- `src/backtest/engine.jl`: `resolve_quote(cut::TimeCut, trade, t)` =
  linear scan of `at(cut, OptionQuote, trade.underlying, t)`, error on
  empty/no match. `run_backtest(agent::Agent, data::MarketData, from,
  to, clock::Clock{R})`: `ticks = tick_times(agent, data, from, to)`;
  `ticks === nothing && (ticks = timestamps(data, clock, from, to))`;
  per tick `cut = TimeCut(data, t)`; spot = `only_or_missing(at(cut,
  SpotPrice, trd.underlying, t))`, error if missing (comment: `spot_for`
  remap is ignored at fill time, as today). Policy overload gains the
  `clock` argument.
- `src/policies/policy.jl`, `src/agents/agent.jl`: `decide(::Policy,
  ::DateTime, ::TimeCut, ::AbstractVector{Position})`,
  `current_policy(..., ::TimeCut, ...)`, `tick_times(::Policy,
  ::MarketData, from, to) = nothing` (and Agent/StaticAgent forms).
- `src/policies/daily_short_strangle.jl`: `tick_times(p, ::MarketData,
  from, to)` unchanged body; `decide`: `surface = only_or_missing(at(data,
  VolatilitySurface, p.underlying, t)); ismissing(surface) && return
  Trade[]`; `chain = at(data, OptionQuote, p.underlying, t);
  isempty(chain) && return Trade[]`; rest unchanged.
- `src/experiment/experiment.jl`: `Experiment` fields `name, agent,
  data::MarketData, clock::Clock, from, to, outputs`; kwarg
  constructor. `run_experiment`:
  ```julia
  with_data(exp.data) do d
      positions  = run_backtest(exp.agent, d, exp.from, exp.to, exp.clock)
      last_block = asof(d, kind_of(exp.clock), exp.clock.sel, exp.to)   # one partition walk
      (isempty(last_block) || first(last_block).timestamp < exp.from) && error("no clock ticks in window")
      window_end = first(last_block).timestamp                           # last clock tick (A3 fix 4)
      spot = only_or_missing(at(d, SpotPrice, exp.clock.sel, window_end)); ismissing(spot) && error(...)
      settle = _build_settle(d, window_end, spot.price)
      series = pnl_series(positions; settle, window_end_spot = spot.price)
      ...
  end
  ```
  `_build_settle(d, window_end, spot)`: `expiry > window_end ? spot :
  (s = only_or_missing(at(d, SpotPrice, u, expiry)); ismissing(s) ?
  missing : s.price)`, `u` = `exp.clock.sel` (comment on the `spot_for`
  simplification, A3 minor). Everything that touches `d` runs inside
  `with_data`.
- `src/experiment/identity.jl`: `to_dict(::MarketData)` =
  `Dict("entries" => Dict(kind_name(kind(s)) => to_dict(s) for s in
  entries))` plus `to_dict(::Clock)`; per-spec `to_dict` for
  `ParquetOptionBars`/`ParquetSpots` (`"type"`, `"dataset" =>
  Dict("root" => root)` — the reserved slot), `QuotesFromBars`
  (`"synthesizer"`), `Constant{RateCurve}`/`Constant{DivCurve}`
  (`"selector"`, `"curve"`), `SurfaceFrom` (`"currency"`, `"spot_for"`
  as a sorted vector of `[from, to]` pairs), `BySelector` (parts sorted
  by `string(selector)`), `InMemory` → error as today. `_core_dict`
  keys: `from, to, data, clock, agent`.
- `src/experiment/config.jl` (transitional): `_experiment_from_cfg`
  still reads the old `[source]` table and maps it to
  `MarketData((ParquetOptionBars(root/options_1min),
  QuotesFromBars(synth), ParquetSpots(root/spots_1min),
  Constant(RateCurve(Currency("USD"), rate)), Constant(DivCurve(u,
  div)), SurfaceFrom(currency=Currency("USD"))))` and
  `Clock{OptionQuote}(Underlying(underlying))`. `max_days_cached` is
  read and ignored (cache sizes are no longer config). This keeps
  `configs/*.toml` and the `.local.toml` valid for gate run #1.
- `src/persistence/store.jl`: drop `_close_experiment_sources` (specs
  hold nothing); `_load_positions` builds
  `Underlying(String(r.underlying))` per row (no `exp.source`
  reach-in).
- `src/experiment/show.jl`: any `exp.source` reference →
  `exp.clock`/`exp.data` summary.
- `src/VolSurfaceAnalysis.jl`: export `kind_name` only if needed by
  scripts (probably not).
- Tests rewritten on in-memory maps (`_MD_` fixture helpers reused from
  `test/market_data/fixtures.jl`): `test/backtest/test_engine.jl`,
  `test/policies/test_policy.jl`, `test/agents/test_agent.jl`,
  `test/experiment/test_experiment.jl` (the `_ExOpenOnceAt` policy
  takes `TimeCut`; fixtures use `with_data`; new assertion: window end
  is the last **clock** tick even when `tick_times` emits later
  candidates), `test/experiment/test_config.jl` (`exp.data isa
  MarketData`, `exp.clock == Clock{OptionQuote}(SPY)`),
  `test/experiment/test_identity.jl` (`to_dict(exp.data)` has the six
  kind keys; `spot_for` order and `BySelector` part order do not change
  `core_hash`; `dataset.root` present), `test/persistence/test_store.jl`
  (`loaded.experiment.data isa MarketData`; the "data absent" test uses
  `open_data` throwing).
- Docs (same commit): `backtest.md` (TimeCut, clock, loop,
  `resolve_quote` by trade underlying), `policies.md` (`decide`
  signature, `DailyShortStrangle` body), `agents.md` (signatures),
  `experiment.md` (struct, window-end rule, `with_data`, failure modes),
  `metrics.md` line 190, `persistence.md` (`load_run` rebuilds specs,
  no lazy validation caveat needed).
- Commit message flags the rule changes (design rule 3): absence
  convention, window-end rule, rate/div cut passthrough removed, clock
  declared.

**Gate run #1** (engine semantics, config unchanged): rerun the baseline
exactly as in step 0 with `--save` (new id because `to_dict` changed),
then `julia --project=. scripts/compare_runs.jl scripts <baseline_id>
<new_id>` must pass. If it fails on `pnl_series` only, first suspect the
window-end rule; on `positions`, suspect `resolve_quote` selector or the
spot lookup.

### Commit 2.2 — `[data.*]` + `clock` config schema, kind table, load-time checks, configs rewritten

Files:
- `src/experiment/config.jl`: `_KINDS = Dict("option_bar" => OptionBar,
  "option_quote" => OptionQuote, "spot_price" => SpotPrice,
  "rate_curve" => RateCurve, "div_curve" => DivCurve, "vol_surface" =>
  VolatilitySurface)` and `kind_name(::Type)` (the only string↔type
  table); `_PROVIDER_BUILDERS = Dict("parquet_option_bars",
  "parquet_spots", "from_bars", "constant", "surface_from",
  "by_selector")` with signature `(d, R)`;
  `_selector_key(::Type{Underlying}) = "underlying"`,
  `(::Type{Currency}) = "currency"`; `_parse_selector(::Type{S}, str)`;
  `constant` builds the record via `_constant_record(::Type{RateCurve},
  sel, curve)` / `DivCurve`, with `value` → `FlatCurve` or `curve =
  {type=...}` → `build_curve`; `by_selector` reads every non-`type` key
  as `selector => sub-table`; `surface_from` reads `currency` and
  `spot_for`. `build_market_data(cfg["data"])` and
  `build_clock(cfg["clock"])`. Load-time checks with clear messages:
  table name known; built spec's `kind` matches the table's kind; no
  duplicate kinds; every `inputs(spec)` kind present;
  `has_lifecycle(spec)`; clock kind present in the map and selector
  type matches. Old `[source]` table → error "use [data.*] and clock
  (see docs/modules/experiment.md)".
- `configs/noop_smoke.toml`, `configs/strangle_spy_16d_1dte.toml`
  rewritten to the new schema (header comment updated; `root` values
  stay the Windows paths as today), and
  `configs/strangle_spy_16d_1dte.local.toml` (untracked) rewritten by
  hand with `/home/ale/data/massive/options_1min` and
  `/home/ale/data/massive/spots_1min`.
- `docs/modules/experiment.md` "Config loading" replaced with the new
  schema and the kind-name table; `docs/modules/market_data.md`
  cross-links; `docs/proposals/data_kinds.md` section 10 gets the
  inference measurement on a config-built map (`@inferred entry`,
  `Base.return_types(open_data, (typeof(exp.data),))` concrete,
  `@allocated entry(...) == 0`).
- Tests `test/experiment/test_config.jl`: each builder from Dicts; each
  load-time check errors (wrong kind under a table name, missing
  `option_bar` for `from_bars`, unknown table name, duplicate selector
  in `by_selector`, clock kind absent); end-to-end TOML →
  `run_experiment` on the parquet smoke tree;
  `test/experiment/test_identity.jl`: whitespace/key order/omitted
  defaults invariance on the new schema; `spot_for` and `by_selector`
  reorderings invariant.

**Gate run #2**: rerun with the rewritten `.local.toml`,
`compare_runs.jl` against the baseline id passes (a third id; only
identity changed).

### Commit 2.3 — manifest `schema_version`, `load_run` refusal

Files:
- `src/persistence/store.jl`: `const RUN_SCHEMA_VERSION = 2`; manifest
  gains `schema_version INTEGER` (outside the hash); `_load_manifest`
  reads it and `load_run` throws `ArgumentError("load_run: run $id was
  written with manifest schema_version $v (pre data-kinds); this store
  reads version $RUN_SCHEMA_VERSION only — rerun the config to
  regenerate it")` when the column is absent or differs.
- `docs/modules/persistence.md`: schema table row, the one-time id
  break, `dataset` slot mention, the "DataSource-shaped" paragraph
  reworded to "mirrors the parquet readers' open/close pair".
- Tests `test/persistence/test_store.jl`: saved manifest has
  `schema_version == 2`; a manifest rewritten via DuckDB without the
  column, and one with `1`, make `load_run` throw with the message;
  round-trip test unchanged.

### Commit 2.4 — point-vs-range benchmark, section 10, status

Files:
- `scripts/bench_point_vs_range.jl [root=~/data/massive] [symbol=SPY]
  [month=2024-01]`: opens `ParquetOptionBars` once; enumerates
  `timestamps` for the month; **A** point: `at` for every timestamp
  (chain LRU cold, `max_chains=10`), **B** range: `for (ts, chain) in
  by_timestamp(between(r, OptionBar, u, first, last))`, **C** the
  strangle workload: `at` at 19:30 for each day vs `between` over each
  day's `[19:30, 19:30]`. Reports wall time (`@elapsed`), allocations,
  `Sys.maxrss()` delta, and asserts record counts of A and B are equal.
  One warm-up pass before timing.
- `docs/proposals/data_kinds.md` section 10: results table (this box: 2
  cores, 3.7 GB, Julia 1.12.7), plus the three gate run ids and their
  `compare_runs.jl` verdicts.
- `docs/status.md`: step 2 complete; the "leaning out docs" entry notes
  `market_data.md` follows the template.

Acceptance: benchmark runs to completion under `ws run` inside memory;
A vs B counts equal.

---

## Step 3 — deletion (2 commits)

### Commit 3.1 — remove the old layer

Deleted: `src/data/source.jl`, `src/data/parquet_source.jl`,
`src/model_data/source.jl`, `src/backtest/time_cut.jl`, the
`src/model_data/` directory, `test/data/test_source.jl`,
`test/data/test_parquet_source.jl`, `test/model_data/test_source.jl`,
`test/backtest/test_time_cut.jl`, `test/model_data/` directory,
`docs/modules/model_data.md`.

Exports removed from `src/VolSurfaceAnalysis.jl`: `DataSource,
InMemoryDataSource, ParquetDataSource, SpotDay, option_path, spot_path,
with_parquet_source, available_timestamps, get_chain, get_spot,
get_spots, clear_cache!, ModelDataSource, get_surface, get_rate,
get_div, TimeCutModelDataSource, build_data_source`. Remaining `using
DuckDB/Tables/OrderedCollections` lines move to
`market_data/parquet.jl`/`lru.jl`.

Scripts ported: `scripts/spot_demo.jl`
(`with_data(MarketData(ParquetSpots(...)))` + `collect(between(...))`),
`scripts/delta_map_demo.jl` and `scripts/surface_slice_demo.jl`
(`with_data` on a map built by `build_market_data` from a small inline
Dict, `only_or_missing(at(d, VolatilitySurface, u, ts))`).

Docs: `docs/modules/data.md` final shape — title "`data` module:
canonical records"; sections: the kinds it defines (`Underlying`,
`OptionType`, `OptionQuote`, `SpotPrice`, `OptionBar`; `Currency` stays
in `market_data`), the visibility-time rule pointer, vendor row mapping
(Polygon ticker parsing, parsed_* precedence, ET→UTC, the bar-open
timestamp allowance), `QuoteSynthesizer`/`SpreadFromOHLCV` as the
synthesis policy consumed by `QuotesFromBars`, key decisions (ticker
mismatch throws; `OptionBar` is vendor-level, policies depend on
`OptionQuote`), and a pointer to `market_data.md` for the protocol.
Remove the `DataSource` protocol, cache, lifecycle and layout sections.
`docs/modules/surfaces.md` line 64, `docs/modules/positions.md` lines
14/33, `docs/modules/persistence.md`, `README.md` references updated.
`docs/modules/market_data.md` loses the "beside the old layer" remarks.

Acceptance: `ws test` green; `grep -rn
'ModelDataSource\|DataSource\|get_chain\|get_spot\|available_timestamps\|TimeCutModelDataSource\|clear_cache!'
src test docs scripts README.md` returns only proposal-appendix history.

### Commit 3.2 — status and proposal closure

`docs/status.md`: "Data" and "Modelling" progress items rewritten around
kinds/providers; in-flight entry removed; backlog gains "dataset
fingerprint in identity (declined in v3, own proposal)",
"capability-restricted views (declined)", "bar-end timestamp convention
as a spec option". `docs/proposals/data_kinds.md` status line:
implemented at `<sha>`; final gate ids recorded. Final baseline
reproduction: rerun once more on this commit with `--save`,
`compare_runs.jl` against the step-0 id passes; record in section 10.

---

## Risks and open decisions (with recommendations)

1. **Bar-time visibility convention.** Recommend the *documented
   allowance*: keep Polygon's bar-open stamp as the visibility time and
   state in 2.1, `data.md` and `market_data.md` that a decision at `t`
   sees the `[t, t+1min)` bar (one minute of lookahead in the
   close/high/low). Reason: shifting to bar end changes every
   timestamp, so the 19:30 entry would read the 19:29 bar and the
   expiry settle would read the 19:59 spot bar — the step-0 gate could
   not pass and the change would not be attributable. A `stamp =
   :bar_end` field on `ParquetOptionBars` (in identity) is the clean
   later addition and goes to the backlog.
2. **LRU implementation.** Recommend the in-repo `OrderedDict`-backed
   `LRU{K,V}` (`market_data/lru.jl`), not LRUCache.jl: the helpers
   already exist and are tested, it is ~25 lines, and adding a
   dependency on a 2-core box with no depot copy costs a
   resolve/precompile cycle for nothing.
3. **`asof` on the parquet bars reader.** Recommend walking the
   per-selector partition `Vector{Date}` backward with the cached
   `PartitionMeta.timestamps` (one `DISTINCT timestamp` query per
   visited day, cached), not a DuckDB `ORDER BY DESC LIMIT 1`: a
   single-file query cannot see other partitions, and a hive-glob query
   would open metadata for ~3000 files per call. Worst case (a `ts`
   before all data) visits every partition once; acceptable because it
   is bounded by the partition list and cached, and `run_experiment`
   calls it once.
4. **`_day_bars` memory.** One SPY day is ~103k rows; as
   `Vector{OptionBar}` (two `String`s per record) roughly 25–40 MB,
   with `Tables.columntable` transiently doubling it. Recommend: one
   query per day with the `BETWEEN` filter, materialize the day, yield
   it from a generator so `Iterators.flatten` holds one day; never
   insert day vectors into the chain LRU (which stays for `at` only, 10
   chains × ~300 rows); reuse the contract-meta `Dict` across days (a
   few thousand entries per symbol). Record the measured `Sys.maxrss()`
   delta in the benchmark. If a day ever exceeds budget, chunk the
   query by hour; not needed now.
5. **`Currency`: new type, not a `Symbol`.** Recommend `struct Currency;
   code::String; end` mirroring `Underlying`. A distinct type makes the
   selector contract enforceable by dispatch (`Clock{RateCurve}(SPY)`
   fails at construction; `BySelector{RateCurve}(SPY => …)` fails), and
   `to_dict`/config need a canonical string anyway.
6. **`selector(r)` definition.** Recommend explicit methods per kind
   (`selector(r::SpotPrice) = r.underlying`) plus
   `selector_type(::Type{R})`. A field-name trait saves nothing (one
   line per kind either way) and hides the contract from
   `@code_warntype`; the two functions together are what `BySelector`,
   `Clock`, `Constant` and the loader check.
7. **Where `Clock` lives.** Recommend `src/market_data/clock.jl`: it is
   a pure value over `(kind, selector)` and its only method is
   `timestamps(m, clock, from, to)`, so step 1 can test it without the
   engine; the engine and `Experiment` consume it in 2.1.
8. **Spot partitions spill past midnight UTC** (found in the data:
   `date=2024-01-16` spots run 09:00 to 00:59 next day). Today's
   `get_spot(ts)` looks only in `Date(ts)`'s partition, so `SpotPrice`
   rows after midnight are unreachable; the strangle never asks for
   them (fills at 19:30, settles at 20:00/21:00 UTC), so the gate is
   unaffected either way. Recommend the reader define the partition
   convention explicitly ("partition `D` may hold timestamps in `[D,
   D+1 02:00)`"), have `at`, `asof`, `between`, `timestamps` all
   consult `Date(ts)-1` and `Date(ts)` through the cached meta, and
   test it (1.5). This is what makes `at == collect(between)` an
   identity rather than a coincidence.
9. **Provenance of the baseline.** The untracked `.local.toml` and the
   pending README/nvim edits make `code_provenance()` report
   `dirty=true`; commit 0a fixes that before the run. Do not skip it —
   the reproducibility harness in the backlog keys on `dirty=false`.
10. **Comparing runs across the identity break.** `load_run` refuses
    the step-0 run after 2.3, so the gate uses
    `scripts/compare_runs.jl` over raw parquet (commit 0a). Keep the
    step-0 folder until 3.2 is done.
11. **Loader `close_data!` presence check.** `hasmethod(close_data!, …)`
    needs the reader type; recommend checking `hasmethod(open_data,
    Tuple{typeof(spec)})` at load time and `hasmethod(close_data!,
    Tuple{only(Base.return_types(open_data, (typeof(spec),)))})`
    guarded by `try` — a loader-only inference query, never on the hot
    path. The lifecycle test suite is the real guarantee.
12. **Recompilation per config shape.** Each distinct `MarketData{P}`
    tuple type compiles once; with one config family this is a few
    seconds on 2 cores. Accept; note in `market_data.md`.
13. **Type stability of `open_data`.** The Appendix B `Any[]` loop is
    not inferable; the recursive `_open_all` in 1.4 is, and it gives
    the unwind for free. Verify with `Base.return_types` in 2.2.
14. **`between` naming vs `Base.between`.** No clash (unexported,
    Integer-only), but never `import Base: between`; the test in 1.1
    pins `length(methods(Base.between)) == 1`.
15. **Experiment field rename `source` → `data`.** It ripples through
    tests, `show.jl`, `store.jl`; do it in 2.1 with `grep -rn
    '\.source\b'` as the checklist rather than keeping a deprecated
    alias.

---

## Commit count and ordering

| # | Step | Commit | Gate |
|---|---|---|---|
| 1 | 0a | Housekeeping (`*.local.toml` ignored, pending docs), `scripts/compare_runs.jl` | clean tree; self-compare passes |
| — | 0 | Baseline run saved to `scripts/runs/run_id=<id0>/` (no commit) | manifest row, `dirty=false` |
| 2 | 0b | Proposal v3.1: A3 fixes folded in, section 10 convention check + baseline record, status | reviewed text |
| 3 | 1.1 | `market_data/`: kinds, `Currency`, selector contract, protocol stubs, library, `market_data.md` | suite green |
| 4 | 1.2 | `InMemory`, `Constant`, `QuotesFromBars`, `MarketData`, `TimeCut`, `Clock`; cut-through-derived test | suite green, `@inferred` |
| 5 | 1.3 | `BySelector` + inference measurement recorded in section 10 | suite green |
| 6 | 1.4 | `open_data`/`close_data!`/`with_data`, unwind tests | suite green |
| 7 | 1.5 | Parquet specs/readers, `LRU`, fixture port, `at == collect(between)`, spill test | suite green; real-day cross-check |
| 8 | 1.6 | Curves moved, `RateCurve`/`DivCurve`, `SurfaceFrom` with bounded cache; status | suite green (step 1 done) |
| 9 | 2.1 | Engine/policies/agents/experiment/identity/store on `MarketData`+`Clock`; transitional `[source]` mapping | gate run #1 reproduces baseline |
| 10 | 2.2 | `[data.*]` + `clock` schema, kind table, load-time checks, configs rewritten; section 10 inference on config-built map | gate run #2 reproduces baseline |
| 11 | 2.3 | Manifest `schema_version`, `load_run` refusal | suite green |
| 12 | 2.4 | `scripts/bench_point_vs_range.jl`, section 10 results, status | benchmark recorded |
| 13 | 3.1 | Delete old layer, exports, tests, `model_data.md`; final `data.md`; scripts ported | suite green; grep clean |
| 14 | 3.2 | Status/proposal closure; final gate run recorded | gate run #3 reproduces baseline |

Fourteen commits: 2 + 6 + 4 + 2. Commits 3–8 and 11–12 are independent
of the real data; 9, 10, 14 each end with a detached full-history run
via `ws run` and a `compare_runs.jl` verdict against `<id0>`.

Critical files for implementation:

- `src/data/parquet_source.jl` (the reader logic to port: day meta,
  chain load, spot blocks, LRU helpers)
- `src/experiment/experiment.jl` (Experiment struct, `run_experiment`,
  window end and `_build_settle` switch)
- `src/experiment/config.jl` (kind-name table, provider builders,
  load-time checks)
- `src/experiment/identity.jl` (`to_dict` projection, `dataset` slot,
  sorted `BySelector`/`spot_for`)
- `src/VolSurfaceAnalysis.jl` (include order and exports across all
  three steps)
