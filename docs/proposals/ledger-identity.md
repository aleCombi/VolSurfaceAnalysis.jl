# PR 3: config and identity, the run-id break

The third pull request of the ledger rebuild
([ledger-orchestration.md](ledger-orchestration.md)). Every value that
changes a backtest result becomes either a declared input in the run id
or a stated constant, and the contract facts the ledger already resolves
join the hash. Stored runs do not survive it, by design.

## Read first

1. `docs/design.md`, all seven rules -- rule 7 (name the unanswerable
   question) decides the `AMSettled` shape below.
2. `docs/modules/experiment.md` and `docs/modules/persistence.md`, for
   identity as it stands.
3. `src/experiment/identity.jl` in full. It is short, and the round is
   mostly additions to it.
4. `src/ledger/contracts.jl`, whose header already promises this work.
5. `docs/status.md`, the in-flight entry: the run-id ambiguity this PR
   closes is described there at the end of the slice 3 paragraph.

## The problem

Four values change backtest results and appear in no run id:
`fill_rule`, `cost_model`, `settlement_rule`, `tick_cents`. They are
keyword defaults on `run_backtest` (`src/backtest/engine.jl:291`) and
`run_experiment` passes none of them -- it calls `run_backtest(exp.agent,
d, exp.from, exp.to, exp.clock)` bare. The venue is therefore a
compile-time constant that identity cannot see, and `run_experiment`'s
docstring says so in as many words ("there is deliberately no keyword
here, since a value that changes results must be in the run id, which
slice 4 arranges").

A fifth omission: `_CONTRACT_TABLE`'s facts reach cash through
`contract_spec` and are in no hash either.

The consequence on disk is the ambiguity `status.md` records: schema 3
was already being written before lifecycle existed, so a stored
pre-lifecycle run loads clean under the same run id as a run of the same
config made today. `load_run` checks the schema number, not which code
produced the run.

## What this round decides

Two of the four turn out not to be choices at all.

**`fill_rule` and `cost_model` are choices.** Nothing about SPY says
whether you cross the spread or whose commissions you pay. They become
config, and they enter `core_hash`.

**Settlement style is a contract fact.** `ContractSpec.settlement`
already records `PMSettled` per underlying and is read by nothing --
today the only field any code reads is `multiplier` (`src/ledger/cash.jl:44`);
`exercise`, `settlement` and `delivery` are declared and inert. A
run-level `settlement_rule` symbol cannot be right for a run holding
both an AM- and a PM-settled contract, and `settlements` passes the same
symbol for every lot. It works today only because SPY, QQQ and IWM are
all `PMSettled`, so the constant cannot disagree with the table. Route
per lot off `contract_spec` instead.

**The settlement price source is a constant, for now.** The stand-in
`:session_close` reads (the last regular-session print of the settlement
session in place of the official closing auction print) is the model's
one stated departure in settlement. Its replacement is a real
official-close feed, and that arrives as a market-data kind with a
provider spec -- a `[data.*]` entry, which is already in identity --
not as a venue symbol. So `:session_close` stays hardcoded as the one
PM rule this round and appears in no config. A constant that cannot vary
needs no hash entry; `commit_sha` covers the code version.

**`tick_cents` is a constant.** `const TICK_CENTS = 1` in
`src/backtest/execution.jl`, used by `run_backtest`. It stays a defaulted
argument on `fill_price`, `fill_legs` and `check_join`: the join check
needs it to recompute a fill from an observation, and
`test/backtest/test_engine.jl:474` deliberately drives a 5-cent tick to
prove the tick is part of the rule. Fixed at the experiment boundary,
parameterised inside. The penny program's price-dependent tick remains a
stated simplification, untouched.

**Two fields, no struct.** With the tick and the settlement source out,
the venue is two symbols. `Experiment` gains `fill_rule::Symbol` and
`cost_model::Symbol` directly; there is no `VenueSpec`. (2026-09-12:
before proposing a struct, ask whether a symbol, a function or an
existing type does the job.)

## Scope

### In

1. `Experiment` gains `fill_rule` and `cost_model`; `run_experiment`
   threads them to `run_backtest`.
2. A `[venue]` table in the config; both keys optional, defaulting to
   today's values.
3. Both into `_core_dict`, plus the resolved `ContractSpec` for the
   experiment's underlying.
4. `settlements` routes on `contract_spec(...).settlement`; `AMSettled`
   is a named failure at load and a named throw at run time.
5. `TICK_CENTS`; `settlement_rule` removed from `run_backtest`.
6. `RUN_SCHEMA_VERSION` 3 -> 4.
7. The `market_data.md` `SpotPrice` correction (below).

### Do not touch

- `src/ledger/` beyond adding `to_dict(::ContractSpec)` and the enum
  projections. No new fields on `ContractSpec`, no tick in the table.
- The `_SETTLEMENT_RULES` table's one entry. **Do not write
  `:session_open`.** No AM-settled underlying exists to test it against,
  and an untested settlement rule is worse than an absent one.
- `pnl_series`, the metrics, the equity curve, the structure series.
  Those are PR 4.
- The named-column parquet writes (backlog), even though this round is
  inside `store.jl` for the version bump. Resist.
- The official-close data kind. That is step 3 of the sequencing below,
  and it needs the collector first.
- Results. **This round must not change a single number.** Only ids.

## The `SpotPrice` correction

`docs/modules/market_data.md:45-60` states that a provider serving
`SpotPrice` serves "regular-session prints only -- no pre-market, no
post-close, no extended-hours session", frames it as a contract on the
tree, and the `:session_close` rule plus a PR #13 review finding both
rest on it.

**The production tree does not satisfy it.** `spots_1min` is Polygon /
Massive `us_stocks_sip/minute_aggs_v1`, and the vendor is explicit that
minute aggregates deliberately relax the SIP sale-condition rules so
extended-hours trades *do* update them ("trades in extended-hours
markets can update OHLC for minute bars, otherwise there would be no
minute aggregates during extended trading hours"). Daily bars are the
opposite -- SIP end-of-day guidelines apply, and extended-hours trades
do not update a daily bar's open or close. Measured on the tree
(2026-09-14): SPY on 2024-12-24 holds bars from 04:00 to 16:59 ET.

**Nothing is wrong today, and that is luck rather than contract.** All
six early closes in the ten-year strangle hold **zero** bars in
(13:00, 16:00] ET -- the after-hours burst begins after 16:00 ET, outside
the rule's window -- so all six still settle at their 13:00 ET print, as
`status.md` records. The exposure the doc describes is real and
currently unrealised.

Rewrite the paragraph to say that: the requirement is what the rule
needs, the tree does not meet it, the rule survives on a measured
property of the data rather than on a guarantee, and the official-close
kind is what will make it structural. This is design rule 7's sibling
and the standing "describe the boundary, never claim impossibility"
decision -- the codebase currently claims a property of its input that
its vendor documents it does not have.

## Files and public surface

| File | Change |
|---|---|
| `src/backtest/execution.jl` | `const TICK_CENTS = 1`. |
| `src/backtest/settlement.jl` | `settlements` routes on `contract_spec(lot.contract.underlying).settlement`; `UnsupportedSettlement`; `settlement_rule` keyword gone. |
| `src/backtest/engine.jl` | `run_backtest` drops `settlement_rule` and `tick_cents`; keeps `fill_rule` / `cost_model` keywords with today's defaults for direct callers. |
| `src/experiment/experiment.jl` | two new `Experiment` fields, kwarg defaults; `run_experiment` threads them; drop the "slice 4 arranges it" paragraph. |
| `src/experiment/config.jl` | `build_venue`; `_experiment_from_cfg` reads an optional `[venue]`; the `PMSettled` load check. |
| `src/experiment/identity.jl` | `to_dict(::ContractSpec)` + enum projections; venue and contract into `_core_dict`. |
| `src/persistence/store.jl` | `RUN_SCHEMA_VERSION = 4` (line 35). |
| `src/VolSurfaceAnalysis.jl` | export `UnsupportedSettlement`, `TICK_CENTS`. |

New public surface:

```julia
Experiment(; name, agent, data, clock, from, to, outputs,
             fill_rule = :cross_spread, cost_model = :ibkr_pro_us_options)

struct UnsupportedSettlement <: Exception      # carries underlying, style
const TICK_CENTS = 1
```

Config:

```toml
[venue]
fill_rule  = "cross_spread"            # optional
cost_model = "ibkr_pro_us_options"     # optional
```

Identity projection, added to `_core_dict`:

```julia
"venue"    => Dict("fill_rule" => "cross_spread",
                   "cost_model" => "ibkr_pro_us_options"),
"contract" => Dict("multiplier" => 100.0, "exercise" => "American",
                   "settlement" => "PMSettled", "delivery" => "Physical"),
```

`"contract"` is the resolved spec for the experiment's underlying, not
the whole table -- one experiment is one underlying (`load_experiment`
asserts it), and projecting the table would fork every id on an
unrelated entry. Guard it: a clock selector that is not an `Underlying`
errors with `run_experiment`'s existing message rather than a
`MethodError` out of `contract_spec`.

## The `AMSettled` shape

Two places, deliberately:

- **At load.** `_experiment_from_cfg`, after the existing
  one-underlying assertion, checks
  `contract_spec(clock.sel).settlement === PMSettled` and errors naming
  the underlying and its style. A config that cannot be run should fail
  when it is read, not four hours into a backtest.
- **At run.** `settlements` throws `UnsupportedSettlement` when it meets
  one. This is *not* caught and warned like `UnpriceableLeg`: that names
  a lot whose price is unavailable now, and design rule 7 says leave it
  open and say so; this names a contract class the codebase cannot settle
  at all, which is a configuration error and must stop the run. The
  distinction is the point -- keep both failures, and say why in the
  comment.

## Tests

Beside the source, one file per source file, as the standing rule has
it. Every failure test checks that it fires, that the ledger is
untouched, and that it prints its name.

**`test/experiment/test_identity.jl`** -- structural, following the file's
existing shape (`load_experiment_str` against nonexistent roots):

- `cost_model = "none"` vs. the default changes `core_hash`, and
  `full_hash` with it.
- An `Experiment` built directly with a different `fill_rule` symbol
  changes `core_hash`. Direct construction, because the loader rejects an
  unknown rule -- identity is computed on an `Experiment`, not on TOML.
- An omitted `[venue]` and an explicitly-default `[venue]` give the same
  hash (the standing omitted-vs-explicit invariant, extended).
- A `ContractSpec` differing in any field forks the id. Build two
  `Experiment`s and project directly rather than mutating the table.
- `name` is still in neither hash.

**`test/experiment/test_config.jl`**:

- `[venue]` round-trips both keys.
- An unknown `fill_rule` / `cost_model` errors naming the known ones
  (the `fill_price` / `commission` message shape).
- An AM-settled underlying fails at load, naming the underlying.

**`test/backtest/test_settlement.jl`**:

- Every existing PM test passes unchanged with the keyword gone.
- A fixture `AMSettled` entry throws `UnsupportedSettlement`, the ledger
  is untouched, the lot stays open, and the message names the underlying.
- `:pre_open_expiry` still fires for a PM contract expiring before
  09:30 ET -- it is no longer standing in for the AM case.

**`test/backtest/test_engine.jl`**:

- The 5-cent tick test at line 474 survives verbatim: bid 0.89, Short,
  `floor(89/5)*5 = 85`, fill 0.85 passes `check_join(L; tick_cents=5)`
  and fails on the default. That test is the reason the parameter stays.
- A `TICK_CENTS` fill is unchanged: ask 0.857, Long,
  `ceil(85.7) = 86`, fill 0.86.

**`test/persistence/test_store.jl`**:

- A manifest at `schema_version = 3` is refused, with the "rerun the
  config to regenerate it" message.

**Regression** -- the round's real gate: the strangle config produces the
same 4478 `Expiry` events over 1699 instants, the same 2240 orders, the
same metrics, under a **new** run id.

## How to run here

Gate (the box has 2 cores and 3.7 GB; check `free -m` first and exit the
`julia` REPL if under about 1.3 GB available, then relaunch it):

```
ws run "JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. -e 'using Pkg; Pkg.test()'"
```

Do not wait on pane text -- a grep for a marker matches the echoed
command line. Wait on the julia process exiting, then `ws capture shell 60`.

The ten-year strangle, for the regression:

```
julia --project=. scripts/run_experiment.jl configs/strangle_spy_16d_1dte.local.toml
```

Not `--save` unless the REPL is down; that needs about 1.7 GB.

## Done means

- No value that changes a result is invisible to the run id: the two
  venue choices and the resolved contract facts are in `core_hash`, and
  the two constants are constants in code rather than defaulted keywords.
- `ContractSpec.settlement` is read by the code rather than by the
  comments.
- A v3 run is refused with the rerun message.
- `market_data.md` states what the tree actually provides.
- The strangle reruns to identical numbers under a new id.
- Gate green.

## After this

The collector gains a `us_stocks_sip/day_aggs_v1` pipeline (a copy of
`download_spot_flatfiles.py` with a new prefix, a `SPOTS_1DAY` dataset,
`prepare_spot_data_for_storage` reused, and a line in `sync_from_r2.sh`);
then an official-close kind and provider spec land here, `:session_close`
reads it instead of walking the minute tree, and the stand-in retires
along with the `SpotPrice` window exposure. That work changes results,
and ids change with them, which is correct. PR 4 is outputs.
