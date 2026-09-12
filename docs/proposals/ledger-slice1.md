# Slice 1 brief: the `ledger` module

An implementation brief for the first slice of
[ledger.md](ledger.md). The proposal is the design and is binding where
this brief is silent; this brief is the contract for one slice of code.

## Read first

1. [docs/design.md](../design.md), rules 1, 5, 6 and 7.
2. [docs/proposals/ledger.md](ledger.md), sections 2 and 3 in full.
3. [docs/modules/data.md](../modules/data.md), the template a module
   doc follows.
4. `src/positions/trade.jl`, `src/positions/position.jl`: what this
   module supersedes. Do not delete them; the engine still uses them
   until slice 2.
5. `src/metrics/pnl_series.jl`: the `PnLSeries` struct the adapter must
   produce, and the ordering convention it uses.

## Scope

Add, on the current branch, without committing:

- `src/ledger/` (new), included in `src/VolSurfaceAnalysis.jl` after
  `positions/position.jl`.
- `src/metrics/ledger_series.jl` (new), included after
  `metrics/pnl_series.jl`. Metrics depend on the ledger, never the
  reverse.
- `test/ledger/` (new), included in `test/runtests.jl` after the
  positions tests; one file per source file.
- `docs/modules/ledger.md` (new), following the shape of `data.md`,
  with a *Conventions consulted* section carrying the relevant rows of
  the proposal's section 5.
- `docs/status.md`: the in-flight entry says slice 1 landed.
- Exports for the public names listed below.

Touch nothing else. In particular: no change to the engine, policies,
experiment, persistence, or any existing metrics file.

The ledger module may use `Underlying`, `OptionType`, `Call` and `Put`
from `data/quotes.jl` (identity vocabulary). It may not reference
`OptionQuote`, `SpotPrice`, `TimeCut`, or anything from `market_data`.
That is what keeps it testable on hand-built ledgers.

## Files and public surface

### `src/ledger/contracts.jl`

```julia
@enum ExerciseStyle American European
@enum SettlementStyle AMSettled PMSettled
@enum Delivery Physical Cash

struct ContractSpec
    multiplier :: Float64
    exercise   :: ExerciseStyle
    settlement :: SettlementStyle
    delivery   :: Delivery
end

contract_spec(u::Underlying) -> ContractSpec
struct UnknownContract <: Exception  # thrown for any underlying not in the table
```

The table is code, keyed by ticker. Entries: SPY, QQQ, IWM, each
`(100.0, American, PMSettled, Physical)`. Nothing else. A comment
names the source (OCC product specifications).

### `src/ledger/types.jl`

```julia
@enum Side Long Short
@enum Intent Open Close
@enum ExpiryOutcome Worthless CashSettled

struct ContractKey
    underlying  :: Underlying
    strike      :: Float64
    expiry      :: DateTime
    option_type :: OptionType
end

struct Leg
    contract :: ContractKey
    side     :: Side
    quantity :: Int          # > 0, checked in the constructor
    intent   :: Intent
end

struct Order
    label     :: Symbol
    legs      :: Vector{Leg}
    group     :: Union{Nothing,Int}   # nothing => the ledger mints one
    operation :: Union{Nothing,Int}   # links the two orders of a roll
end

struct EventHeader
    id          :: Int
    effective_at:: DateTime
    recorded_at :: DateTime
    sequence    :: Int
end

struct Fill
    header       :: EventHeader
    group        :: Int
    order_leg_id :: Int
    execution_id :: Int
    contract     :: ContractKey
    side         :: Side
    intent       :: Intent
    quantity     :: Int
    price        :: Float64     # per share, > 0
    fill_rule    :: Symbol      # :cross_spread | :broker_execution | ...
end

struct Match
    header        :: EventHeader
    group         :: Int
    open_fill_id  :: Int
    close_fill_id :: Int
    quantity      :: Int
end

struct Expiry
    header           :: EventHeader
    group            :: Int
    open_fill_id     :: Int
    contract         :: ContractKey   # copied from the opening fill
    side             :: Side          # copied from the opening fill
    quantity         :: Int
    settlement_price :: Float64
    outcome          :: ExpiryOutcome
end

struct Fee
    header    :: EventHeader
    source_id :: Int        # the fill that caused it
    amount    :: Float64    # signed cash; a cost is negative
end

const LedgerEvent = Union{Fill, Match, Expiry, Fee}

mutable struct Ledger
    events :: Vector{LedgerEvent}
    # private counters: next id, sequence, group, execution id
end
Ledger() -> Ledger

# accessors, one method per kind where the field differs
header(e), event_id(e), effective_at(e), recorded_at(e), sequence(e)
group(e) -> Union{Nothing,Int}        # nothing for Fee
side_sign(::Side) -> Int              # Long => +1, Short => -1
```

`Trade` is not used here. `ContractKey` is `Trade` minus direction and
quantity.

### `src/ledger/cash.jl`

```julia
intrinsic(c::ContractKey, spot::Real) -> Float64
cash(e::LedgerEvent, spec::ContractSpec) -> Float64
cash(e::LedgerEvent) = cash(e, contract_spec(underlying of e))   # Fee has no contract: amount
```

Exactly the proposal's cash rules:

- `Fill`: `-side_sign(side) * price * quantity * multiplier`
- `Match`: `0.0`
- `Expiry`: `side_sign(side) * intrinsic(contract, settlement_price) * quantity * multiplier`
- `Fee`: `amount`

The two-argument form exists so a test can pin the multiplier without
the table; the one-argument form resolves it from the contract's
underlying.

### `src/ledger/book.jl`

```julia
struct Lot
    group        :: Int
    contract     :: ContractKey
    side         :: Side
    open_fill_id :: Int
    remaining    :: Int
    unit_price   :: Float64
end

mutable struct Book
    lots :: Dict{Tuple{Int,ContractKey}, Vector{Lot}}   # FIFO order within
    cash :: Float64
end
Book() -> Book

apply!(book, e::LedgerEvent, spec) -> book     # one method per kind
open_lots(book) -> Vector{Lot}
lots(book, group::Int) -> Vector{Lot}
open_groups(book) -> Vector{Int}

book_as_known(L, sequence::Int) -> Book   # fold over events with sequence <= boundary
book_effective(L, t::DateTime) -> Book    # fold over events with effective_at <= t,
                                          # ordered by (effective_at, priority, sequence)
```

`apply!` rules: an `Open` fill adds a lot and credits its cash; a
`Close` fill only credits its cash (the matches that follow consume the
lots); a `Match` reduces the named lot's remaining and drops it at zero;
an `Expiry` does the same and credits its cash; a `Fee` credits its
amount. `priority` orders lifecycle before fills at an equal effective
time.

### `src/ledger/append.jl`

The writers the engine will call in slice 2. Each builds the events,
validates the whole batch, appends it as one unit, and applies it to
the book. Nothing is appended if any check fails.

```julia
mint_group!(L) -> Int

record_fill!(L, book, leg::Leg, group::Int;
      price, effective_at, recorded_at, order_leg_id, fill_rule) -> Vector{LedgerEvent}
      # Open: one Fill.  Close: one Fill followed by its Matches, FIFO within
      # (group, contract) among lots of the opposite side.

record_expiry!(L, book, lot::Lot; settlement_price, effective_at, recorded_at) -> Expiry
      # outcome is Worthless when intrinsic is zero, CashSettled otherwise

record_fee!(L, book, source_id::Int, amount; effective_at, recorded_at) -> Fee

commit!(L, book, batch::AbstractVector{<:LedgerEvent}, spec) -> nothing
      # the single validated write path; the three above call it
```

Names avoid `Base.fill!` and `Base.append!`. `Order` shadows the
`Base.Order` module inside `VolSurfaceAnalysis`; nothing in the module
refers to it by name, so that is acceptable, but do not `using
Base.Order` anywhere.

Typed errors, all `<: Exception`, each with a test:
`NothingToClose`, `ExceedsOpen`, `FillAfterExpiry`, `DanglingReference`,
`MatchMismatch` (matches do not exhaust the close, or side/contract
copied onto an `Expiry` differ from its opening fill),
`SequenceGap`, `NonPositiveQuantity`.

Validation at append, from the proposal's invariants: sequence is
contiguous and equals the ledger's next; every reference points to an
earlier event of the right kind; a fill's effective time is at or before
its contract's expiry; the matches immediately following a `Close` fill
reference it and sum to its quantity; consumption never exceeds a lot's
remaining; copied `Expiry` fields equal the opening fill's.

Ids: `id` and `sequence` are separate counters that happen to coincide
in a fresh ledger. Code must never use one for the other.

### `src/ledger/round_trips.jl`

```julia
struct RoundTrip
    group     :: Int
    contract  :: ContractKey
    side      :: Side          # of the opening fill
    quantity  :: Int
    open_id   :: Int
    close_id  :: Int           # the closing Fill or the Expiry
    opened_at :: DateTime
    closed_at :: DateTime      # effective time of the close
    kind      :: Symbol        # :closed | :expired
    pnl       :: Float64
end

round_trips(L, spec = resolved per contract) -> Vector{RoundTrip}
```

One row per `Match` and per `Expiry`, in sequence order. PnL:

- closed: `side_sign(side) * (close.price - open.price) * quantity * multiplier`
- expired: `side_sign(side) * (intrinsic - open.price) * quantity * multiplier`

plus the fee share: for every `Fee` whose source is the opening or the
closing fill, `amount * quantity / that fill's quantity`. When no lot
is left open, the sum of `pnl` over all round trips equals the replayed
`book.cash`; write that reconciliation as a test.

### `src/metrics/ledger_series.jl`

```julia
pnl_series(L::Ledger; unit::Symbol = :structure) -> PnLSeries
```

Produces today's `PnLSeries` so the existing metrics run unchanged on a
ledger. `unit = :structure`: one sample per `(group, closed_at)`, pnl
summed. `unit = :leg`: one sample per round trip. `timestamps` are
`closed_at`. `n_opens` and `n_closes` count `Open` and `Close` fills.
`window_end_spot` is `NaN` and `n_unmarked` is `0`, with a comment
saying both fields leave in slice 5. Order by `(timestamp, pnl)`
exactly as `pnl_series(positions)` does.

## Tests

`test/ledger/` with hand-computed expected numbers written as literals,
not derived in the test:

1. **Full round trip.** Short 1 put at 0.85, buy to close at 0.40,
   multiplier 100: one match, cash `+85 - 40 = 45`, one round trip of
   `+45`, book empty.
2. **Close split across lots.** Lots of 2 and 1 on one contract in one
   group; close 3: two matches of 2 and 1, both lots gone, per-trip pnl
   by lot.
3. **Two groups on one contract.** Groups 1 and 2 each short the same
   call; close group 2 only: group 1 untouched; closing group 1 for
   more than it holds throws `ExceedsOpen`; closing an empty group
   throws `NothingToClose`.
4. **Mixed expiries in one group.** Two lots with different expiries;
   expire one; `book_effective` before and after the settlement
   instant; the other lot still open.
5. **Fees across a partial close.** A fee of `-1.30` on a close of 3
   against lots 2 and 1: shares `-0.8667` and `-0.4333`; trips sum to
   the fee.
6. **Open at window end.** A lot never closed: no round trip for it,
   its fill cash present in `book.cash`, `open_lots` returns it.
7. **Incremental equals replay.** In every case above, the book built
   by `apply!` step by step equals `book_effective(L, far future)`, and
   `book.cash` equals the sum of `cash(e)` over all events.
8. **Known versus true.** An `Expiry` with effective time before its
   recorded time: absent from `book_as_known` at the sequence just
   before it, present in `book_effective` at its effective time.
9. **Every error type** fires, and after a failed batch the ledger's
   length and counters are unchanged.
10. **Contract table.** SPY resolves; an unknown ticker throws
    `UnknownContract`.
11. **Adapter.** `pnl_series(L)` at structure level on case 2 gives one
    sample; at leg level gives two; ordering matches today's rule.

## How to run here

The gate is `Pkg.test()` in the shell window: `ws test`. The
`ws wait` pattern false-matches the echoed command, so wait on the
julia process exiting rather than on output text, then
`ws capture shell 40`. The box has 3.7 GB; check `free -m` first and, if
less than about 1.2 GB is available, say so rather than killing the
julia REPL pane. Set `JULIA_NUM_PRECOMPILE_TASKS=1` if precompilation
is triggered.

## Done means

- The full gate is green; the 1206 existing tests still pass and the
  new ones are added to it.
- `docs/modules/ledger.md` exists and describes boundaries, invariants,
  the cash rules and the conventions consulted; no API walkthrough.
- `docs/status.md` in-flight entry updated.
- No file outside the scope list changed. Nothing committed.
- A short report: what landed, the test count, anything in the
  proposal that had to be interpreted or that turned out wrong.
