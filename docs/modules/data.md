# `data` module: canonical records

Defines *what* a market datum is: the canonical record types every
consumer depends on, and the mapping from vendor rows to them. *How*
records are obtained -- storage, readers, the time cut -- is the
[`market_data`](market_data.md) module; this one knows no I/O beyond
the row-level mapping.

## The kinds it defines

- `Underlying` -- the selector for market-data kinds; an
  uppercase-normalized ticker with content hash and equality.
- `OptionType` -- `Call` / `Put`.
- `OptionQuote` -- one contract's bid/ask/mark (and optional iv, open
  interest, volume) at a `timestamp`. What policies and the fill path
  depend on.
- `SpotPrice` -- one underlying's price at a `timestamp`.
- `OptionBar` -- a faithful mirror of one Polygon OHLCV minute-bar row
  with the contract identity attached. Vendor-level: addressable as a
  kind, but downstream code depends on `OptionQuote`.

`Currency`, the selector for rate curves, lives in `market_data`.

Every kind's `timestamp` is its **visibility time**, the moment the
record became knowable; the time cut filters on it and on nothing else
(the rule is stated in [`market_data`](market_data.md)). Missing
scalar fields inside a record stay `missing`; an absent record is an
empty result from the protocol, never `nothing` or `missing`.

## Vendor row mapping (Polygon)

Polygon option rows are normalized into `OptionBar`. Contract identity
comes from the collector's `parsed_*` columns when present, with ticker
parsing (`parse_polygon_ticker`) as the fallback; expiries are stamped
at 16:00 ET converted to UTC (`et_to_utc`). Spot rows map directly to
`SpotPrice`.

**Bar-end visibility is the invariant.** A vendor minute bar is stamped
at its OPEN, but its close, high and low -- and every spread synthesized
from them -- are knowable only when the minute has finished. A record
read off such a row is therefore stamped `row timestamp + BAR_INTERVAL`,
one minute for both production trees: the 19:29 row is visible at 19:30,
and a decision at 19:30 reads the completed 19:29-19:30 minute.
`bar_visible_at` / `bar_row_time` are that mapping and its inverse; the
readers in [`market_data`](market_data.md) apply them at the one boundary
where rows become records, and every shape above it speaks visibility
time. Synthesis preserves the instant it is handed and adds nothing.

This is **the** convention, fixed in code. It is not a spec option and
not a config key: one of the two settings would enable lookahead, so
offering both would let an experiment choose an incorrect clock. A feed
whose bars are not one minute needs its own reader stating its own
interval -- the general "declare your visibility convention" rule -- but a
completed minute's availability is not a choice.

Malformed tickers throw. A row whose underlying is not the partition's
throws as well: under `symbol=` partitioning that is a corrupt store,
not a row to skip.

## Quote synthesis

Polygon has only OHLCV, no bid/ask. A `QuoteSynthesizer` turns an
`OptionBar` into an `OptionQuote`; the concrete policy today is
`SpreadFromOHLCV(lambda)`, which interpolates bid and ask between the
bar's range and its close (`lambda = 1` is the midpoint, `0` the full
range) and keeps the close as the mark. Missing high/low/close yields
missing bid/ask rather than an invented market. The synthesizer is
consumed by the `QuotesFromBars` derived provider in `market_data`, so
the bid/ask construction is part of an experiment's identity and a
future feed that has quotes simply does not configure it.

## Key decisions

| Decision | Why |
|---|---|
| **`OptionBar` is a first-class, vendor-level kind** | Keeps the synthesis policy explicit and testable instead of buried in a reader; policies still depend on `OptionQuote`, so a live feed with real quotes needs no adapter. |
| **Synthesizer is declared, not defaulted** | Bid/ask construction is part of provenance; `lambda` is required at the type level so the fill policy is always visible in the experiment record. |
| **`Underlying` hashes by content** | The default struct hash falls back to `objectid`, which for a type in a precompiled package changes with every build, so a `Dict` keyed on `Underlying` iterated in a build-dependent order (surfaced as a nondeterministic PnL series order). Explicit `hash` / `==` on the ticker make such dictionaries deterministic; `Currency` in `market_data` does the same. |
| **Ticker-underlying mismatch throws** | Path partitioning makes a foreign ticker a data-corruption signal. Silent skipping would hide bugs. |
| **Bar end is the visibility time, as a constant and not a setting** | Every value read off a minute bar is knowable only when the minute ends, so a bar-open stamp hands each fill and settlement price up to a minute of future information -- below the time cut, where `TimeCut` cannot see it. Making it configurable would keep the incorrect clock reachable. The cost is paid once and recorded in status: every stamp moves, `core_hash` and every run id move with it, and stored runs stop reproducing. |

## Layout

```
src/data/
    quotes.jl     # Underlying, OptionType, OptionQuote, SpotPrice
    polygon.jl    # ticker parsing, ET -> UTC, ContractMeta
    synth.jl      # OptionBar, QuoteSynthesizer, SpreadFromOHLCV
test/data/
```

Files are `include`d into the top-level `VolSurfaceAnalysis` module;
no submodule wrappers. The protocol, specs and readers are documented
in [`market_data`](market_data.md).
