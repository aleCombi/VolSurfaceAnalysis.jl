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
`SpotPrice`. Rows carry Polygon's bar-open timestamp, kept as the
visibility time: a decision at `t` sees the `[t, t+1min)` bar, a
documented one-minute allowance rather than a shift (a bar-end stamp
option is backlog).

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
| **Bar-open stamp kept as visibility time** | Shifting to bar end would move every timestamp (the 19:30 entry would read the 19:29 bar) and break comparability with earlier runs; the allowance is stated instead. |

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
