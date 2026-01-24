# VolSurfaceAnalysis

[![Build Status](https://github.com/aleCombi/VolSurfaceAnalysis.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/aleCombi/VolSurfaceAnalysis.jl/actions/workflows/CI.yml?query=branch%3Amaster)

A Julia package for analyzing cryptocurrency options volatility surfaces using data from [Deribit](https://www.deribit.com/).

## Data Source

This package consumes options chain data collected from the **Deribit API** via the companion project [DeribitVols](../DeribitVols/).

### Deribit API

Deribit is a cryptocurrency derivatives exchange offering options and futures on BTC and ETH. The data is fetched from the public endpoint:

```
GET /public/get_book_summary_by_currency
```

**API Documentation:** [docs.deribit.com](https://docs.deribit.com/) | [get_book_summary_by_currency](https://docs.deribit.com/api-reference/market-data/public-get_book_summary_by_currency.md)

### Data Fields

The following fields are extracted from the API response and mapped to the `VolRecord` struct:

| API Field | VolRecord Field | Type | Description |
|-----------|-----------------|------|-------------|
| `instrument_name` | `instrument_name` | `String` | Unique identifier, e.g., `BTC-27DEC24-50000-C` |
| *(parsed)* | `underlying` | `Underlying` | Asset: `BTC` or `ETH` |
| *(parsed)* | `expiry` | `DateTime` | Option expiration date (UTC) |
| *(parsed)* | `strike` | `Float64` | Strike price in USD |
| *(parsed)* | `option_type` | `OptionType` | `Call` or `Put` |
| `underlying_price` | `underlying_price` | `Float64` | Spot price of BTC/ETH at observation time |
| `bid_price` | `bid_price` | `Float64?` | Best bid price (`missing` if no bids) |
| `ask_price` | `ask_price` | `Float64?` | Best ask price (`missing` if no asks) |
| `last` | `last_price` | `Float64?` | Last traded price (`missing` if no trades) |
| `mark_price` | `mark_price` | `Float64?` | Theoretical fair value (mid-market) |
| `mark_iv` | `mark_iv` | `Float64?` | **Implied volatility at mark price** (annualized %) |
| `open_interest` | `open_interest` | `Float64?` | Total outstanding contracts |
| `volume` | `volume` | `Float64?` | 24-hour trading volume |
| `creation_timestamp` | `timestamp` | `DateTime` | Observation timestamp |

### Instrument Name Format

The `instrument_name` follows the pattern:

```
{UNDERLYING}-{DD}{MMM}{YY}-{STRIKE}-{C|P}
```

**Example:** `BTC-27DEC24-50000-C`
- **Underlying:** BTC
- **Expiry:** December 27, 2024
- **Strike:** 50,000 USD
- **Type:** Call

### Key Field: `mark_iv`

The `mark_iv` (mark implied volatility) is the primary field for volatility surface construction. It represents:
- **Annualized** implied volatility as a percentage
- Derived from the mark price using Black-Scholes model
- Provided by Deribit's pricing engine

## Data Pipeline

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────────┐
│    Deribit API      │────▶│     DeribitVols      │────▶│   VolSurfaceAnalysis    │
│                     │     │    (Python)          │     │       (Julia)           │
│ /get_book_summary   │     │                      │     │                         │
│ _by_currency        │     │ • Fetches every 60s  │     │ • Reads parquet files   │
│                     │     │ • Stores as parquet  │     │ • Builds vol surfaces   │
│                     │     │ • Cloud sync (S3)    │     │ • Analysis & modeling   │
└─────────────────────┘     └──────────────────────┘     └─────────────────────────┘
```

See [DeribitVols](../DeribitVols/) for data collection configuration and storage options.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/aleCombi/VolSurfaceAnalysis.jl")
```

## Usage

### Reading Data

```julia
using VolSurfaceAnalysis

# Read parquet file from DeribitVols output
records = read_vol_records("path/to/deribit_chain/date=2024-12-27/underlying=BTC/batch_001.parquet")

# Group records by timestamp
by_time = split_by_timestamp(records)

# Or group by rounded timestamp (e.g., hourly)
using Dates
by_hour = split_by_timestamp(records, Hour(1))
```

### Building Volatility Surfaces

```julia
# Build surface from records at a single timestamp
surface = build_surface(records_at_t)

# Access surface data
surface.spot           # Underlying spot price
surface.timestamp      # Observation time
surface.underlying     # BTC or ETH
surface.points         # Vector{VolPoint}

# Each VolPoint contains:
# - log_moneyness: log(K/S)
# - τ: time to expiry in years
# - vol: implied volatility (decimal, e.g., 0.65 for 65%)
```

### Surface Construction Logic

When building a surface, for each (strike, expiry) pair:
1. **ITM preference**: Uses the in-the-money option (Call when S > K, Put when S < K)
2. **Volume tiebreaker**: If both options qualify, picks the one with higher volume
3. **Coordinates**: Converts to (log-moneyness, time-to-expiry) space

## Types

### `VolRecord`

Raw option chain record with all fields from Deribit.

### `VolPoint`

A point on the volatility surface:
- `log_moneyness`: log(K/S) — negative for ITM calls, positive for ITM puts
- `τ`: time to expiry in years (using 365.25 days/year)
- `vol`: implied volatility as decimal

### `VolatilitySurface`

Complete surface at a single timestamp:
- `spot`: underlying price
- `timestamp`: observation time
- `underlying`: BTC or ETH
- `points`: vector of `VolPoint`

### `TermStructure`

A 1D slice of the volatility surface at fixed moneyness (τ → vol):
- `spot`: underlying price
- `timestamp`: observation time
- `underlying`: BTC or ETH
- `moneyness`: log-moneyness level (0.0 for ATM)
- `tenors`: vector of times to expiry (years)
- `vols`: vector of implied volatilities (decimal)

```julia
# Extract ATM term structure from a surface
term_struct = atm_term_structure(surface; atm_threshold=0.05)

# Or directly from records
term_struct = atm_term_structure(records)

# Access data
term_struct.tenors  # [0.01, 0.08, 0.25, ...]  (years)
term_struct.vols    # [0.65, 0.58, 0.52, ...]  (decimal)
```

## Experiments (scratch/)

The `scratch/` folder contains standalone experiments with their own Project.toml that develops the main package:

```bash
cd scratch
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### ATM Term Structure Extraction

Extract ATM vol term structures for each snapshot of a day:

```bash
julia --project=. atm_term_structure_day.jl <data_path> [date] [underlying]

# Examples:
julia --project=. atm_term_structure_day.jl ../data/deribit_chain 2024-12-27 BTC
julia --project=. atm_term_structure_day.jl /path/to/snapshot.parquet
```

## References

- [Deribit API Documentation](https://docs.deribit.com/)
- [DVOL - Deribit Implied Volatility Index](https://insights.deribit.com/exchange-updates/dvol-deribit-implied-volatility-index/)
- [DeribitVols Data Collector](../DeribitVols/) — companion project for data collection
