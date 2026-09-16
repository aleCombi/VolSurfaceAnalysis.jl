module VolSurfaceAnalysis

using Dates

# ── data ────────────────────────────────────────────────────────────────────
# kinds -- what a market datum IS
include("data/kinds/curves.jl")        # Curve payload math; before kinds.jl,
include("data/kinds/kinds.jl")         # because RateCurve carries a Curve

# protocol -- how a datum is ASKED FOR
include("data/protocol/protocol.jl")   # the four shapes and the refusals
include("data/protocol/library.jl")
include("data/protocol/map.jl")
include("data/protocol/by_selector.jl")
include("data/protocol/time_cut.jl")
include("data/protocol/clock.jl")
include("data/protocol/lifecycle.jl")
include("data/protocol/lru.jl")

# providers -- where a datum COMES FROM
include("data/providers/synth.jl")     # before providers.jl (QuotesFromBars
include("data/providers/providers.jl") # is parameterised on a synthesizer)
include("data/providers/massive.jl")
include("data/providers/parquet.jl")   # needs LRU from protocol/

include("surfaces/bs.jl")
include("surfaces/surface.jl")
include("surfaces/build.jl")
include("surfaces/surface_from.jl")
include("ledger/contracts.jl")
include("ledger/types.jl")
include("ledger/cash.jl")
include("ledger/book.jl")
include("ledger/append.jl")
include("ledger/round_trips.jl")
include("policies/policy.jl")
include("policies/daily_short_strangle.jl")
include("agents/agent.jl")
include("backtest/execution.jl")
include("backtest/settlement.jl")
include("backtest/engine.jl")
include("metrics/curve.jl")
include("metrics/marks.jl")
include("metrics/trades.jl")
include("metrics/core.jl")
include("metrics/optional.jl")
include("metrics/dispatch.jl")
include("experiment/experiment.jl")
include("experiment/identity.jl")
include("experiment/show.jl")
include("experiment/config.jl")
include("persistence/provenance.jl")
include("persistence/store.jl")
include("viz/spot.jl")
include("viz/pnl.jl")

export OptionType, Call, Put,
       Underlying, ticker,
       OptionQuote, SpotPrice,
       OptionBar, QuoteSynthesizer, SpreadFromOHLCV, synthesize,
       parse_massive_ticker, et_to_utc,
       Currency, selector, selector_type, snapshot,
       at, between, asof, timestamps, kind,
       only_or_missing, by_timestamp,
       serves, served_description,
       UnservedSelector, ConflictingRecords, DerivationExhausted,
       InMemory, Constant, QuotesFromBars, inputs, demands,
       MarketData, entry, BySelector, TimeCut, Clock,
       open_data, close_data!, with_data,
       ParquetOptionBars, ParquetSpots,
       RateCurve, DivCurve, SurfaceFrom,
       Curve, FlatCurve, PCCurve,
       VolatilitySurface, RawSurface, ExpirySlice,
       expiries, get_slice, iv, price, delta, gamma, vega, forward,
       invert_delta,
       build_surface,
       bs_price, bs_delta, bs_gamma, bs_vega, implied_vol, time_to_expiry,
       ExerciseStyle, American, European,
       SettlementStyle, AMSettled, PMSettled,
       Delivery, Physical, Cash,
       ContractSpec, contract_spec, UnknownContract,
       Side, Long, Short, Intent, Open, Close,
       ExpiryOutcome, Worthless, CashSettled,
       ContractKey, Leg, Order, EventHeader,
       Fill, Match, Expiry, Fee, LedgerEvent, Ledger,
       LegObservation, OrderRecord, last_sequence, order_leg,
       header, event_id, effective_at, recorded_at, sequence, group, side_sign,
       intrinsic, cash,
       Lot, Book, open_lots, lots, open_groups,
       book_as_known, book_effective,
       mint_group!, record_fill!, record_expiry!, record_fee!, record_order!, commit!,
       NothingToClose, ExceedsOpen, FillAfterExpiry, DanglingReference,
       MatchMismatch, SequenceGap, NonPositiveQuantity, NonIntegralCash,
       InvalidPrice, RecordedOutOfOrder, DuplicateExecution,
       RoundTrip, round_trips,
       Policy, NoOpPolicy, DailyShortStrangle, decide, tick_times,
       declared_underlyings,
       Agent, StaticAgent, current_policy,
       fill_price, commission, TICK_CENTS,
       resolve_quote, fill_legs, settlement_price, settlements, session_closes,
       check_join, run_backtest,
       UnpriceableLeg, JoinViolation, UnsupportedSettlement, RunFailure,
       MarkedCurve, marked_curve, mark_price, session_changes,
       n_marked, n_unmarked, cents_to_usd, trade_pnl,
       total_pnl, n_round_trips, hit_rate, n_opens, n_closes,
       sharpe, sortino, max_drawdown, volatility, profit_factor,
       compute_metrics,
       Experiment, ExperimentResult, OutputSpec, run_experiment,
       canonical_failures,
       core_hash, full_hash,
       load_experiment, load_experiment_str,
       kind_name, build_market_data, build_clock,
       build_agent, build_policy, build_curve,
       build_synthesizer, build_output_spec, build_venue,
       RunStore, with_run_store, save_run, load_run, run_dir,
       reproduce, ReproductionReport, Divergence,
       code_provenance, dependency_manifest, MissingManifest

end
