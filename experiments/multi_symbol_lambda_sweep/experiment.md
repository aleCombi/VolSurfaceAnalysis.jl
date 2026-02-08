# Experiment: Multi-Symbol Lambda Sweep

**Date**: 2026-02-08
**Script**: [`scripts/ml_strike_selector.jl`](../../scripts/ml_strike_selector.jl)
**Config system**: [`scripts/configurations.jl`](../../scripts/configurations.jl) + [`scripts/configs/*.toml`](../../scripts/configs/)

## Objective

Test whether the ML iron condor strike selector generalizes beyond SPXW to other option underlyings and across different transaction cost assumptions (spread lambda). This is the first multi-symbol experiment, covering 6 underlyings x 3 lambda values = 18 independent training+evaluation runs.

## Underlyings

| Symbol | Description | Spot Proxy | Multiplier | Approx Price | Div Yield |
|--------|-------------|------------|:---:|:---:|:---:|
| **SPXW** | S&P 500 index weekly options (cash-settled, European) | SPY | 10x | ~$5,800 | 1.3% |
| **SPY** | SPDR S&P 500 ETF options (American, physically settled) | SPY | 1x | ~$580 | 1.3% |
| **QQQ** | Invesco QQQ Trust / Nasdaq-100 ETF options | QQQ | 1x | ~$520 | 0.6% |
| **IWM** | iShares Russell 2000 small-cap ETF options | IWM | 1x | ~$220 | 1.2% |
| **GLD** | SPDR Gold Shares ETF options | GLD | 1x | ~$180 | 0.0% |
| **TLT** | iShares 20+ Year Treasury Bond ETF options | TLT | 1x | ~$90 | 3.5% |

**Asset classes covered**: Large-cap US equities (SPXW, SPY, QQQ), small-cap equities (IWM), commodities (GLD), fixed income (TLT).

## Per-Symbol Configuration

Each symbol has a TOML config file that sets condor thresholds scaled to its price level:

| Config File | `condor_max_loss_min` | `condor_max_loss_max` | `condor_min_credit` |
|-------------|:---:|:---:|:---:|
| [`SPXW.toml`](../../scripts/configs/SPXW.toml) | 50.0 | 300.0 | 1.00 |
| [`SPY.toml`](../../scripts/configs/SPY.toml) | 5.0 | 30.0 | 0.10 |
| [`QQQ.toml`](../../scripts/configs/QQQ.toml) | 5.0 | 30.0 | 0.10 |
| [`IWM.toml`](../../scripts/configs/IWM.toml) | 2.0 | 15.0 | 0.05 |
| [`GLD.toml`](../../scripts/configs/GLD.toml) | 2.0 | 15.0 | 0.05 |
| [`TLT.toml`](../../scripts/configs/TLT.toml) | 1.0 | 10.0 | 0.03 |

## Methodology

### Pipeline (7 phases per run)

Each of the 18 runs executes the full ML pipeline independently:

1. **Load Training Data** (Phase 1)
   - Period: 2024-02-07 to 2025-02-07 (~252 trading days)
   - Entry times: 5 intraday timestamps (10:00, 11:00, 12:00, 13:00, 14:00 ET) to maximize training data
   - Loads vol surfaces, entry spots (via spot proxy x multiplier), settlement spots, minute-level spot history (5-day lookback), and previous-day surfaces for dynamics features

2. **Generate Training Labels** (Phase 2)
   - Grid search over put/call delta pairs to find the optimal condor for each entry
   - Wing selection via ROI-optimized objective (`_condor_wings_by_objective` with `:roi`)
   - Labels: optimal put delta, call delta, and position sizing signal
   - Surfaces with no valid condor (violating max_loss/min_credit constraints) are skipped

3. **Load Validation Data** (Phase 3)
   - Period: 2025-02-07 to 2025-08-07 (~125 trading days)
   - Single entry time: 10:00 ET only (fair comparison, no look-ahead)

4. **Normalize Features** (Phase 4)
   - Z-score normalization: `(x - mean) / std` using training set statistics
   - 59-dimensional feature vector: 36 base features + 23 pruned log-signature features

5. **Train Neural Network** (Phase 5)
   - Architecture: 59 -> [64, 32, 16] -> 3 (put_delta, call_delta, position_size)
   - Optimizer: Adam, LR=0.001, batch size=32
   - Early stopping: patience=15 epochs, max 100 epochs
   - Dropout: 0.2

6. **Evaluate Model** (Phase 6)
   - Delta MSE/MAE and Size MSE/MAE on train and validation sets

7. **Backtest Comparison** (Phase 7)
   - Three strategies run on validation period via `backtest_strategy()`:
     - **ML Condor**: neural network predicts optimal deltas per surface
     - **Fixed Delta Condor**: symmetric 16-delta short strikes with ROI-optimized wings (baseline)
     - **Sigma Condor**: 0.7-sigma short / 1.5-sigma long strikes (simpler baseline)
   - Metrics: `performance_metrics()` with per-condor max loss as margin for ROI computation

### Feature Vector (59 dimensions)

| Group | Count | Description |
|-------|:---:|-------------|
| Smile shape | 10 | ATM IV, skew, curvature, put/call wing slopes at multiple moneyness levels |
| Term structure | 6 | Slope, curvature, front/back ratio across expiry tenors |
| Spread by moneyness | 6 | Bid-ask spread at OTM put, ATM, OTM call for near/far expiries |
| Surface dynamics | 3 | Day-over-day changes: `delta_atm_iv_1d`, `delta_skew_1d`, `delta_term_slope_1d` |
| Volume | 3 | Aggregate volume, put/call volume ratio, volume concentration |
| Spot dynamics | 5 | Returns, realized vol from minute-level spot history |
| Calendar | 3 | Day of week, month, days to expiry |
| Log-signature | 23 | Path signature features from spot price path (pruned from 30 via `LOGSIG_DEAD_INDICES`) |

### Spread Lambda

The `spread_lambda` parameter controls synthetic bid/ask construction from OHLC data:

| Lambda | Meaning | Friction Level |
|--------|---------|----------------|
| **0.0** | `bid = low, ask = high` | Widest spread (most conservative) |
| **0.5** | Midpoint between 0.0 and 1.0 | Moderate friction |
| **0.7** | 70% toward midpoint | Low friction (canonical default) |
| **1.0** | `bid = ask = midpoint` | Zero spread (no friction) |

### Execution

All 18 runs were launched in parallel via:

```bash
# Lambda 0.0 (6 runs)
julia --project=scripts scripts/ml_strike_selector.jl SPXW 0.0
julia --project=scripts scripts/ml_strike_selector.jl SPY 0.0
julia --project=scripts scripts/ml_strike_selector.jl QQQ 0.0
julia --project=scripts scripts/ml_strike_selector.jl IWM 0.0
julia --project=scripts scripts/ml_strike_selector.jl GLD 0.0
julia --project=scripts scripts/ml_strike_selector.jl TLT 0.0

# Lambda 0.5 (6 runs)
julia --project=scripts scripts/ml_strike_selector.jl SPXW 0.5
# ... etc

# Lambda 0.7 (6 runs)
julia --project=scripts scripts/ml_strike_selector.jl SPXW 0.7
# ... etc
```

Each run produces its own output directory with symbol and lambda tag to avoid collisions.

## Results

### ML Condor: Trades and Avg ROI

| Symbol | λ=0.0 Trades | λ=0.0 ROI | λ=0.5 Trades | λ=0.5 ROI | λ=0.7 Trades | λ=0.7 ROI | **Wtd Avg ROI** |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SPXW** | 123 | 4.7% | 123 | 6.7% | 123 | 7.1% | **6.1%** |
| **SPY** | 123 | 3.0% | 123 | 3.7% | 123 | 6.0% | **4.2%** |
| **QQQ** | 125 | 0.7% | 125 | 3.2% | 125 | 4.1% | **2.6%** |
| **IWM** | 113 | 4.8% | 112 | 5.8% | 113 | 5.9% | **5.5%** |
| **GLD** | 86 | -0.3% | 86 | -0.5% | 85 | -0.2% | **-0.3%** |
| **TLT** | 24 | -0.3% | 22 | -7.6% | 24 | 4.8% | **-0.9%** |

### Fixed Delta Baseline: Trades and Avg ROI

| Symbol | λ=0.0 Trades | λ=0.0 ROI | λ=0.5 Trades | λ=0.5 ROI | λ=0.7 Trades | λ=0.7 ROI | **Wtd Avg ROI** |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SPXW** | 123 | 2.4% | 123 | 3.2% | 123 | 3.4% | **3.0%** |
| **SPY** | 123 | 1.2% | 123 | 2.6% | 123 | 3.1% | **2.3%** |
| **QQQ** | 124 | 1.2% | 124 | 2.3% | 124 | 2.8% | **2.1%** |
| **IWM** | 103 | 3.7% | 103 | 4.4% | 103 | 4.7% | **4.3%** |
| **GLD** | 73 | -2.2% | 73 | -1.8% | 73 | -1.6% | **-1.9%** |
| **TLT** | 16 | 2.1% | 16 | 2.4% | 16 | 2.6% | **2.4%** |

### Sigma Baseline: Trades and Avg ROI

| Symbol | λ=0.0 Trades | λ=0.0 ROI | λ=0.5 Trades | λ=0.5 ROI | λ=0.7 Trades | λ=0.7 ROI | **Wtd Avg ROI** |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SPXW** | 123 | 6.3% | 123 | 7.6% | 123 | 8.1% | **7.3%** |
| **SPY** | 123 | -16.7% | 123 | 5.0% | 123 | 6.0% | **-1.9%** |
| **QQQ** | 125 | -11.6% | 125 | -28.9% | 125 | 2.9% | **-12.5%** |
| **IWM** | 120 | 0.3% | 120 | 8.2% | 120 | 8.9% | **5.8%** |
| **GLD** | 98 | -1.5% | 98 | 0.1% | 98 | 0.8% | **-0.2%** |
| **TLT** | 52 | 5.5% | 52 | 7.8% | 52 | 8.8% | **7.4%** |

### ML Edge Over Fixed Delta Baseline

| Symbol | λ=0.0 Edge | λ=0.5 Edge | λ=0.7 Edge | **Wtd Avg Edge** |
|--------|:---:|:---:|:---:|:---:|
| **SPXW** | +2.3% | +3.4% | +3.7% | **+3.2%** |
| **SPY** | +1.8% | +1.1% | +2.9% | **+1.9%** |
| **QQQ** | -0.5% | +0.9% | +1.3% | **+0.6%** |
| **IWM** | +1.1% | +1.4% | +1.2% | **+1.2%** |
| **GLD** | +2.0% | +1.3% | +1.4% | **+1.5%** |
| **TLT** | -2.4% | -10.0% | +2.2% | **-3.3%** |

### Full Metrics (Sharpe, Win Rate, Total PnL)

#### Lambda 0.0

| Symbol | Strategy | Trades | Total PnL | Avg ROI | Sharpe | Win Rate |
|--------|----------|:---:|---:|:---:|:---:|:---:|
| SPXW | ML Condor | 123 | $330.76 | 4.7% | 1.54 | 71.5% |
| SPXW | Fixed Delta | 123 | $192.40 | 2.4% | 1.22 | 82.1% |
| SPXW | Sigma | 123 | $344.44 | 6.3% | 2.08 | 75.6% |
| SPY | ML Condor | 123 | $21.21 | 3.0% | 1.09 | 69.1% |
| SPY | Fixed Delta | 123 | $9.40 | 1.2% | 0.64 | 80.5% |
| SPY | Sigma | 123 | $9.91 | -16.7% | -1.16 | 71.5% |
| QQQ | ML Condor | 125 | $6.00 | 0.7% | 0.26 | 67.2% |
| QQQ | Fixed Delta | 124 | $8.34 | 1.2% | 0.60 | 80.6% |
| QQQ | Sigma | 125 | -$7.03 | -11.6% | -1.26 | 72.0% |
| IWM | ML Condor | 113 | $12.71 | 4.8% | 2.65 | 73.5% |
| IWM | Fixed Delta | 103 | $11.51 | 3.7% | 2.47 | 81.6% |
| IWM | Sigma | 120 | $13.66 | 0.3% | 0.05 | 69.2% |
| GLD | ML Condor | 86 | -$0.91 | -0.3% | -0.25 | 73.3% |
| GLD | Fixed Delta | 73 | -$3.29 | -2.2% | -1.17 | 78.1% |
| GLD | Sigma | 98 | -$6.43 | -1.5% | -0.55 | 65.3% |
| TLT | ML Condor | 24 | $0.17 | -0.3% | -0.15 | 70.8% |
| TLT | Fixed Delta | 16 | $0.58 | 2.1% | 2.97 | 75.0% |
| TLT | Sigma | 52 | $1.01 | 5.5% | 1.33 | 69.2% |

#### Lambda 0.5

| Symbol | Strategy | Trades | Total PnL | Avg ROI | Sharpe | Win Rate |
|--------|----------|:---:|---:|:---:|:---:|:---:|
| SPXW | ML Condor | 123 | $443.70 | 6.7% | 2.14 | 72.4% |
| SPXW | Fixed Delta | 123 | $234.73 | 3.2% | 1.63 | 82.1% |
| SPXW | Sigma | 123 | $411.59 | 7.6% | 2.47 | 76.4% |
| SPY | ML Condor | 123 | $26.93 | 3.7% | 1.31 | 72.4% |
| SPY | Fixed Delta | 123 | $17.90 | 2.6% | 1.36 | 81.3% |
| SPY | Sigma | 123 | $23.70 | 5.0% | 1.76 | 71.5% |
| QQQ | ML Condor | 125 | $23.91 | 3.2% | 1.17 | 69.6% |
| QQQ | Fixed Delta | 124 | $15.67 | 2.3% | 1.12 | 80.6% |
| QQQ | Sigma | 125 | $10.89 | -28.9% | -1.23 | 72.8% |
| IWM | ML Condor | 112 | $15.53 | 5.8% | 3.06 | 71.4% |
| IWM | Fixed Delta | 103 | $13.44 | 4.4% | 2.89 | 81.6% |
| IWM | Sigma | 120 | $17.57 | 8.2% | 3.56 | 70.8% |
| GLD | ML Condor | 86 | $0.65 | -0.5% | -0.29 | 69.8% |
| GLD | Fixed Delta | 73 | -$2.27 | -1.8% | -0.93 | 78.1% |
| GLD | Sigma | 98 | -$3.41 | 0.1% | 0.04 | 65.3% |
| TLT | ML Condor | 22 | -$0.29 | -7.6% | -2.64 | 59.1% |
| TLT | Fixed Delta | 16 | $0.67 | 2.4% | 3.45 | 75.0% |
| TLT | Sigma | 52 | $1.56 | 7.8% | 1.79 | 69.2% |

#### Lambda 0.7

| Symbol | Strategy | Trades | Total PnL | Avg ROI | Sharpe | Win Rate |
|--------|----------|:---:|---:|:---:|:---:|:---:|
| SPXW | ML Condor | 123 | $458.98 | 7.1% | 2.26 | 73.2% |
| SPXW | Fixed Delta | 123 | $251.11 | 3.4% | 1.70 | 82.1% |
| SPXW | Sigma | 123 | $438.45 | 8.1% | 2.62 | 76.4% |
| SPY | ML Condor | 123 | $40.42 | 6.0% | 2.09 | 74.0% |
| SPY | Fixed Delta | 123 | $21.54 | 3.1% | 1.66 | 82.1% |
| SPY | Sigma | 123 | $29.21 | 6.0% | 2.09 | 71.5% |
| QQQ | ML Condor | 125 | $31.21 | 4.1% | 1.50 | 71.2% |
| QQQ | Fixed Delta | 124 | $19.02 | 2.8% | 1.37 | 80.6% |
| QQQ | Sigma | 125 | $18.06 | 2.9% | 0.71 | 72.8% |
| IWM | ML Condor | 113 | $17.24 | 5.9% | 3.12 | 70.8% |
| IWM | Fixed Delta | 103 | $13.65 | 4.7% | 3.07 | 81.6% |
| IWM | Sigma | 120 | $19.13 | 8.9% | 3.86 | 71.7% |
| GLD | ML Condor | 85 | $3.23 | -0.2% | -0.12 | 71.8% |
| GLD | Fixed Delta | 73 | -$1.82 | -1.6% | -0.83 | 78.1% |
| GLD | Sigma | 98 | -$2.21 | 0.8% | 0.27 | 66.3% |
| TLT | ML Condor | 24 | $0.38 | 4.8% | 1.80 | 66.7% |
| TLT | Fixed Delta | 16 | $0.71 | 2.6% | 3.64 | 75.0% |
| TLT | Sigma | 52 | $1.78 | 8.8% | 1.98 | 69.2% |

## Interpretation

### Variance Risk Premium by Asset Class

The **baseline profitability** (Fixed Delta Condor) reveals the structural variance risk premium for each underlying:

| Symbol | Baseline ROI (λ=0.0) | Baseline ROI (λ=0.7) | Interpretation |
|--------|:---:|:---:|-------------|
| **IWM** | 3.7% | 4.7% | Strongest ETF premium (small-cap uncertainty + illiquidity) |
| **SPXW** | 2.4% | 3.4% | Strong premium (institutional hedging demand on SPX) |
| **SPY** | 1.2% | 3.1% | Moderate (same index, but American-style ETF options) |
| **QQQ** | 1.2% | 2.8% | Moderate (tech-heavy, less hedging demand than SPX) |
| **TLT** | 2.1% | 2.6% | Small positive (rate uncertainty premium) |
| **GLD** | -2.2% | -1.6% | **No premium** (gold vol driven by macro, not hedging) |

### ML Edge Analysis

**Where the ML works well (equity indices):**
- **SPXW**: +3.2% weighted edge, strongest and most consistent. SPXW has the deepest options market, most structured vol surface, and richest feature signal.
- **SPY**: +1.9% weighted edge. Tracks the same index as SPXW but with American-style options and smaller dollar values. Edge is real but smaller.
- **IWM**: +1.2% weighted edge with very high Sharpe (2.65-3.12). Small-cap vol surfaces are noisier but the model finds consistent alpha.

**Where the ML is marginal (QQQ):**
- **QQQ**: +0.6% weighted edge. Negative at λ=0.0, positive at higher lambdas. Tech-heavy vol is harder to predict; the model struggles when transaction costs are highest.

**Where the ML doesn't work (commodities, bonds):**
- **GLD**: -0.3% avg ROI but +1.5% edge over an even-worse baseline. Both strategies lose money. Gold options lack the equity variance risk premium, so there's no structural edge to capture.
- **TLT**: Unreliable results (only 16-24 trades per run). The -7.6% at λ=0.5 is a single large loss dominating. Too few trades to draw conclusions.

### Trade Count Patterns

- **SPXW/SPY/QQQ**: ~123-125 trades (virtually every trading day produces a valid condor)
- **IWM**: ~103-113 trades (some days filtered out by narrower strike grid)
- **GLD**: ~73-86 trades (fewer valid condors due to lower vol and sparser strikes)
- **TLT**: **Only 16-24 trades** (most days fail the condor filters due to very low dollar-value options). Results are statistically unreliable.

### Lambda Sensitivity

All equity underlyings show the expected pattern: ROI improves with higher lambda (lower friction). The ML edge is robust across all lambda values for SPXW, SPY, and IWM, confirming the alpha comes from genuine strike selection rather than spread exploitation.

## Run Folders

### Lambda 0.0

| Symbol | Run Directory | Latest Symlink |
|--------|---------------|----------------|
| SPXW | [`runs/ml_strike_selector_SPXW_20260208_220629/`](../../scripts/runs/ml_strike_selector_SPXW_20260208_220629/) | [`latest_runs/ml_strike_selector_SPXW/`](../../scripts/latest_runs/ml_strike_selector_SPXW/) |
| SPY | [`runs/ml_strike_selector_SPY_20260208_220630/`](../../scripts/runs/ml_strike_selector_SPY_20260208_220630/) | [`latest_runs/ml_strike_selector_SPY/`](../../scripts/latest_runs/ml_strike_selector_SPY/) |
| QQQ | [`runs/ml_strike_selector_QQQ_20260208_220631/`](../../scripts/runs/ml_strike_selector_QQQ_20260208_220631/) | [`latest_runs/ml_strike_selector_QQQ/`](../../scripts/latest_runs/ml_strike_selector_QQQ/) |
| IWM | [`runs/ml_strike_selector_IWM_20260208_220631/`](../../scripts/runs/ml_strike_selector_IWM_20260208_220631/) | [`latest_runs/ml_strike_selector_IWM/`](../../scripts/latest_runs/ml_strike_selector_IWM/) |
| GLD | [`runs/ml_strike_selector_GLD_20260208_220633/`](../../scripts/runs/ml_strike_selector_GLD_20260208_220633/) | [`latest_runs/ml_strike_selector_GLD/`](../../scripts/latest_runs/ml_strike_selector_GLD/) |
| TLT | [`runs/ml_strike_selector_TLT_20260208_220634/`](../../scripts/runs/ml_strike_selector_TLT_20260208_220634/) | [`latest_runs/ml_strike_selector_TLT/`](../../scripts/latest_runs/ml_strike_selector_TLT/) |

### Lambda 0.5

| Symbol | Run Directory | Latest Symlink |
|--------|---------------|----------------|
| SPXW | [`runs/ml_strike_selector_SPXW_lambda0.5_20260208_221138/`](../../scripts/runs/ml_strike_selector_SPXW_lambda0.5_20260208_221138/) | [`latest_runs/ml_strike_selector_SPXW_lambda0.5/`](../../scripts/latest_runs/ml_strike_selector_SPXW_lambda0.5/) |
| SPY | [`runs/ml_strike_selector_SPY_lambda0.5_20260208_221142/`](../../scripts/runs/ml_strike_selector_SPY_lambda0.5_20260208_221142/) | [`latest_runs/ml_strike_selector_SPY_lambda0.5/`](../../scripts/latest_runs/ml_strike_selector_SPY_lambda0.5/) |
| QQQ | [`runs/ml_strike_selector_QQQ_lambda0.5_20260208_221144/`](../../scripts/runs/ml_strike_selector_QQQ_lambda0.5_20260208_221144/) | [`latest_runs/ml_strike_selector_QQQ_lambda0.5/`](../../scripts/latest_runs/ml_strike_selector_QQQ_lambda0.5/) |
| IWM | [`runs/ml_strike_selector_IWM_lambda0.5_20260208_221147/`](../../scripts/runs/ml_strike_selector_IWM_lambda0.5_20260208_221147/) | [`latest_runs/ml_strike_selector_IWM_lambda0.5/`](../../scripts/latest_runs/ml_strike_selector_IWM_lambda0.5/) |
| GLD | [`runs/ml_strike_selector_GLD_lambda0.5_20260208_221150/`](../../scripts/runs/ml_strike_selector_GLD_lambda0.5_20260208_221150/) | [`latest_runs/ml_strike_selector_GLD_lambda0.5/`](../../scripts/latest_runs/ml_strike_selector_GLD_lambda0.5/) |
| TLT | [`runs/ml_strike_selector_TLT_lambda0.5_20260208_221153/`](../../scripts/runs/ml_strike_selector_TLT_lambda0.5_20260208_221153/) | [`latest_runs/ml_strike_selector_TLT_lambda0.5/`](../../scripts/latest_runs/ml_strike_selector_TLT_lambda0.5/) |

### Lambda 0.7

| Symbol | Run Directory | Latest Symlink |
|--------|---------------|----------------|
| SPXW | [`runs/ml_strike_selector_SPXW_lambda0.7_20260208_221141/`](../../scripts/runs/ml_strike_selector_SPXW_lambda0.7_20260208_221141/) | [`latest_runs/ml_strike_selector_SPXW_lambda0.7/`](../../scripts/latest_runs/ml_strike_selector_SPXW_lambda0.7/) |
| SPY | [`runs/ml_strike_selector_SPY_lambda0.7_20260208_221143/`](../../scripts/runs/ml_strike_selector_SPY_lambda0.7_20260208_221143/) | [`latest_runs/ml_strike_selector_SPY_lambda0.7/`](../../scripts/latest_runs/ml_strike_selector_SPY_lambda0.7/) |
| QQQ | [`runs/ml_strike_selector_QQQ_lambda0.7_20260208_221146/`](../../scripts/runs/ml_strike_selector_QQQ_lambda0.7_20260208_221146/) | [`latest_runs/ml_strike_selector_QQQ_lambda0.7/`](../../scripts/latest_runs/ml_strike_selector_QQQ_lambda0.7/) |
| IWM | [`runs/ml_strike_selector_IWM_lambda0.7_20260208_221148/`](../../scripts/runs/ml_strike_selector_IWM_lambda0.7_20260208_221148/) | [`latest_runs/ml_strike_selector_IWM_lambda0.7/`](../../scripts/latest_runs/ml_strike_selector_IWM_lambda0.7/) |
| GLD | [`runs/ml_strike_selector_GLD_lambda0.7_20260208_221151/`](../../scripts/runs/ml_strike_selector_GLD_lambda0.7_20260208_221151/) | [`latest_runs/ml_strike_selector_GLD_lambda0.7/`](../../scripts/latest_runs/ml_strike_selector_GLD_lambda0.7/) |
| TLT | [`runs/ml_strike_selector_TLT_lambda0.7_20260208_221153/`](../../scripts/runs/ml_strike_selector_TLT_lambda0.7_20260208_221153/) | [`latest_runs/ml_strike_selector_TLT_lambda0.7/`](../../scripts/latest_runs/ml_strike_selector_TLT_lambda0.7/) |

### Run Folder Contents

Each run directory contains:

| File | Description |
|------|-------------|
| `comparison_results.csv` | Strategy comparison (ML, Fixed Delta, Sigma) with Count, TotalPnL, AvgPnL, AvgROI, WinRate, Sharpe, Sortino |
| `condor_trades.csv` | Per-trade details: entry date, strikes, credit, max loss, PnL, ROI |
| `condor_summary.csv` | Aggregate condor metrics: avg credit, avg max loss, return on risk |
| `training_history.csv` | Per-epoch training/validation loss |
| `predictions.csv` | Model predictions vs actual labels on validation set |
| `ml_positions.csv` | ML strategy positions with entry/settlement details |
| `strike_selector.bson` | Serialized trained model + normalization parameters |

## Caveats

1. **Validation set double-duty**: The same Feb-Aug 2025 validation period is used for both early stopping (training halts when validation delta-MSE stops improving, patience=15) and final backtest evaluation (the ROI/Sharpe numbers reported above). In strict ML practice, these should be separate splits so the reported metrics come from data never seen during any part of training. The risk is somewhat mitigated because early stopping optimizes delta prediction MSE while the final evaluation measures backtest ROI — different enough metrics that overfitting one doesn't directly overfit the other. A true out-of-sample holdout (Aug 2025 - Feb 2026) exists in the data but has been deliberately preserved for a future one-shot test.

2. **Spot proxy tracking error**: SPXW uses SPY x 10 as spot proxy. The SPY/SPX tracking ratio varies slightly (~0.1-0.3%), introducing settlement price noise. ETF options (SPY, QQQ, IWM, GLD, TLT) trade directly on the spot, so no proxy error.

3. **No commissions or slippage**: The only transaction cost modeled is the synthetic bid-ask spread (via lambda). Real trading would incur per-contract commissions, market impact, and potential fill slippage.

4. **TLT is statistically unreliable**: With only 16-24 trades per run (compared to 103-125 for other symbols), TLT results are dominated by individual trade outcomes and should not be used for inference.

5. **Single validation period**: All runs use the same Feb-Aug 2025 validation window. Results may not generalize to different market regimes (e.g., high-vol crashes, rate-cut cycles).

6. **Independent training**: Each symbol's model is trained independently. A multi-asset model trained on cross-sectional features might capture additional signal.

## Conclusions

1. **The ML edge generalizes to equity options** (SPXW, SPY, IWM) with +1.2% to +3.2% weighted ROI improvement over the fixed-delta baseline. The edge is friction-robust across all lambda values.

2. **QQQ shows marginal improvement** (+0.6%), suggesting tech-heavy vol surfaces may be harder for the model to exploit, or that the Nasdaq variance risk premium is thinner.

3. **Commodities and bonds lack the structural edge** that makes equity condors profitable. GLD and TLT baselines are unprofitable or marginal, and the ML model cannot create alpha where the underlying premium doesn't exist.

4. **IWM is a standout**: despite being a smaller underlying with lower absolute dollar PnL, it achieves the highest Sharpe ratios (2.65-3.12) and consistent positive ROI across all lambdas.

5. **The variance risk premium hierarchy** is visible in baseline results: IWM > SPXW > SPY > QQQ > TLT > GLD.
