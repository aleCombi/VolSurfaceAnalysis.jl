# Composable components for strangle-selector exploration: every strangle
# selector is a triple (Conditioning, Score, Picker). Components dispatch by
# type and operate on per-entry state (rv, mom) and a precomputed (n_p, n_c)
# PnL grid produced from a single backtest's `state.history`.
#
# Used by `run_strangle_selector_sweep` (post-hoc replay over many triplets
# sharing one backtest pass).

using Statistics
using Dates

# =============================================================================
# Conditioning — buckets trailing entries before scoring
# =============================================================================

abstract type Conditioning end

"""All trailing trades pooled into one bin (no regime split)."""
struct Unconditional <: Conditioning end

"""Median split of trailing-`rv_window`-day realized vol → 2 bins (LOW, HIGH)."""
struct RVBinary <: Conditioning
    rv_window::Int
end
RVBinary(; rv_window::Int=20) = RVBinary(rv_window)

"""Tertile split of trailing realized vol → 3 bins."""
struct RVTertile <: Conditioning
    rv_window::Int
end
RVTertile(; rv_window::Int=20) = RVTertile(rv_window)

"""Sign of trailing-`mom_window`-day cumulative log-return → 2 bins (DN, UP)."""
struct MomSign <: Conditioning
    mom_window::Int
end
MomSign(; mom_window::Int=5) = MomSign(mom_window)

"""Cross of RV-binary and momentum-sign → 4 bins."""
struct RVxMom <: Conditioning
    rv_window::Int
    mom_window::Int
end
RVxMom(; rv_window::Int=20, mom_window::Int=5) = RVxMom(rv_window, mom_window)

"""Number of bins this conditioning produces."""
n_bins(::Unconditional) = 1
n_bins(::RVBinary) = 2
n_bins(::RVTertile) = 3
n_bins(::MomSign) = 2
n_bins(::RVxMom) = 4

"""
    fit_threshold(conditioning, train_idxs, state) -> threshold

Compute the bin threshold(s) from the training window's state vectors.
Returns `nothing` when conditioning is unconditional, or threshold values
otherwise. State is a NamedTuple `(rv::Vector{Float64}, mom::Vector{Float64})`.
"""
fit_threshold(::Unconditional, train_idxs, state) = nothing

function fit_threshold(c::RVBinary, train_idxs, state)
    rv_t = [state.rv[k] for k in train_idxs if !isnan(state.rv[k])]
    isempty(rv_t) && return NaN
    median(rv_t)
end

function fit_threshold(c::RVTertile, train_idxs, state)
    rv_t = [state.rv[k] for k in train_idxs if !isnan(state.rv[k])]
    length(rv_t) < 6 && return (NaN, NaN)
    (quantile(rv_t, 1/3), quantile(rv_t, 2/3))
end

fit_threshold(::MomSign, train_idxs, state) = 0.0

function fit_threshold(c::RVxMom, train_idxs, state)
    rv_t = [state.rv[k] for k in train_idxs if !isnan(state.rv[k])]
    isempty(rv_t) && return NaN
    median(rv_t)
end

"""
    classify(conditioning, k, threshold, state) -> Int (1-based bin index)

Route entry `k` to a bin given the frozen threshold. Defaults to bin 1 when
state is missing for the entry.
"""
classify(::Unconditional, k, threshold, state) = 1

function classify(::RVBinary, k, threshold, state)
    (isnan(state.rv[k]) || (threshold isa Float64 && isnan(threshold))) && return 1
    state.rv[k] >= threshold ? 2 : 1
end

function classify(::RVTertile, k, threshold, state)
    (isnan(state.rv[k]) || isnan(threshold[1])) && return 1
    state.rv[k] < threshold[1] ? 1 : (state.rv[k] < threshold[2] ? 2 : 3)
end

function classify(::MomSign, k, threshold, state)
    isnan(state.mom[k]) && return 1
    state.mom[k] >= threshold ? 2 : 1
end

function classify(::RVxMom, k, threshold, state)
    (isnan(state.rv[k]) || (threshold isa Float64 && isnan(threshold)) ||
     isnan(state.mom[k])) && return 1
    rv_hi = state.rv[k] >= threshold
    mom_up = state.mom[k] >= 0
    1 + (rv_hi ? 2 : 0) + (mom_up ? 1 : 0)
end

# =============================================================================
# Score — applied to a bucket's PnL series → cell score
# =============================================================================

abstract type Score end

"""Mean of the PnL series."""
struct MeanScore <: Score end

"""`mean(pnl) - λ·var(pnl)`. Higher λ penalises variance more."""
struct MeanVarScore <: Score
    lambda::Float64
end

"""Annualized Sharpe (mean/std × √annualization)."""
struct SharpeScore <: Score
    annualization::Float64
end
SharpeScore(; annualization::Real=252) = SharpeScore(Float64(annualization))

"""Sharpe computed only on the last `fraction` of the input vector (held-out)."""
struct SharpeHeldOut <: Score
    fraction::Float64
    annualization::Float64
end
SharpeHeldOut(; fraction::Real=1/3, annualization::Real=252) =
    SharpeHeldOut(Float64(fraction), Float64(annualization))

"""`mean(pnl) - z · |CVaR_α(pnl)|` (CVaR-regularized expected return)."""
struct CVaRScore <: Score
    z::Float64
    alpha::Float64
end
CVaRScore(; z::Real=1.0, alpha::Real=0.05) = CVaRScore(Float64(z), Float64(alpha))

"""Compute the score on a vector of (possibly NaN) PnL observations."""
function score_pnls(::MeanScore, v::AbstractVector{<:Real})
    c = filter(!isnan, v)
    isempty(c) ? -Inf : mean(c)
end

function score_pnls(s::MeanVarScore, v::AbstractVector{<:Real})
    c = filter(!isnan, v)
    length(c) < 2 ? -Inf : mean(c) - s.lambda * var(c)
end

function score_pnls(s::SharpeScore, v::AbstractVector{<:Real})
    c = filter(!isnan, v)
    (isempty(c) || length(c) < 2 || std(c) == 0) && return -Inf
    mean(c) / std(c) * sqrt(s.annualization)
end

function score_pnls(s::SharpeHeldOut, v::AbstractVector{<:Real})
    c = filter(!isnan, v)
    K = max(1, round(Int, length(c) * s.fraction))
    K > length(c) && return -Inf
    score_pnls(SharpeScore(s.annualization), c[end-K+1:end])
end

function score_pnls(s::CVaRScore, v::AbstractVector{<:Real})
    c = filter(!isnan, v)
    length(c) < 2 && return -Inf
    sorted = sort(c)
    n_tail = max(1, floor(Int, s.alpha * length(sorted)))
    cvar = mean(sorted[1:n_tail])
    mean(c) - s.z * abs(cvar)
end

# =============================================================================
# Picker — extracts (i, j) from the score grid
# =============================================================================

abstract type Picker end

"""Single-cell maximum of the score grid."""
struct ArgmaxPicker <: Picker end

"""Mean (i, j) of the top-K cells, rounded to nearest grid point."""
struct TopKPicker <: Picker
    K::Int
end

"""Replace each cell with the mean of its 3×3 neighbourhood, then argmax."""
struct ShrinkagePicker <: Picker end

"""Argmax with `−γ` penalty applied to cells differing from the prior pick."""
struct StickyPicker <: Picker
    gamma::Float64
end

"""
    pick(picker, S, prev) -> Union{Nothing, CartesianIndex{2}}

Choose a cell from the score grid `S`. `prev` is the prior fold's chosen
`(i, j)` for the same bin (or `nothing` on the first fold). Returns
`nothing` if no cell is finite.
"""
function pick(::ArgmaxPicker, S::AbstractMatrix, prev)
    all(==(-Inf), S) && return nothing
    argmax(S)
end

function pick(p::TopKPicker, S::AbstractMatrix, prev)
    all(==(-Inf), S) && return nothing
    nfin = count(isfinite, S)
    K = min(p.K, nfin)
    K == 0 && return nothing
    cis = CartesianIndices(S)[sortperm(vec(S), rev=true)[1:K]]
    np, nc = size(S)
    CartesianIndex(clamp(round(Int, mean(c[1] for c in cis)), 1, np),
                   clamp(round(Int, mean(c[2] for c in cis)), 1, nc))
end

function pick(::ShrinkagePicker, S::AbstractMatrix, prev)
    np, nc = size(S)
    Sm = fill(-Inf, np, nc)
    for i = 1:np, j = 1:nc
        acc, n = 0.0, 0
        for di = -1:1, dj = -1:1
            ii, jj = i + di, j + dj
            (1 <= ii <= np && 1 <= jj <= nc && isfinite(S[ii, jj])) || continue
            acc += S[ii, jj]; n += 1
        end
        n > 0 && (Sm[i, j] = acc / n)
    end
    all(==(-Inf), Sm) && return nothing
    argmax(Sm)
end

function pick(p::StickyPicker, S::AbstractMatrix, prev)
    all(==(-Inf), S) && return nothing
    if prev === nothing
        return argmax(S)
    end
    Sp = copy(S)
    np, nc = size(Sp)
    @inbounds for i = 1:np, j = 1:nc
        if !(i == prev[1] && j == prev[2])
            Sp[i, j] -= p.gamma
        end
    end
    argmax(Sp)
end

# =============================================================================
# Helper: state-vector construction from a strangle-selector history
# =============================================================================

"""
    compute_strangle_state(history; rv_window, mom_window)
        -> (rv::Vector{Float64}, mom::Vector{Float64})

Compute per-entry trailing realized-vol and momentum vectors from a strangle
selector's `state.history` (entries with `:spot_at_entry` field). NaN where
not enough trailing data.
"""
function compute_strangle_state(history; rv_window::Int=20, mom_window::Int=5)
    N = length(history)
    spots = [h.spot_at_entry for h in history]
    log_rets = [k == 1 ? NaN : log(spots[k] / spots[k-1]) for k in 1:N]

    rv = fill(NaN, N)
    for k in (rv_window+1):N
        r = filter(!isnan, log_rets[(k-rv_window+1):k])
        length(r) >= 5 && (rv[k] = std(r) * sqrt(252))
    end

    mom = fill(NaN, N)
    for k in (mom_window+1):N
        r = filter(!isnan, log_rets[(k-mom_window+1):k])
        length(r) >= 2 && (mom[k] = sum(r))
    end

    return (rv=rv, mom=mom)
end

"""
    compute_pnl_grid(history, source) -> Vector{Matrix{Float64}}

For each entry in `history`, compute the n_p × n_c PnL matrix using the
recorded `credit_frac` and the settlement spot from `source`. Cells are NaN
where `credit_frac` is NaN or settlement is missing.
"""
function compute_pnl_grid(history, source)
    N = length(history)
    isempty(history) && return Matrix{Float64}[]
    e1 = first(history)
    n_p, n_c = size(e1.credit_frac)
    pnl = [fill(NaN, n_p, n_c) for _ in 1:N]
    for k in 1:N
        e = history[k]
        spot_settle = get_settlement_spot(source, e.expiry)
        ismissing(spot_settle) && continue
        M = pnl[k]
        @inbounds for i in 1:n_p, j in 1:n_c
            cf = e.credit_frac[i, j]
            isnan(cf) && continue
            M[i, j] = cf * e.spot_at_entry -
                      max(e.put_strikes[i] - Float64(spot_settle), 0.0) -
                      max(Float64(spot_settle) - e.call_strikes[j], 0.0)
        end
    end
    return pnl
end
