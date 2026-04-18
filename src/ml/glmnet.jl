# GLMNet integration — ridge, lasso, elastic net for sizing/classification
#
# Drop-in replacement for Flux models in the MLSizer pipeline.
# Uses GLMNet.jl (wrapper around Fortran glmnet) with built-in CV for lambda selection.

using GLMNet: glmnet, Binomial, CompressedPredictorMatrix

# =============================================================================
# Model wrapper
# =============================================================================

"""
    GLMNetModel(coefs, intercept)

Lightweight wrapper around GLMNet coefficients that satisfies the callable
model interface expected by `MLSizer` and `score_candidates`.

Callable as `model(X)` where `X` is `(features, samples)` → `(1, samples)`.
"""
struct GLMNetModel
    coefs::Vector{Float32}
    intercept::Float32
end

function (m::GLMNetModel)(X::AbstractMatrix)
    # X is (features, samples) — our convention
    return reshape(m.coefs' * X .+ m.intercept, 1, :)
end

# Single-column convenience (MLSizer path)
function (m::GLMNetModel)(x::AbstractVector)
    return [sum(m.coefs .* x) + m.intercept]
end

# =============================================================================
# Ridge / Elastic Net regression
# =============================================================================

"""
    train_ridge!(::Nothing, X, Y; alpha=0.0, val_fraction=0.2, kwargs...)

Train a GLMNet linear model (ridge/lasso/elastic net) with lambda selected on
a chronological validation holdout (last `val_fraction` of samples).

Data is assumed to be in temporal order (earliest first). No random shuffling.

# Arguments
- First argument is ignored (API symmetry with `train_model!`; GLMNet builds its own model).
- `X::Matrix{Float32}`: features `(dims, samples)` in temporal order
- `Y::Matrix{Float32}`: targets `(1, samples)` in temporal order

# Keyword Arguments
- `alpha`: elastic net mixing (0=ridge, 1=lasso). Clamped to `max(alpha, 1e-4)`.
- `lambda`: specific lambda value, or `nothing` for temporal holdout selection (default).
- `val_fraction`: fraction of data held out chronologically for lambda selection (default 0.2).
- Extra kwargs are silently absorbed for drop-in compatibility with `train_model!`.

Returns `(model::GLMNetModel, feature_means, feature_stds, history)`.
"""
function train_ridge!(
    ::Nothing,
    X::Matrix{Float32},
    Y::Matrix{Float32};
    alpha::Float64=0.0,
    lambda::Union{Nothing,Float64}=nothing,
    val_fraction::Float64=0.2,
    kwargs...  # absorb Flux kwargs (epochs, lr, etc.)
)
    n = size(X, 2)
    n_val = max(1, round(Int, n * val_fraction))
    n_train = n - n_val

    # Chronological split: train on first n_train, validate on last n_val
    X_train = X[:, 1:n_train]
    Y_train = Y[:, 1:n_train]
    X_val = X[:, n_train+1:end]
    Y_val = Y[:, n_train+1:end]

    # Normalization (from training set only)
    feature_means = vec(mean(X_train; dims=2))
    feature_stds = vec(std(X_train; dims=2))
    X_train_norm = _normalize(X_train, feature_means, feature_stds)
    X_val_norm = _normalize(X_val, feature_means, feature_stds)

    # GLMNet expects (samples, features) and Float64
    Xt = Matrix{Float64}(X_train_norm')
    yt = vec(Float64.(Y_train))

    alpha_clamped = max(alpha, 1e-4)

    if lambda === nothing
        # Fit full lambda path on training data, select best on chronological val
        path = glmnet(Xt, yt; alpha=alpha_clamped)
        best_idx = _select_lambda_mse(path, X_val_norm, Y_val)
        best_lambda = path.lambda[best_idx]
    else
        path = glmnet(Xt, yt; alpha=alpha_clamped, lambda=[lambda])
        best_idx = 1
        best_lambda = lambda
    end

    coefs = Float32.(Vector(path.betas[:, best_idx]))
    intercept = Float32(path.a0[best_idx])
    model = GLMNetModel(coefs, intercept)

    # Validation loss
    pred_val = model(X_val_norm)
    val_mse = Float64(mean((pred_val .- Y_val) .^ 2))

    n_nonzero = count(!=(0), coefs)
    history = (
        val_loss=[val_mse],
        train_loss=[Float64(mean((model(X_train_norm) .- Y_train) .^ 2))],
        lambda=best_lambda,
        alpha=alpha_clamped,
        n_nonzero=n_nonzero,
    )

    return (model, feature_means, feature_stds, history)
end

"""Select lambda index that minimizes MSE on validation data."""
function _select_lambda_mse(path, X_val_norm, Y_val)
    y_val = vec(Float64.(Y_val))
    best_idx = 1
    best_mse = Inf
    for i in 1:length(path.lambda)
        coefs_i = Float32.(Vector(path.betas[:, i]))
        intercept_i = Float32(path.a0[i])
        m = GLMNetModel(coefs_i, intercept_i)
        pred = vec(m(X_val_norm))
        mse = mean((pred .- y_val) .^ 2)
        if mse < best_mse
            best_mse = mse
            best_idx = i
        end
    end
    return best_idx
end

# =============================================================================
# Logistic classifier (elastic net)
# =============================================================================

"""
    train_glmnet_classifier!(::Nothing, X, Y; alpha=1.0, val_fraction=0.2, kwargs...)

Train a GLMNet logistic regression classifier with lambda selected on a
chronological validation holdout (last `val_fraction` of samples).

Data is assumed to be in temporal order (earliest first). No random shuffling.
The model outputs **logits** (linear predictor), not probabilities — compatible
with `probability_sizing` which applies sigmoid externally.

# Arguments
Same as `train_ridge!`, but `Y` should contain binary labels (0/1).

# Keyword Arguments
- `alpha`: elastic net mixing (1=lasso, 0=ridge). Default 1.0 (lasso for feature selection).
- `pos_weight`: ignored (GLMNet uses its own internal weighting). Accepted for API compatibility.

Returns `(model::GLMNetModel, feature_means, feature_stds, history)`.
"""
function train_glmnet_classifier!(
    ::Nothing,
    X::Matrix{Float32},
    Y::Matrix{Float32};
    alpha::Float64=1.0,
    lambda::Union{Nothing,Float64}=nothing,
    val_fraction::Float64=0.2,
    pos_weight::Float64=1.0,  # absorbed for API compat
    kwargs...
)
    n = size(X, 2)
    n_val = max(1, round(Int, n * val_fraction))
    n_train = n - n_val

    # Chronological split: train on first n_train, validate on last n_val
    X_train = X[:, 1:n_train]
    Y_train = Y[:, 1:n_train]
    X_val = X[:, n_train+1:end]
    Y_val = Y[:, n_train+1:end]

    feature_means = vec(mean(X_train; dims=2))
    feature_stds = vec(std(X_train; dims=2))
    X_train_norm = _normalize(X_train, feature_means, feature_stds)
    X_val_norm = _normalize(X_val, feature_means, feature_stds)

    # GLMNet Binomial needs (samples, features) and a (samples, 2) response matrix
    Xt = Matrix{Float64}(X_train_norm')
    yt_vec = vec(Float64.(Y_train))
    Yt_mat = hcat(1.0 .- yt_vec, yt_vec)  # [P(class=0), P(class=1)]

    alpha_clamped = max(alpha, 1e-4)

    if lambda === nothing
        # Fit full lambda path on training data, select best on chronological val
        path = glmnet(Xt, Yt_mat, Binomial(); alpha=alpha_clamped)
        best_idx = _select_lambda_bce(path, X_val_norm, Y_val)
        best_lambda = path.lambda[best_idx]
    else
        path = glmnet(Xt, Yt_mat, Binomial(); alpha=alpha_clamped, lambda=[lambda])
        best_idx = 1
        best_lambda = lambda
    end

    coefs = Float32.(Vector(path.betas[:, best_idx]))
    intercept = Float32(path.a0[best_idx])
    model = GLMNetModel(coefs, intercept)

    # Validation BCE loss (model outputs logits)
    logits_val = vec(model(X_val_norm))
    y_val_vec = vec(Y_val)
    val_bce = -Float64(mean(
        y_val_vec .* log.(max.(Flux.sigmoid.(logits_val), 1f-7)) .+
        (1f0 .- y_val_vec) .* log.(max.(1f0 .- Flux.sigmoid.(logits_val), 1f-7))
    ))

    n_nonzero = count(!=(0), coefs)
    history = (
        val_loss=[val_bce],
        train_loss=Float64[],
        lambda=best_lambda,
        alpha=alpha_clamped,
        n_nonzero=n_nonzero,
    )

    return (model, feature_means, feature_stds, history)
end

"""Select lambda index that minimizes BCE on validation data."""
function _select_lambda_bce(path, X_val_norm, Y_val)
    y_val = vec(Float64.(Y_val))
    best_idx = 1
    best_bce = Inf
    for i in 1:length(path.lambda)
        coefs_i = Float32.(Vector(path.betas[:, i]))
        intercept_i = Float32(path.a0[i])
        m = GLMNetModel(coefs_i, intercept_i)
        logits = vec(Float64.(m(X_val_norm)))
        sig = 1.0 ./ (1.0 .+ exp.(-logits))
        bce = -mean(
            y_val .* log.(max.(sig, 1e-7)) .+
            (1.0 .- y_val) .* log.(max.(1.0 .- sig, 1e-7))
        )
        if bce < best_bce
            best_bce = bce
            best_idx = i
        end
    end
    return best_idx
end
