# Strategy helpers

function _nearest_strike(strikes::Vector{Float64}, target::Float64)::Float64
    distances = abs.(strikes .- target)
    return strikes[argmin(distances)]
end

function _pick_otm_strike(
    strikes::Vector{Float64},
    spot::Float64,
    target::Float64;
    side::Symbol
)::Float64
    otm = side == :put ? filter(s -> s < spot, strikes) : filter(s -> s > spot, strikes)
    return !isempty(otm) ? _nearest_strike(otm, target) : _nearest_strike(strikes, target)
end

function _select_expiry(
    expiry_interval::Period,
    surface::VolatilitySurface
)::Union{Nothing,Tuple{DateTime,Float64,Float64}}
    expiry_target = surface.timestamp + expiry_interval
    tau_target = time_to_expiry(expiry_target, surface.timestamp)
    tau_target <= 0.0 && return nothing

    expiries = unique(rec.expiry for rec in surface.records)
    isempty(expiries) && return nothing

    taus = [time_to_expiry(e, surface.timestamp) for e in expiries]
    idx = argmin(abs.(taus .- tau_target))
    expiry = expiries[idx]
    tau_closest = taus[idx]

    return (expiry, tau_target, tau_closest)
end

function _condor_strikes(
    put_strikes::Vector{Float64},
    call_strikes::Vector{Float64},
    spot::Float64,
    short_put_pct::Float64,
    short_call_pct::Float64,
    long_put_pct::Float64,
    long_call_pct::Float64
)::Union{Nothing, Tuple{Float64,Float64,Float64,Float64}}
    target_short_put = spot * short_put_pct
    target_short_call = spot * short_call_pct
    target_long_put = spot * long_put_pct
    target_long_call = spot * long_call_pct

    otm_puts = filter(s -> s < spot, put_strikes)
    otm_calls = filter(s -> s > spot, call_strikes)

    short_put = !isempty(otm_puts) ? _nearest_strike(otm_puts, target_short_put) :
                _nearest_strike(put_strikes, target_short_put)
    short_call = !isempty(otm_calls) ? _nearest_strike(otm_calls, target_short_call) :
                 _nearest_strike(call_strikes, target_short_call)

    far_otm_puts = filter(s -> s < short_put, put_strikes)
    far_otm_calls = filter(s -> s > short_call, call_strikes)

    long_put = if !isempty(far_otm_puts)
        _nearest_strike(far_otm_puts, target_long_put)
    else
        !isempty(otm_puts) ? minimum(otm_puts) : _nearest_strike(put_strikes, target_long_put)
    end

    long_call = if !isempty(far_otm_calls)
        _nearest_strike(far_otm_calls, target_long_call)
    else
        !isempty(otm_calls) ? maximum(otm_calls) : _nearest_strike(call_strikes, target_long_call)
    end

    if !(long_put < short_put < spot < short_call < long_call)
        return nothing
    end

    return (short_put, short_call, long_put, long_call)
end
