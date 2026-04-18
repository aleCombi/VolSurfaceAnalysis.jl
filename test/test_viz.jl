using VolSurfaceAnalysis
using Test
using Dates
using Plots

# Build a synthetic VolatilitySurface from Black-76 prices, including both
# call and put records at each strike, with bid/ask populated so build_surface
# accepts them. Then verify plot_smile_with_condors returns a Plot without error.

function _synth_surface(spot::Float64, T::Float64, F::Float64; rate::Float64=0.045)
    ts = DateTime(2025, 1, 15, 18, 0)
    expiry = ts + Day(round(Int, T * 365.25))
    Ks = collect((spot - 10):1.0:(spot + 10))
    σ = 0.20
    recs = OptionRecord[]
    for K in Ks, opt in (Call, Put)
        price_usd = black76_price(F, K, T, σ, opt; r=rate)
        mark_frac = price_usd / spot
        # Synthetic ±2% bid/ask so build_surface keeps the record
        bid_frac = max(mark_frac * 0.98, 0.0)
        ask_frac = mark_frac * 1.02
        iv = price_to_iv(price_usd / F, F, K, T, opt; r=rate)
        mark_iv = isnan(iv) ? missing : iv * 100.0
        push!(recs, OptionRecord(
            "X", Underlying("X"), expiry, K, opt,
            bid_frac, ask_frac, mark_frac, mark_iv,
            missing, 100.0, spot, ts,
        ))
    end
    return build_surface(recs), expiry
end

@testset "Visualization" begin
    @testset "plot_smile_with_condors smoke" begin
        spot = 100.0
        T = 30 / 365.25
        F = spot * exp((0.045 - 0.013) * T)
        surface, expiry = _synth_surface(spot, T, F)

        specs = [
            CondorSpec(0.30, 0.10, :firebrick, "30/10"),
            CondorSpec(0.16, 0.05, :royalblue, "16/05"),
        ]

        plt = plot_smile_with_condors(surface, expiry, specs;
            rate=0.045, div_yield=0.013, atm_window=0.10,
            title="synthetic smoke test",
        )
        @test plt isa Plots.Plot
    end

    @testset "no condors → smile-only plot" begin
        spot = 100.0
        T = 7 / 365.25
        F = spot
        surface, expiry = _synth_surface(spot, T, F; rate=0.0)

        plt = plot_smile_with_condors(surface, expiry, CondorSpec[];
            rate=0.0, div_yield=0.0, atm_window=0.10,
        )
        @test plt isa Plots.Plot
    end

    @testset "expiry not on surface throws" begin
        spot = 100.0
        T = 30 / 365.25
        F = spot
        surface, _ = _synth_surface(spot, T, F)
        bad_expiry = surface.timestamp + Day(999)
        @test_throws ArgumentError plot_smile_with_condors(
            surface, bad_expiry, CondorSpec[];
        )
    end
end
