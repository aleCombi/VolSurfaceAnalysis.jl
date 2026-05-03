# Sweep over (Conditioning, Score, Picker) triplets for short-strangle selection.
# One backtest pass; each triplet replayed against the recorded shadow data.

using Pkg; Pkg.activate(@__DIR__)
using VolSurfaceAnalysis
using Dates

experiment = (;
    symbol="SPY",
    start_date=Date(2014, 6, 2),
    end_date=Date(2026, 3, 27),
    entry_time=Time(14, 0),
    expiry_interval=Day(1),
    max_tau_days=2.0,
    spread_lambda=0.7,
    rate=0.045,
    div_yield=0.013,
    put_deltas=collect(0.05:0.05:0.40),
    call_deltas=collect(0.05:0.05:0.40),
    train_days=90,
    test_days=30,
    step_days=30,
    rv_window=20,
    mom_window=5,
    peeking_combo=(0.20, 0.05),
    triplets=[
        # References
        (name="0a. static (0.20, 0.05) [peeking]",  conditioning=:peeking,         score=nothing,                  picker=nothing),
        (name="0b. frozen first-fold argmax",       conditioning=:frozen,          score=SharpeScore(),            picker=ArgmaxPicker()),

        # Unconditional, score sweep
        (name="1.  uncon | mean (λ=0)         | argmax",     conditioning=Unconditional(), score=MeanScore(),           picker=ArgmaxPicker()),
        (name="2.  uncon | mean−0.05·var      | argmax",     conditioning=Unconditional(), score=MeanVarScore(0.05),    picker=ArgmaxPicker()),
        (name="3.  uncon | mean−1.0·var       | argmax",     conditioning=Unconditional(), score=MeanVarScore(1.0),     picker=ArgmaxPicker()),
        (name="4.  uncon | Sharpe             | argmax",     conditioning=Unconditional(), score=SharpeScore(),         picker=ArgmaxPicker()),

        # Unconditional, picker sweep
        (name="5.  uncon | Sharpe             | top-3",      conditioning=Unconditional(), score=SharpeScore(),         picker=TopKPicker(3)),
        (name="6.  uncon | Sharpe             | top-5",      conditioning=Unconditional(), score=SharpeScore(),         picker=TopKPicker(5)),
        (name="7.  uncon | Sharpe             | shrinkage",  conditioning=Unconditional(), score=SharpeScore(),         picker=ShrinkagePicker()),
        (name="8.  uncon | Sharpe             | sticky γ=1", conditioning=Unconditional(), score=SharpeScore(),         picker=StickyPicker(1.0)),
        (name="9.  uncon | Sharpe(heldout⅓)  | argmax",      conditioning=Unconditional(), score=SharpeHeldOut(),       picker=ArgmaxPicker()),

        # RV-binary
        (name="10. RV-bin| Sharpe             | argmax",     conditioning=RVBinary(20),    score=SharpeScore(),         picker=ArgmaxPicker()),
        (name="11. RV-bin| Sharpe             | top-3 ⭐",  conditioning=RVBinary(20),    score=SharpeScore(),         picker=TopKPicker(3)),
        (name="12. RV-bin| Sharpe             | top-5",      conditioning=RVBinary(20),    score=SharpeScore(),         picker=TopKPicker(5)),
        (name="13. RV-bin| Sharpe             | shrinkage",  conditioning=RVBinary(20),    score=SharpeScore(),         picker=ShrinkagePicker()),
        (name="14. RV-bin| Sharpe             | sticky γ=1", conditioning=RVBinary(20),    score=SharpeScore(),         picker=StickyPicker(1.0)),
        (name="15. RV-bin| mean (λ=0)         | argmax",     conditioning=RVBinary(20),    score=MeanScore(),           picker=ArgmaxPicker()),
        (name="16. RV-bin| mean−0.05·var      | argmax",     conditioning=RVBinary(20),    score=MeanVarScore(0.05),    picker=ArgmaxPicker()),
        (name="17. RV-bin| mean−1.0·var       | argmax",     conditioning=RVBinary(20),    score=MeanVarScore(1.0),     picker=ArgmaxPicker()),
        (name="18. RV-bin| mean−0.05·var      | top-3",      conditioning=RVBinary(20),    score=MeanVarScore(0.05),    picker=TopKPicker(3)),

        # Other conditionings
        (name="19. RV-tertile| Sharpe         | top-3",      conditioning=RVTertile(20),   score=SharpeScore(),         picker=TopKPicker(3)),
        (name="20. mom-sign  | Sharpe         | argmax",     conditioning=MomSign(5),      score=SharpeScore(),         picker=ArgmaxPicker()),
        (name="21. mom-sign  | Sharpe         | top-3",      conditioning=MomSign(5),      score=SharpeScore(),         picker=TopKPicker(3)),
        (name="22. RV×mom    | Sharpe         | argmax",     conditioning=RVxMom(20, 5),   score=SharpeScore(),         picker=ArgmaxPicker()),
    ],
)

run_strangle_selector_sweep(; output_root=@__DIR__, experiment...)
