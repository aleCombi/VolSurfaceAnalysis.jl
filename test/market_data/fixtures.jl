# Shared fixtures for the market_data suites. Constants are prefixed `_MD_`
# because runtests.jl includes every suite into one module.

const _MD_SPY = Underlying("SPY")
const _MD_SPX = Underlying("SPX")
const _MD_USD = Currency("USD")

const _MD_T1 = DateTime(2024, 1, 15, 19, 30)
const _MD_T2 = DateTime(2024, 1, 15, 19, 31)
const _MD_T3 = DateTime(2024, 1, 16, 19, 30)
const _MD_EXPIRY = DateTime(2024, 2, 16, 21, 0)

# One option bar; `underlying` and `timestamp` are the selector and the
# visibility time the protocol keys on.
function _md_bar(u::Underlying, ts::DateTime, strike::Float64;
                 otype::OptionType=Call, open=1.00, high=1.20, low=0.80, close=1.00, volume=10.0)
    OptionBar("O:$(ticker(u))240216$(otype == Call ? "C" : "P")$(lpad(round(Int, strike * 1000), 8, '0'))",
              u, _MD_EXPIRY, strike, otype, open, high, low, close, volume, ts)
end

_md_spot(u::Underlying, ts::DateTime, price::Float64) = SpotPrice(u, price, ts)

# Bars for SPY and SPX at the three timestamps, two strikes each, SPX rows
# interleaved with SPY rows in input order so selector filtering is tested.
function _md_bars()
    out = OptionBar[]
    for ts in (_MD_T1, _MD_T2, _MD_T3), u in (_MD_SPY, _MD_SPX)
        base = u === _MD_SPY ? 480.0 : 4800.0
        push!(out, _md_bar(u, ts, base - 10.0; otype=Put))
        push!(out, _md_bar(u, ts, base + 10.0; otype=Call))
    end
    out
end

function _md_spots()
    out = SpotPrice[]
    for (i, ts) in enumerate((_MD_T1, _MD_T2, _MD_T3))
        push!(out, _md_spot(_MD_SPY, ts, 480.0 + i))
        push!(out, _md_spot(_MD_SPX, ts, 4800.0 + i))
    end
    out
end
