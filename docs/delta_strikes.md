# Quoting Strikes by Delta

## Intuition

Identifying an option by its Black-Scholes delta (e.g. "the 25-delta call") rather than its dollar strike (e.g. "SPY 580 call"). Delta is the option's sensitivity to spot — it ranges 0→1 for calls, 0→-1 for puts — and conveniently doubles as a rough risk-neutral probability of finishing ITM.

**Why traders prefer it:**

- **Moneyness-normalized.** A 25-delta put is "the same kind of trade" whether SPY is at 400 or 600, or whether you're trading SPY vs BTC.
- **Vol- and tenor-adjusted.** As IV rises or expiry lengthens, the strike at a given delta moves further OTM automatically.
- **Probabilistic intuition.** 16-delta ≈ ~1σ OTM ≈ ~16% chance ITM at expiry under the risk-neutral measure.
- **Surface-native.** Vol surfaces are often parameterized in delta space (delta skew, 25Δ risk reversal, 25Δ butterfly).

## Setup

Under Black-Scholes, with spot $S$, strike $K$, rate $r$, dividend yield $q$, vol $\sigma$, time to expiry $\tau$:

$$d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau}}, \quad d_2 = d_1 - \sigma\sqrt{\tau}$$

Spot deltas:

$$\Delta_C = e^{-q\tau}\,N(d_1), \qquad \Delta_P = -e^{-q\tau}\,N(-d_1) = e^{-q\tau}\big(N(d_1) - 1\big)$$

where $N(\cdot)$ is the standard normal CDF.

## Inverting delta → strike

Fix a target call delta $\Delta^\star \in (0, e^{-q\tau})$. Solve for $d_1$:

$$d_1 = N^{-1}\!\big(e^{q\tau}\,\Delta^\star\big)$$

Then back out $K$ from the definition of $d_1$:

$$\boxed{\,K(\Delta^\star) = S\,\exp\!\Big[(r - q + \tfrac{1}{2}\sigma^2)\tau \;-\; \sigma\sqrt{\tau}\,N^{-1}\!\big(e^{q\tau}\,\Delta^\star\big)\Big]\,}$$

For a put with target $\Delta^\star \in (-e^{-q\tau}, 0)$, replace the inversion with $d_1 = -N^{-1}(-e^{q\tau}\Delta^\star)$.

## Why it normalizes

Take logs and rearrange — log-moneyness at a fixed delta is

$$\ln(K/S) = (r - q + \tfrac{1}{2}\sigma^2)\tau \;-\; \sigma\sqrt{\tau}\cdot z^\star, \quad z^\star := N^{-1}(e^{q\tau}\Delta^\star)$$

So strike-distance scales as $\sigma\sqrt{\tau}$ — i.e. **in standard deviations of log-spot**. A 16-delta call sits at $z^\star \approx 1$, so $\ln(K/S) \approx \sigma\sqrt{\tau}$, the classic "1σ OTM" shorthand.

## Delta ≈ risk-neutral ITM probability

Under the risk-neutral measure $\mathbb{Q}$:

$$\mathbb{Q}(S_T > K) = N(d_2)$$

Compare to $\Delta_C = e^{-q\tau}N(d_1)$. They differ by:

- the discount factor $e^{-q\tau}$ (small for short tenors),
- $d_1$ vs $d_2$, gap $= \sigma\sqrt{\tau}$.

For short-dated, low-vol contracts, $N(d_1) \approx N(d_2)$, hence the rule of thumb "$|\Delta|$ ≈ probability ITM." It's an approximation, not an identity.

## Surface parameterization

Once strikes live on the delta axis, you can write the smile as $\sigma(\Delta, \tau)$ and define standard quotes:

$$\text{RR}_{25} = \sigma(25\Delta\,\text{call}) - \sigma(25\Delta\,\text{put})$$

$$\text{BF}_{25} = \tfrac{1}{2}\big[\sigma(25\Delta\,\text{call}) + \sigma(25\Delta\,\text{put})\big] - \sigma_{\text{ATM}}$$

— skew and curvature in vol-of-vol units, comparable across underlyings and tenors.

## Subtlety: $\sigma$ depends on $K$

The inversion above assumed a single $\sigma$. With a smile, $\sigma = \sigma(K,\tau)$, so

$$\Delta_C(K) = e^{-q\tau}N\!\big(d_1(K, \sigma(K,\tau))\big)$$

is a fixed-point problem in $K$. In `src/strategies/strike_selection.jl` (`delta_selector`), this is solved numerically against the fitted surface.
