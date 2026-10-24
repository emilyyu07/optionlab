# Conventions

Every pricing function in `optionlab` follows these conventions. Docstrings refer back
here; if code and this document disagree, one of them is a bug.

## Inputs

| Quantity | Symbol | Field | Unit / convention |
|---|---|---|---|
| Spot | S | `Market.spot` | Price per share, > 0 |
| Strike | K | `Option.strike` | Price per share, > 0 |
| Time to expiry | T | `Option.expiry_years` | Years, ACT/365 (calendar days / 365), >= 0 |
| Risk-free rate | r | `Market.rate` | Annualized decimal, continuously compounded, flat (may be negative) |
| Dividend yield | q | `Market.div_yield` | Annualized decimal, continuous, flat, default 0 |
| Volatility | σ | `Market.vol` | Annualized decimal (0.20 = 20%), >= 0 |

- Discount factor to expiry: `e^{-rT}`. Forward: `F = S e^{(r-q)T}`.
- Converting dates: `T = (expiry_date - valuation_date).days / 365`. The library itself
  takes `T` directly and never handles dates.

## Outputs

- **Prices are per share.** A contract multiplier (e.g. 100) is applied only when
  displaying results, never inside the library.
- **Greeks are in raw analytical units:**

| Greek | Definition | Unit | Common display conversion |
|---|---|---|---|
| delta | ∂V/∂S | per 1.00 of spot | n/a |
| gamma | ∂²V/∂S² | per 1.00 of spot, squared | n/a |
| vega | ∂V/∂σ | per 1.00 of vol | ÷ 100 for "per vol point" |
| theta | ∂V/∂t = −∂V/∂T | per year of calendar time | ÷ 365 for "per day" |
| rho | ∂V/∂r | per 1.00 of rate | ÷ 100 for "per 1%", ÷ 10,000 per bp |

Theta is the change in value as calendar time *passes*, so it is the negative of the
derivative with respect to time-to-expiry `T`. It is usually negative for long options.

## Edge cases

- `T == 0`: return intrinsic value, `max(S − K, 0)` for a call, `max(K − S, 0)` for a put.
- `σ == 0`: the terminal price is the forward, so return the discounted forward payoff:
  `max(S e^{−qT} − K e^{−rT}, 0)` for a call, `max(K e^{−rT} − S e^{−qT}, 0)` for a put.
  (European. For an American option on a tree, decide and document the behaviour.)
- Invalid inputs raise `ValueError`: non-positive spot or strike, negative vol or expiry,
  any non-finite input, an American option passed to a European-only model, binomial
  `steps < 1`, Monte Carlo `n_paths < 2` or odd `n_paths` with antithetic sampling.
  Validation of `Option` and `Market` happens on construction (`types.py`).

## No-arbitrage bounds (European)

- Call: `max(S e^{−qT} − K e^{−rT}, 0) ≤ C ≤ S e^{−qT}`
- Put: `max(K e^{−rT} − S e^{−qT}, 0) ≤ P ≤ K e^{−rT}`
- Put-call parity: `C − P = S e^{−qT} − K e^{−rT}`

`implied_vol` raises `ValueError` for a price outside these bounds.

## Randomness

- All randomness comes from a `numpy.random.Generator` backed by `PCG64`, created from an
  explicit `seed` argument (`optionlab.make_rng`). No `np.random.seed`, no module-level
  generators, no global state. Same inputs + same seed ⇒ bit-identical results.
- Monte Carlo `std_err` is the sample standard deviation (`ddof=1`) of the i.i.d.
  quantities averaged, divided by the square root of their count. With antithetic
  sampling those quantities are the pair averages, not the individual paths.
- `MCResult.ci95` is the normal-approximation interval `price ± 1.96 · std_err`.

## Models

| Model | Function | Exercise |
|---|---|---|
| Black-Scholes-Merton (closed form) | `price_bs`, `greeks_bs`, `implied_vol` | European |
| Cox-Ross-Rubinstein tree (`u = e^{σ√Δt}`, `d = 1/u`) | `price_binomial` | European, American |
| Monte Carlo, exact GBM terminal sampling | `price_mc` | European |
