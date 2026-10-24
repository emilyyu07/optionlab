"""Monte Carlo pricing of European options under geometric Brownian motion.

Units and conventions: docs/conventions.md.
"""

from __future__ import annotations

import numpy as np

from optionlab.types import Market, MCResult, Option, require_european


def make_rng(seed: int) -> np.random.Generator:
    """Return a fresh PCG64 generator for `seed`. Never use numpy's global random state."""
    return np.random.Generator(np.random.PCG64(seed))


def price_mc(
    opt: Option,
    mkt: Market,
    n_paths: int,
    seed: int,
    antithetic: bool = True,
    control_variate: bool = True,
) -> MCResult:
    """Estimate a European option price by simulating terminal stock prices.

    Args:
        opt: European call or put. American options raise ValueError.
        mkt: Spot, flat continuously compounded rate r, vol sigma, dividend yield q.
        n_paths: Total number of simulated terminal prices, >= 2. With antithetic=True
            this counts both members of each pair, so it must be even.
        seed: Seed for make_rng(seed). Same inputs + same seed must give identical results.
        antithetic: Pair each normal draw Z with -Z.
        control_variate: Use the discounted terminal stock price as a control variate.

    Returns:
        MCResult(price, std_err): price per share and the standard error of that estimate.

    Conventions (docs/conventions.md):
        - All randomness comes from make_rng(seed); no global state.
        - T == 0: return MCResult(intrinsic value, 0.0).
        - sigma == 0: the payoff is deterministic; return MCResult(discounted forward
          payoff, 0.0). Watch for 0/0 in the control-variate coefficient.
        - std_err uses the sample standard deviation (ddof=1) of the i.i.d. quantities
          being averaged, divided by sqrt(number of those quantities).

    To implement:
        - Under the risk-neutral measure S_T is lognormal; sample it exactly in one step
          from standard normals (no time-stepping is needed for a European payoff). Work
          out the drift correction yourself (Hull, "Monte Carlo simulation" section).
        - Price = e^{-rT} * mean(payoff).
        - Antithetic: with pairs (Z, -Z), the two payoffs in a pair are *not* independent.
          Average within each pair first, then compute the mean and std_err over the
          n_paths / 2 pair averages. Getting this wrong understates std_err.
        - Control variate: X = e^{-rT} S_T has known mean S e^{-qT}. Estimate the optimal
          coefficient b from the sample (cov(Y, X) / var(X)) and use Y - b (X - E[X]).
          When combined with antithetic, apply it to the pair averages.

    Raises:
        ValueError: if `opt` is American, n_paths < 2, or n_paths is odd with antithetic.
    """
    require_european(opt, "price_mc")
    if n_paths < 2:
        raise ValueError(f"n_paths must be >= 2, got {n_paths}")
    if antithetic and n_paths % 2 != 0:
        raise ValueError(f"n_paths must be even with antithetic=True, got {n_paths}")
    raise NotImplementedError("price_mc: see docstring and docs/conventions.md")
