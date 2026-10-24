"""Black-Scholes-Merton: closed-form price, Greeks and implied volatility.

European exercise only. Units and conventions: docs/conventions.md.
"""

from __future__ import annotations

import math

from optionlab.types import Greeks, Market, Option, require_european


def price_bs(opt: Option, mkt: Market) -> float:
    """Price a European option under Black-Scholes-Merton with a continuous dividend yield.

    Args:
        opt: European call or put. American options raise ValueError.
        mkt: Spot, flat continuously compounded rate r, vol sigma, dividend yield q.

    Returns:
        Price per share (no contract multiplier).

    Conventions (docs/conventions.md):
        - T = opt.expiry_years (ACT/365); r and q continuously compounded; sigma annualized.
        - T == 0: return intrinsic value max(S - K, 0) for a call, max(K - S, 0) for a put.
        - sigma == 0: return the discounted forward payoff, i.e. intrinsic value of the
          forward: max(S e^{-qT} - K e^{-rT}, 0) for a call, mirrored for a put.
          (Check for yourself that this is the sigma -> 0 limit of the general formula.)

    To implement:
        The Black-Scholes-Merton formula with continuous dividend yield (Merton 1973;
        Hull, "The Black-Scholes-Merton Model" chapter, and the chapter on options on
        indices for the q term). It is written in terms of two quantities d1 and d2 and
        the standard normal CDF N(.). Handle the two edge cases above *before* the
        general formula: both make d1/d2 divide by zero.

        Suggested numerics: use scipy.special.ndtr for N(.), and write the put via its own
        formula rather than via parity, so that test_properties.py's parity test is a
        genuine check rather than a tautology.

    Raises:
        ValueError: if `opt` is American.
    """
    require_european(opt, "price_bs")
    raise NotImplementedError("price_bs: see docstring and docs/conventions.md")


def greeks_bs(opt: Option, mkt: Market) -> Greeks:
    """Analytical Black-Scholes-Merton Greeks for a European option.

    Args:
        opt: European call or put. American options raise ValueError.
        mkt: Market state, as in price_bs.

    Returns:
        Greeks in raw analytical units (docs/conventions.md):
            delta = dV/dS
            gamma = d2V/dS2
            vega  = dV/dsigma, per 1.00 of vol (divide by 100 for "per vol point")
            theta = dV/dt per year of calendar time, i.e. -dV/dT (divide by 365 for per day)
            rho   = dV/dr, per 1.00 of rate (divide by 10_000 for per basis point)

    To implement:
        Differentiate your price_bs formula analytically with respect to S, sigma, T and
        r. Gamma and vega are identical for calls and puts; delta, theta and rho differ.
        The q term changes delta (an e^{-qT} factor) and adds a term to theta.
        tests/test_black_scholes.py checks every Greek against central finite
        differences of price_bs, so derive them, then let the tests catch mistakes.

        Edge cases (T == 0 or sigma == 0) are not tested for Greeks; decide what you want
        to return there and document it (the derivatives are discontinuous at the strike).

    Raises:
        ValueError: if `opt` is American.
    """
    require_european(opt, "greeks_bs")
    raise NotImplementedError("greeks_bs: see docstring and docs/conventions.md")


def implied_vol(price: float, opt: Option, mkt: Market) -> float:
    """Recover the Black-Scholes-Merton volatility that reproduces `price`.

    Args:
        price: Observed option price per share.
        opt: European call or put with T > 0. American options raise ValueError.
        mkt: Market state; mkt.vol is ignored (it is the unknown).

    Returns:
        sigma >= 0 such that price_bs(opt, Market(spot, rate, sigma, div_yield)) == price,
        to within 1e-8 in sigma on the test grid.

    To implement:
        A one-dimensional root find of f(sigma) = price_bs(sigma) - price. f is strictly
        increasing in sigma (vega > 0 for T > 0), so a bracketing method such as
        scipy.optimize.brentq on a wide bracket is robust; Newton's method using
        greeks_bs(...).vega converges faster but needs a fallback when vega is tiny.
        Use tight solver tolerances: the tests demand 1e-8 in sigma.

        Before solving, check the no-arbitrage bounds (docs/conventions.md):
        a price outside them has no implied vol and must raise ValueError.

    Raises:
        ValueError: if price is not finite, if `opt` is American, if T == 0, or if the
            price violates the no-arbitrage bounds.
    """
    if not math.isfinite(price):
        raise ValueError(f"price must be finite, got {price!r}")
    require_european(opt, "implied_vol")
    if opt.expiry_years == 0.0:
        raise ValueError("implied vol is undefined at expiry (T == 0)")
    raise NotImplementedError("implied_vol: see docstring and docs/conventions.md")
