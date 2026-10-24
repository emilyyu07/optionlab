"""Cox-Ross-Rubinstein (CRR) binomial tree for European and American options.

Units and conventions: docs/conventions.md.
"""

from __future__ import annotations

from optionlab.types import Market, Option


def price_binomial(opt: Option, mkt: Market, steps: int) -> float:
    """Price a European or American option on a recombining CRR binomial tree.

    Args:
        opt: Call or put, European or American.
        mkt: Spot, flat continuously compounded rate r, vol sigma, dividend yield q.
        steps: Number of time steps N >= 1; dt = T / N.

    Returns:
        Price per share (no contract multiplier).

    Conventions (docs/conventions.md):
        - CRR parametrisation: up factor u = exp(sigma * sqrt(dt)), down factor d = 1 / u.
        - The risk-neutral up probability p is chosen so the expected one-step growth of
          the stock matches the forward, e^{(r - q) dt}. Derive it; then check it lies in
          [0, 1] and think about when it does not.
        - T == 0: return intrinsic value (for any steps).
        - sigma == 0: the tree degenerates (u == d). For European options return the
          discounted forward payoff, as in price_bs. Decide and document what an American
          option should return here.

    To implement:
        1. Build terminal stock prices S u^j d^(N-j) for j = 0..N and their payoffs.
        2. Step backwards: each node's continuation value is the discounted risk-neutral
           expectation of its two children.
        3. American only: at each node take max(continuation, immediate exercise value).
        Vectorize across a time slice with numpy; a pure Python double loop is fine for
        learning but slow at the 2000 steps used in the convergence tests.
        (Hull, "Binomial Trees" and "Numerical Procedures" chapters.)

    Raises:
        ValueError: if steps < 1.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    raise NotImplementedError("price_binomial: see docstring and docs/conventions.md")
