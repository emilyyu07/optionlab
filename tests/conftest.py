"""Shared tolerances, markers, strategies and reference-data loading.

Every numerical tolerance used by the test suite lives here, each with a one-line
justification. If a test fails, fix the code before loosening a tolerance; if a
tolerance genuinely has to change, update its justification too.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from hypothesis import settings
from hypothesis import strategies as st

from optionlab import ExerciseStyle, Market, Option, OptionKind

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

# Exact identities (parity, bounds, monotonicity) only lose float rounding; prices are <= ~1e3.
IDENTITY_ATOL = 1e-9

# Central differences: relative spot bump; truncation O(h^2) ~ 1e-8, rounding eps/h ~ 1e-12.
FD_SPOT_REL_BUMP = 1e-4
# Absolute bump for vol, rate and expiry; same O(h^2) vs rounding trade-off as above.
FD_ABS_BUMP = 1e-5
# First-order Greeks vs central differences: observed error ~1e-8, so 1e-6 leaves headroom.
GREEK_FD_RTOL = 1e-6
GREEK_FD_ATOL = 1e-7
# Gamma is a second difference, which amplifies rounding by 1/h^2; observed error ~4e-8 relative.
GAMMA_FD_RTOL = 1e-5
GAMMA_FD_ATOL = 1e-8

# Implied vol must recover sigma to this accuracy (project requirement).
IV_ATOL = 1e-8

# Steps for the "converged" binomial price in convergence tests.
BINOMIAL_FINE_STEPS = 2000
# CRR error is O(1/N) with oscillation; observed <= 2.6e-3 at N=2000 for S ~ 100, T <= 3.
BINOMIAL_CONV_ATOL = 5e-3
# Steps for property tests on the tree (speed matters more than accuracy there).
BINOMIAL_PROPERTY_STEPS = 100
# American and European calls with q = 0 share the same tree, so only rounding separates them.
BINOMIAL_SAME_TREE_ATOL = 1e-10

# A correct estimator lands within 3 standard errors of the truth ~99.7% of the time.
MC_Z = 3.0
# Paths for the accuracy and variance-reduction tests: std_err ~ 0.05 for S ~ 100.
MC_PATHS = 100_000
# The std_err estimate itself is noisy (~1/sqrt(2N) relative), so allow 15% on the 1/sqrt(N) ratio.
MC_SE_SCALING_RTOL = 0.15

# QuantLib's blackFormula and a scipy-based formula agree to ~1e-14; 1e-10 is pure headroom.
QL_BS_ATOL = 1e-10
# QuantLib's BlackCalculator Greeks vs ours: both analytic, observed agreement ~1e-8.
QL_GREEK_ATOL = 1e-6
# QuantLib's CRR uses a log-space drift for p; both trees are O(1/N) off: ~7e-3 apart at N=1000.
QL_AMERICAN_ATOL = 1e-2
QL_AMERICAN_STEPS = 1000


def ref_tol(decimals: int) -> float:
    """Half a unit in the last printed decimal place of a textbook value."""
    return 0.5 * 10.0**-decimals


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def pending(model: str) -> pytest.MarkDecorator:
    """Mark a test that depends on an unimplemented model.

    The test is expected to fail *only* with NotImplementedError. Any other failure (a
    wrong number, a crash) still fails the run. Remove the marker once `model` is
    implemented, so a pass is reported as a pass rather than XPASS.
    """
    return pytest.mark.xfail(
        raises=NotImplementedError,
        strict=False,
        reason=f"{model} not implemented yet: remove this marker once it is",
    )


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------

# No deadline: a tree with hundreds of steps can legitimately take longer than 200 ms.
settings.register_profile("optionlab", deadline=None, max_examples=100)
settings.load_profile("optionlab")


@st.composite
def contracts(
    draw: st.DrawFn,
    kind: OptionKind | None = None,
    style: ExerciseStyle = ExerciseStyle.EUROPEAN,
    zero_dividend: bool = False,
    non_negative_rate: bool = False,
) -> tuple[Option, Market]:
    """Draw a realistic (Option, Market) pair; ranges avoid numerically degenerate corners."""
    spot = draw(st.floats(1.0, 1000.0))
    moneyness = draw(st.floats(0.5, 2.0))
    rate = draw(st.floats(0.0 if non_negative_rate else -0.02, 0.15))
    div_yield = 0.0 if zero_dividend else draw(st.floats(0.0, 0.10))
    vol = draw(st.floats(0.01, 1.5))
    expiry = draw(st.floats(0.01, 5.0))
    opt_kind = kind if kind is not None else draw(st.sampled_from(OptionKind))
    return (
        Option(opt_kind, strike=spot * moneyness, expiry_years=expiry, style=style),
        Market(spot=spot, rate=rate, vol=vol, div_yield=div_yield),
    )


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

_REFERENCE_PATH = Path(__file__).parent / "data" / "reference_values.json"


def reference_cases(section: str) -> list[Any]:
    """Load one section of reference_values.json as pytest params; TODO entries are skipped."""
    entries: list[dict[str, Any]] = json.loads(_REFERENCE_PATH.read_text())[section]
    params = []
    for entry in entries:
        marks = []
        if entry["status"] == "TODO":
            marks.append(pytest.mark.skip(reason=f"TODO reference value: {entry['source']}"))
        params.append(pytest.param(entry, id=entry["id"], marks=marks))
    return params


def market_from(inputs: dict[str, Any]) -> Market:
    return Market(
        spot=inputs["spot"],
        rate=inputs["rate"],
        vol=inputs["vol"],
        div_yield=inputs["div_yield"],
    )


# ---------------------------------------------------------------------------
# Small helpers used by several test modules
# ---------------------------------------------------------------------------


def intrinsic(opt: Option, spot: float) -> float:
    if opt.kind is OptionKind.CALL:
        return max(spot - opt.strike, 0.0)
    return max(opt.strike - spot, 0.0)


def discounted_forward_payoff(opt: Option, mkt: Market) -> float:
    """The sigma = 0 price: intrinsic value of the forward, discounted (docs/conventions.md)."""
    t = opt.expiry_years
    pv_spot = mkt.spot * math.exp(-mkt.div_yield * t)
    pv_strike = opt.strike * math.exp(-mkt.rate * t)
    if opt.kind is OptionKind.CALL:
        return max(pv_spot - pv_strike, 0.0)
    return max(pv_strike - pv_spot, 0.0)
