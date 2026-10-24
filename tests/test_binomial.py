"""CRR binomial tree: convergence to Black-Scholes and American exercise properties."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from hypothesis import given

from optionlab import ExerciseStyle, Market, Option, OptionKind, price_binomial, price_bs

from conftest import (
    BINOMIAL_CONV_ATOL,
    BINOMIAL_FINE_STEPS,
    BINOMIAL_PROPERTY_STEPS,
    BINOMIAL_SAME_TREE_ATOL,
    IDENTITY_ATOL,
    contracts,
    discounted_forward_payoff,
    intrinsic,
    market_from,
    pending,
    ref_tol,
    reference_cases,
)

# (spot, strike, rate, div_yield, vol, expiry)
CONVERGENCE_CASES = [
    (100.0, 100.0, 0.05, 0.00, 0.20, 1.00),
    (100.0, 90.0, 0.03, 0.02, 0.35, 0.25),
    (50.0, 60.0, 0.10, 0.05, 0.15, 2.00),
    (100.0, 120.0, 0.02, 0.00, 0.60, 3.00),
]


@pending("price_binomial")
@pytest.mark.parametrize("entry", reference_cases("binomial"))
def test_reference_values(entry: dict[str, Any]) -> None:
    mkt = market_from(entry["inputs"])
    style = ExerciseStyle(entry["style"])
    tol = ref_tol(entry["decimals"])
    for kind_name, expected in entry["expected"].items():
        opt = Option(
            OptionKind(kind_name),
            entry["inputs"]["strike"],
            entry["inputs"]["expiry_years"],
            style=style,
        )
        assert price_binomial(opt, mkt, entry["steps"]) == pytest.approx(expected, abs=tol)


@pending("price_bs / price_binomial")
@pytest.mark.parametrize("kind", OptionKind)
@pytest.mark.parametrize("case", CONVERGENCE_CASES)
def test_european_converges_to_black_scholes(kind: OptionKind, case: tuple[float, ...]) -> None:
    spot, strike, rate, div_yield, vol, expiry = case
    opt = Option(kind, strike, expiry)
    mkt = Market(spot, rate, vol, div_yield)
    exact = price_bs(opt, mkt)
    coarse_err = abs(price_binomial(opt, mkt, 50) - exact)
    fine_err = abs(price_binomial(opt, mkt, BINOMIAL_FINE_STEPS) - exact)
    assert fine_err < BINOMIAL_CONV_ATOL
    # CRR error oscillates with N, so only compare widely separated step counts.
    assert fine_err < coarse_err


@pending("price_binomial")
@given(contracts(kind=OptionKind.PUT))
def test_american_put_dominates_european_and_intrinsic(contract: tuple[Option, Market]) -> None:
    european, mkt = contract
    american = dataclasses.replace(european, style=ExerciseStyle.AMERICAN)
    amer = price_binomial(american, mkt, BINOMIAL_PROPERTY_STEPS)
    assert amer >= price_binomial(european, mkt, BINOMIAL_PROPERTY_STEPS) - IDENTITY_ATOL
    assert amer >= intrinsic(american, mkt.spot) - IDENTITY_ATOL


@pending("price_binomial")
@given(contracts(kind=OptionKind.CALL, zero_dividend=True, non_negative_rate=True))
def test_american_call_equals_european_without_dividends(contract: tuple[Option, Market]) -> None:
    """With q = 0 and r >= 0, early exercise of a call is never optimal (Merton 1973)."""
    european, mkt = contract
    american = dataclasses.replace(european, style=ExerciseStyle.AMERICAN)
    assert price_binomial(american, mkt, BINOMIAL_PROPERTY_STEPS) == pytest.approx(
        price_binomial(european, mkt, BINOMIAL_PROPERTY_STEPS), abs=BINOMIAL_SAME_TREE_ATOL
    )


@pending("price_binomial")
@pytest.mark.parametrize("style", ExerciseStyle)
@pytest.mark.parametrize("kind", OptionKind)
@pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
def test_zero_expiry_returns_intrinsic(style: ExerciseStyle, kind: OptionKind, spot: float) -> None:
    opt = Option(kind, strike=100.0, expiry_years=0.0, style=style)
    assert price_binomial(opt, Market(spot, 0.05, 0.2, 0.01), steps=10) == intrinsic(opt, spot)


@pending("price_binomial")
@pytest.mark.parametrize("kind", OptionKind)
@pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
def test_european_zero_vol_returns_discounted_forward_payoff(kind: OptionKind, spot: float) -> None:
    opt = Option(kind, strike=100.0, expiry_years=1.0)
    mkt = Market(spot, rate=0.05, vol=0.0, div_yield=0.01)
    assert price_binomial(opt, mkt, steps=10) == pytest.approx(discounted_forward_payoff(opt, mkt))


@pytest.mark.parametrize("steps", [0, -1])
def test_rejects_non_positive_steps(steps: int) -> None:
    with pytest.raises(ValueError, match="steps"):
        price_binomial(Option(OptionKind.CALL, 100.0, 1.0), Market(100.0, 0.05, 0.2), steps)
