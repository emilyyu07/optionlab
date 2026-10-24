"""Black-Scholes-Merton: reference prices, Greeks vs finite differences, implied vol."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable
from typing import Any

import pytest

from optionlab import ExerciseStyle, Market, Option, OptionKind, greeks_bs, implied_vol, price_bs

from conftest import (
    FD_ABS_BUMP,
    FD_SPOT_REL_BUMP,
    GAMMA_FD_ATOL,
    GAMMA_FD_RTOL,
    GREEK_FD_ATOL,
    GREEK_FD_RTOL,
    IV_ATOL,
    discounted_forward_payoff,
    intrinsic,
    market_from,
    pending,
    ref_tol,
    reference_cases,
)

# (spot, strike, rate, div_yield, vol, expiry): ATM, ITM/OTM, dividends, zero rate, long-dated.
GREEK_CASES = [
    (100.0, 100.0, 0.05, 0.00, 0.20, 1.00),
    (100.0, 90.0, 0.03, 0.02, 0.35, 0.25),
    (50.0, 60.0, 0.10, 0.05, 0.15, 2.00),
    (100.0, 110.0, 0.00, 0.00, 0.50, 0.50),
]


def _case(kind: OptionKind, case: tuple[float, ...]) -> tuple[Option, Market]:
    spot, strike, rate, div_yield, vol, expiry = case
    return Option(kind, strike, expiry), Market(spot, rate, vol, div_yield)


# ---------------------------------------------------------------------------
# Reference values
# ---------------------------------------------------------------------------


@pending("price_bs")
@pytest.mark.parametrize("entry", reference_cases("black_scholes"))
def test_reference_values(entry: dict[str, Any]) -> None:
    mkt = market_from(entry["inputs"])
    strike, expiry = entry["inputs"]["strike"], entry["inputs"]["expiry_years"]
    tol = ref_tol(entry["decimals"])
    for kind in OptionKind:
        expected = entry["expected"][kind.value]
        assert price_bs(Option(kind, strike, expiry), mkt) == pytest.approx(expected, abs=tol)


# ---------------------------------------------------------------------------
# Greeks vs central finite differences of price_bs
# ---------------------------------------------------------------------------


def _central_diff(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)


@pending("price_bs / greeks_bs")
@pytest.mark.parametrize(("kind", "case"), list(itertools.product(OptionKind, GREEK_CASES)))
def test_greeks_match_finite_differences(kind: OptionKind, case: tuple[float, ...]) -> None:
    opt, mkt = _case(kind, case)
    greeks = greeks_bs(opt, mkt)

    def at_spot(s: float) -> float:
        return price_bs(opt, dataclasses.replace(mkt, spot=s))

    def at_vol(v: float) -> float:
        return price_bs(opt, dataclasses.replace(mkt, vol=v))

    def at_rate(r: float) -> float:
        return price_bs(opt, dataclasses.replace(mkt, rate=r))

    def at_expiry(t: float) -> float:
        return price_bs(dataclasses.replace(opt, expiry_years=t), mkt)

    h_spot = FD_SPOT_REL_BUMP * mkt.spot
    fd_gamma = (at_spot(mkt.spot + h_spot) - 2 * at_spot(mkt.spot) + at_spot(mkt.spot - h_spot)) / (
        h_spot**2
    )
    first_order = {
        "delta": _central_diff(at_spot, mkt.spot, h_spot),
        "vega": _central_diff(at_vol, mkt.vol, FD_ABS_BUMP),
        "rho": _central_diff(at_rate, mkt.rate, FD_ABS_BUMP),
        # Theta is per year of calendar time passing, i.e. -dV/dT.
        "theta": -_central_diff(at_expiry, opt.expiry_years, FD_ABS_BUMP),
    }
    for name, fd_value in first_order.items():
        assert getattr(greeks, name) == pytest.approx(
            fd_value, rel=GREEK_FD_RTOL, abs=GREEK_FD_ATOL
        ), name
    assert greeks.gamma == pytest.approx(fd_gamma, rel=GAMMA_FD_RTOL, abs=GAMMA_FD_ATOL)


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

IV_GRID = list(
    itertools.product(
        OptionKind,
        [90.0, 100.0, 110.0],  # strike, with spot fixed at 100
        [0.25, 1.0, 2.0],  # expiry
        [0.10, 0.20, 0.40, 0.80],  # vol to recover
    )
)


@pending("price_bs / implied_vol")
@pytest.mark.parametrize(("kind", "strike", "expiry", "vol"), IV_GRID)
def test_implied_vol_round_trip(kind: OptionKind, strike: float, expiry: float, vol: float) -> None:
    opt = Option(kind, strike, expiry)
    mkt = Market(spot=100.0, rate=0.05, vol=vol, div_yield=0.02)
    price = price_bs(opt, mkt)
    assert implied_vol(price, opt, mkt) == pytest.approx(vol, abs=IV_ATOL)


@pending("implied_vol")
@pytest.mark.parametrize("kind", OptionKind)
def test_implied_vol_rejects_price_above_upper_bound(kind: OptionKind) -> None:
    # A call is worth at most S e^{-qT}; a put at most K e^{-rT}. Both are < 1000 here.
    opt = Option(kind, strike=100.0, expiry_years=1.0)
    mkt = Market(spot=100.0, rate=0.05, vol=0.2)
    with pytest.raises(ValueError, match=r"(?i)arbitrage|bound"):
        implied_vol(1000.0, opt, mkt)


def test_implied_vol_rejects_non_finite_price() -> None:
    opt = Option(OptionKind.CALL, 100.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        implied_vol(float("nan"), opt, Market(100.0, 0.05, 0.2))


def test_implied_vol_rejects_zero_expiry() -> None:
    opt = Option(OptionKind.CALL, 100.0, 0.0)
    with pytest.raises(ValueError, match="T == 0"):
        implied_vol(1.0, opt, Market(100.0, 0.05, 0.2))


# ---------------------------------------------------------------------------
# Edge cases and input validation
# ---------------------------------------------------------------------------


@pending("price_bs")
@pytest.mark.parametrize("kind", OptionKind)
@pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
def test_zero_expiry_returns_intrinsic(kind: OptionKind, spot: float) -> None:
    opt = Option(kind, strike=100.0, expiry_years=0.0)
    assert price_bs(opt, Market(spot, 0.05, 0.2, 0.01)) == intrinsic(opt, spot)


@pending("price_bs")
@pytest.mark.parametrize("kind", OptionKind)
@pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
def test_zero_vol_returns_discounted_forward_payoff(kind: OptionKind, spot: float) -> None:
    opt = Option(kind, strike=100.0, expiry_years=1.0)
    mkt = Market(spot, rate=0.05, vol=0.0, div_yield=0.01)
    assert price_bs(opt, mkt) == pytest.approx(discounted_forward_payoff(opt, mkt))


@pytest.mark.parametrize(
    "call",
    [
        lambda o, m: price_bs(o, m),
        lambda o, m: greeks_bs(o, m),
        lambda o, m: implied_vol(5.0, o, m),
    ],
    ids=["price_bs", "greeks_bs", "implied_vol"],
)
def test_rejects_american(call: Callable[[Option, Market], object]) -> None:
    opt = Option(OptionKind.PUT, 100.0, 1.0, style=ExerciseStyle.AMERICAN)
    with pytest.raises(ValueError, match="European"):
        call(opt, Market(100.0, 0.05, 0.2))
