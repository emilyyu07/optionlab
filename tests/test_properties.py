"""Model-free properties: input validation, parity, no-arbitrage bounds, monotonicity, Greeks signs.

These hold for any correct pricer, so they are checked on random inputs with hypothesis.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from optionlab import Market, Option, OptionKind, greeks_bs, price_bs

from conftest import IDENTITY_ATOL, contracts, pending

finite = st.floats(allow_nan=False, allow_infinity=False)
non_finite = st.sampled_from([math.nan, math.inf, -math.inf])
# Strictly negative. Not negative: hypothesis rewrites that into a
# bound that still admits -0.0, which is a valid (zero) expiry or vol.
negative = st.floats(max_value=-math.ulp(0.0), allow_infinity=False)

# ---------------------------------------------------------------------------
# Input validation (implemented in types.py, so these pass today)
# ---------------------------------------------------------------------------


@given(spot=st.one_of(finite.filter(lambda x: x <= 0), non_finite))
def test_market_rejects_invalid_spot(spot: float) -> None:
    with pytest.raises(ValueError, match="spot"):
        Market(spot=spot, rate=0.05, vol=0.2)


@given(vol=st.one_of(negative, non_finite))
def test_market_rejects_invalid_vol(vol: float) -> None:
    with pytest.raises(ValueError, match="vol"):
        Market(spot=100.0, rate=0.05, vol=vol)


@given(rate=non_finite)
def test_market_rejects_non_finite_rate(rate: float) -> None:
    with pytest.raises(ValueError, match="rate"):
        Market(spot=100.0, rate=rate, vol=0.2)


@given(strike=st.one_of(finite.filter(lambda x: x <= 0), non_finite))
def test_option_rejects_invalid_strike(strike: float) -> None:
    with pytest.raises(ValueError, match="strike"):
        Option(OptionKind.CALL, strike=strike, expiry_years=1.0)


@given(expiry=st.one_of(negative, non_finite))
def test_option_rejects_invalid_expiry(expiry: float) -> None:
    with pytest.raises(ValueError, match="expiry_years"):
        Option(OptionKind.CALL, strike=100.0, expiry_years=expiry)


def test_boundary_values_are_accepted() -> None:
    Option(OptionKind.PUT, strike=1e-12, expiry_years=0.0)
    Market(spot=1e-12, rate=-0.01, vol=0.0, div_yield=-0.01)


# ---------------------------------------------------------------------------
# Black-Scholes properties
# ---------------------------------------------------------------------------


def _discounts(opt: Option, mkt: Market) -> tuple[float, float]:
    """(S e^{-qT}, K e^{-rT}): present values of the stock and of the strike."""
    t = opt.expiry_years
    return mkt.spot * math.exp(-mkt.div_yield * t), opt.strike * math.exp(-mkt.rate * t)


@pending("price_bs")
@given(contracts(kind=OptionKind.CALL))
def test_put_call_parity(contract: tuple[Option, Market]) -> None:
    call, mkt = contract
    put = dataclasses.replace(call, kind=OptionKind.PUT)
    pv_spot, pv_strike = _discounts(call, mkt)
    lhs = price_bs(call, mkt) - price_bs(put, mkt)
    assert lhs == pytest.approx(pv_spot - pv_strike, abs=IDENTITY_ATOL)


@pending("price_bs")
@given(contracts())
def test_no_arbitrage_bounds(contract: tuple[Option, Market]) -> None:
    opt, mkt = contract
    pv_spot, pv_strike = _discounts(opt, mkt)
    price = price_bs(opt, mkt)
    if opt.kind is OptionKind.CALL:
        lower, upper = max(pv_spot - pv_strike, 0.0), pv_spot
    else:
        lower, upper = max(pv_strike - pv_spot, 0.0), pv_strike
    assert lower - IDENTITY_ATOL <= price <= upper + IDENTITY_ATOL


@pending("price_bs")
@given(contracts(), st.floats(1.001, 1.5))
def test_monotone_in_strike(contract: tuple[Option, Market], bump: float) -> None:
    """Calls get cheaper and puts dearer as the strike rises."""
    opt, mkt = contract
    higher = dataclasses.replace(opt, strike=opt.strike * bump)
    diff = price_bs(higher, mkt) - price_bs(opt, mkt)
    if opt.kind is OptionKind.CALL:
        assert diff <= IDENTITY_ATOL
    else:
        assert diff >= -IDENTITY_ATOL


@pending("price_bs")
@given(contracts(), st.floats(0.001, 0.5))
def test_monotone_in_vol(contract: tuple[Option, Market], bump: float) -> None:
    opt, mkt = contract
    higher = dataclasses.replace(mkt, vol=mkt.vol + bump)
    assert price_bs(opt, higher) >= price_bs(opt, mkt) - IDENTITY_ATOL


@pending("price_bs")
@given(contracts(kind=OptionKind.CALL, zero_dividend=True, non_negative_rate=True))
def test_call_monotone_in_expiry_without_dividends(contract: tuple[Option, Market]) -> None:
    """A European call with q = 0 and r >= 0 never loses value with more time.

    This is *not* true for European puts (with r > 0 a deep ITM put can be worth less
    with more time, because the strike is received later) nor for calls with large q.
    """
    opt, mkt = contract
    longer = dataclasses.replace(opt, expiry_years=opt.expiry_years * 1.5)
    assert price_bs(longer, mkt) >= price_bs(opt, mkt) - IDENTITY_ATOL


@pending("greeks_bs")
@given(contracts())
def test_delta_bounds(contract: tuple[Option, Market]) -> None:
    """Call delta lies in [0, e^{-qT}]; put delta in [-e^{-qT}, 0]."""
    opt, mkt = contract
    cap = math.exp(-mkt.div_yield * opt.expiry_years)
    delta = greeks_bs(opt, mkt).delta
    if opt.kind is OptionKind.CALL:
        assert -IDENTITY_ATOL <= delta <= cap + IDENTITY_ATOL
    else:
        assert -cap - IDENTITY_ATOL <= delta <= IDENTITY_ATOL


@pending("greeks_bs")
@given(contracts())
def test_gamma_and_vega_non_negative(contract: tuple[Option, Market]) -> None:
    opt, mkt = contract
    greeks = greeks_bs(opt, mkt)
    assert greeks.gamma >= -IDENTITY_ATOL
    assert greeks.vega >= -IDENTITY_ATOL
