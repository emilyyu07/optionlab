"""Cross-model study: optionlab vs QuantLib on a random grid of contracts.

QuantLib is a test-only oracle (the `oracle` dependency group). This module is skipped
automatically when it is not installed: `uv sync --group oracle` to enable it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from optionlab import (
    ExerciseStyle,
    Market,
    Option,
    OptionKind,
    greeks_bs,
    implied_vol,
    price_binomial,
    price_bs,
)

from conftest import (
    IV_ATOL,
    QL_AMERICAN_ATOL,
    QL_AMERICAN_STEPS,
    QL_BS_ATOL,
    QL_GREEK_ATOL,
    pending,
)

ql = pytest.importorskip("QuantLib")


def _random_grid(n: int, seed: int) -> list[tuple[Option, Market]]:
    """Random contracts with expiries on whole days, so ACT/365 dates map exactly to years."""
    rng = np.random.Generator(np.random.PCG64(seed))
    grid = []
    for _ in range(n):
        spot = float(rng.uniform(50.0, 150.0))
        days = int(rng.integers(30, 731))
        opt = Option(
            OptionKind.CALL if rng.random() < 0.5 else OptionKind.PUT,
            strike=spot * float(rng.uniform(0.8, 1.25)),
            expiry_years=days / 365.0,
        )
        mkt = Market(
            spot=spot,
            rate=float(rng.uniform(0.0, 0.08)),
            vol=float(rng.uniform(0.10, 0.60)),
            div_yield=float(rng.uniform(0.0, 0.05)),
        )
        grid.append((opt, mkt))
    return grid


GRID = _random_grid(25, seed=12_345)
GRID_IDS = [f"case{i}" for i in range(len(GRID))]


def _ql_type(kind: OptionKind) -> Any:
    return ql.Option.Call if kind is OptionKind.CALL else ql.Option.Put


def _black_calculator(opt: Option, mkt: Market) -> Any:
    """QuantLib's closed-form Black calculator, fed forward, total stdev and discount factor."""
    t = opt.expiry_years
    forward = mkt.spot * math.exp((mkt.rate - mkt.div_yield) * t)
    return ql.BlackCalculator(
        ql.PlainVanillaPayoff(_ql_type(opt.kind), opt.strike),
        forward,
        mkt.vol * math.sqrt(t),
        math.exp(-mkt.rate * t),
    )


def _ql_american(opt: Option, mkt: Market, steps: int) -> float:
    """QuantLib CRR tree price with flat curves on an ACT/365 day count."""
    today = ql.Date(15, ql.January, 2025)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    maturity = today + round(opt.expiry_years * 365)
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(mkt.spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, mkt.div_yield, day_count)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, mkt.rate, day_count)),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), mkt.vol, day_count)
        ),
    )
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(_ql_type(opt.kind), opt.strike),
        ql.AmericanExercise(today, maturity),
    )
    option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", steps))
    return float(option.NPV())


@pending("price_bs")
@pytest.mark.parametrize(("opt", "mkt"), GRID, ids=GRID_IDS)
def test_price_bs_matches_quantlib(opt: Option, mkt: Market) -> None:
    expected = float(_black_calculator(opt, mkt).value())
    assert price_bs(opt, mkt) == pytest.approx(expected, abs=QL_BS_ATOL)


@pending("greeks_bs")
@pytest.mark.parametrize(("opt", "mkt"), GRID, ids=GRID_IDS)
def test_greeks_bs_match_quantlib(opt: Option, mkt: Market) -> None:
    # QuantLib's BlackCalculator uses the same raw units: vega and rho per 1.00, theta per year.
    calc = _black_calculator(opt, mkt)
    t = opt.expiry_years
    greeks = greeks_bs(opt, mkt)
    expected = {
        "delta": calc.delta(mkt.spot),
        "gamma": calc.gamma(mkt.spot),
        "vega": calc.vega(t),
        "theta": calc.theta(mkt.spot, t),
        "rho": calc.rho(t),
    }
    for name, value in expected.items():
        assert getattr(greeks, name) == pytest.approx(float(value), abs=QL_GREEK_ATOL), name


@pending("implied_vol")
@pytest.mark.parametrize(("opt", "mkt"), GRID, ids=GRID_IDS)
def test_implied_vol_recovers_sigma_from_quantlib_price(opt: Option, mkt: Market) -> None:
    ql_price = float(_black_calculator(opt, mkt).value())
    assert implied_vol(ql_price, opt, mkt) == pytest.approx(mkt.vol, abs=IV_ATOL)


@pending("price_binomial")
@pytest.mark.parametrize(("opt", "mkt"), GRID[:10], ids=GRID_IDS[:10])
def test_american_binomial_matches_quantlib(opt: Option, mkt: Market) -> None:
    american = Option(opt.kind, opt.strike, opt.expiry_years, ExerciseStyle.AMERICAN)
    ours = price_binomial(american, mkt, QL_AMERICAN_STEPS)
    assert ours == pytest.approx(
        _ql_american(american, mkt, QL_AMERICAN_STEPS), abs=QL_AMERICAN_ATOL
    )
