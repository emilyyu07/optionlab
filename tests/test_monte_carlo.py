"""Monte Carlo: accuracy vs Black-Scholes, reproducibility, std_err and variance reduction."""

from __future__ import annotations

import itertools
import math

import pytest

from optionlab import ExerciseStyle, Market, MCResult, Option, OptionKind, price_bs, price_mc

from conftest import (
    MC_PATHS,
    MC_SE_SCALING_RTOL,
    MC_Z,
    discounted_forward_payoff,
    intrinsic,
    pending,
)

# (spot, strike, rate, div_yield, vol, expiry)
MC_CASES = [
    (100.0, 100.0, 0.05, 0.00, 0.20, 1.00),
    (100.0, 90.0, 0.03, 0.02, 0.35, 0.25),
    (50.0, 60.0, 0.10, 0.05, 0.15, 2.00),
]
# Fixed seeds keep the suite deterministic. With 24 checks at 3 standard errors, an
# unlucky seed is possible (~0.3% per check): if exactly one case fails by a small margin,
# try other seeds before assuming a bug, and never "fix" it by picking a seed that passes.
SEED = 20_240_601
# Every (antithetic, control_variate) combination.
VR_FLAGS = list(itertools.product([False, True], repeat=2))
ATM_CALL = Option(OptionKind.CALL, 100.0, 1.0)
MKT = Market(spot=100.0, rate=0.05, vol=0.2)


def _case(kind: OptionKind, case: tuple[float, ...]) -> tuple[Option, Market]:
    spot, strike, rate, div_yield, vol, expiry = case
    return Option(kind, strike, expiry), Market(spot, rate, vol, div_yield)


# ---------------------------------------------------------------------------
# MCResult (implemented in types.py, so these pass today)
# ---------------------------------------------------------------------------


def test_ci95_is_symmetric_196_standard_errors() -> None:
    lo, hi = MCResult(price=10.0, std_err=0.5).ci95
    assert (lo + hi) / 2 == pytest.approx(10.0)
    assert (hi - lo) / 2 == pytest.approx(1.96 * 0.5, rel=1e-3)


@pytest.mark.parametrize(("price", "std_err"), [(1.0, -0.1), (math.nan, 0.1), (1.0, math.inf)])
def test_mc_result_rejects_invalid(price: float, std_err: float) -> None:
    with pytest.raises(ValueError, match=r"price|std_err"):
        MCResult(price, std_err)


# ---------------------------------------------------------------------------
# Accuracy and reproducibility
# ---------------------------------------------------------------------------


@pending("price_bs / price_mc")
@pytest.mark.parametrize(("antithetic", "control_variate"), VR_FLAGS)
@pytest.mark.parametrize("kind", OptionKind)
@pytest.mark.parametrize("case", MC_CASES)
def test_within_three_standard_errors_of_black_scholes(
    case: tuple[float, ...], kind: OptionKind, antithetic: bool, control_variate: bool
) -> None:
    opt, mkt = _case(kind, case)
    result = price_mc(opt, mkt, MC_PATHS, SEED, antithetic, control_variate)
    assert result.std_err > 0
    assert abs(result.price - price_bs(opt, mkt)) < MC_Z * result.std_err


@pending("price_mc")
def test_same_seed_is_reproducible() -> None:
    assert price_mc(ATM_CALL, MKT, 10_000, seed=7) == price_mc(ATM_CALL, MKT, 10_000, seed=7)


@pending("price_mc")
def test_different_seeds_differ() -> None:
    assert price_mc(ATM_CALL, MKT, 10_000, seed=7) != price_mc(ATM_CALL, MKT, 10_000, seed=8)


# ---------------------------------------------------------------------------
# Standard error behaviour
# ---------------------------------------------------------------------------


@pending("price_mc")
def test_std_err_scales_as_inverse_sqrt_n() -> None:
    """Quadrupling the paths should halve the standard error."""
    small = price_mc(ATM_CALL, MKT, 20_000, SEED, antithetic=False, control_variate=False)
    large = price_mc(ATM_CALL, MKT, 80_000, SEED, antithetic=False, control_variate=False)
    assert small.std_err / large.std_err == pytest.approx(2.0, rel=MC_SE_SCALING_RTOL)


@pending("price_mc")
@pytest.mark.parametrize("kind", OptionKind)
def test_antithetic_reduces_std_err(kind: OptionKind) -> None:
    opt = Option(kind, 100.0, 1.0)
    plain = price_mc(opt, MKT, MC_PATHS, SEED, antithetic=False, control_variate=False)
    anti = price_mc(opt, MKT, MC_PATHS, SEED, antithetic=True, control_variate=False)
    assert anti.std_err < plain.std_err


@pending("price_mc")
@pytest.mark.parametrize("kind", OptionKind)
def test_control_variate_reduces_std_err(kind: OptionKind) -> None:
    opt = Option(kind, 100.0, 1.0)
    plain = price_mc(opt, MKT, MC_PATHS, SEED, antithetic=False, control_variate=False)
    cv = price_mc(opt, MKT, MC_PATHS, SEED, antithetic=False, control_variate=True)
    assert cv.std_err < plain.std_err


# ---------------------------------------------------------------------------
# Edge cases and input validation
# ---------------------------------------------------------------------------


@pending("price_mc")
@pytest.mark.parametrize("kind", OptionKind)
@pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
def test_zero_expiry_returns_intrinsic(kind: OptionKind, spot: float) -> None:
    opt = Option(kind, strike=100.0, expiry_years=0.0)
    result = price_mc(opt, Market(spot, 0.05, 0.2, 0.01), 1_000, SEED)
    assert result == MCResult(intrinsic(opt, spot), 0.0)


@pending("price_mc")
@pytest.mark.parametrize(("antithetic", "control_variate"), VR_FLAGS)
@pytest.mark.parametrize("kind", OptionKind)
def test_zero_vol_returns_discounted_forward_payoff(
    kind: OptionKind, antithetic: bool, control_variate: bool
) -> None:
    opt = Option(kind, strike=100.0, expiry_years=1.0)
    mkt = Market(110.0, rate=0.05, vol=0.0, div_yield=0.01)
    result = price_mc(opt, mkt, 1_000, SEED, antithetic, control_variate)
    assert result.price == pytest.approx(discounted_forward_payoff(opt, mkt))
    assert result.std_err == pytest.approx(0.0, abs=1e-12)


def test_rejects_american() -> None:
    opt = Option(OptionKind.PUT, 100.0, 1.0, style=ExerciseStyle.AMERICAN)
    with pytest.raises(ValueError, match="European"):
        price_mc(opt, MKT, 1_000, SEED)


@pytest.mark.parametrize("n_paths", [0, 1])
def test_rejects_too_few_paths(n_paths: int) -> None:
    with pytest.raises(ValueError, match="n_paths"):
        price_mc(ATM_CALL, MKT, n_paths, SEED, antithetic=False)


def test_rejects_odd_paths_with_antithetic() -> None:
    with pytest.raises(ValueError, match="even"):
        price_mc(ATM_CALL, MKT, 1_001, SEED, antithetic=True)
