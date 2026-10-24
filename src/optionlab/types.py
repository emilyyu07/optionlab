"""Value types shared by every pricing model.

All types are frozen dataclasses that validate themselves on construction, so a
pricing function can assume it was handed a well-formed contract and market.
Units and conventions are documented in docs/conventions.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

# Two-sided 95% standard normal quantile, used for MCResult.ci95.
_Z_95 = 1.959963984540054


class OptionKind(Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


def _require_non_negative(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")


def require_european(opt: Option, model: str) -> None:
    """Raise ValueError if `opt` is not European; `model` names the caller in the message."""
    if opt.style is not ExerciseStyle.EUROPEAN:
        raise ValueError(f"{model} prices European options only, got {opt.style.value}")


@dataclass(frozen=True, slots=True)
class Option:
    """A vanilla option contract.

    Attributes:
        kind: CALL or PUT.
        strike: Strike price per share, > 0.
        expiry_years: Time to expiry in years (ACT/365), >= 0. Zero means "at expiry".
        style: EUROPEAN (default) or AMERICAN.
    """

    kind: OptionKind
    strike: float
    expiry_years: float
    style: ExerciseStyle = ExerciseStyle.EUROPEAN

    def __post_init__(self) -> None:
        _require_positive("strike", self.strike)
        _require_non_negative("expiry_years", self.expiry_years)


@dataclass(frozen=True, slots=True)
class Market:
    """Flat market state for a single underlying.

    Attributes:
        spot: Underlying price per share, > 0.
        rate: Risk-free rate, continuously compounded, annualized decimal. May be negative.
        vol: Volatility, annualized decimal (0.20 = 20%), >= 0.
        div_yield: Continuous dividend yield, annualized decimal. Defaults to 0.
    """

    spot: float
    rate: float
    vol: float
    div_yield: float = 0.0

    def __post_init__(self) -> None:
        _require_positive("spot", self.spot)
        _require_finite("rate", self.rate)
        _require_non_negative("vol", self.vol)
        _require_finite("div_yield", self.div_yield)


@dataclass(frozen=True, slots=True)
class Greeks:
    """First- and second-order sensitivities in raw analytical units.

    Attributes:
        delta: dV/dS.
        gamma: d2V/dS2.
        vega: dV/dsigma, per 1.00 change in vol (not per vol point).
        theta: dV/dt per year of calendar time passing, i.e. -dV/dT.
        rho: dV/dr, per 1.00 change in rate (not per basis point).
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass(frozen=True, slots=True)
class MCResult:
    """A Monte Carlo estimate and its standard error.

    Attributes:
        price: Sample mean of discounted payoffs (after any variance reduction).
        std_err: Standard error of `price`, >= 0.
    """

    price: float
    std_err: float

    def __post_init__(self) -> None:
        _require_finite("price", self.price)
        _require_non_negative("std_err", self.std_err)

    @property
    def ci95(self) -> tuple[float, float]:
        """Normal-approximation 95% confidence interval (price -/+ 1.96 * std_err)."""
        half_width = _Z_95 * self.std_err
        return (self.price - half_width, self.price + half_width)
