# optionlab

A small, carefully tested Python library that prices vanilla equity options three ways:

- **Black-Scholes-Merton**: closed-form prices, analytical Greeks and implied volatility
- **Cox-Ross-Rubinstein binomial tree**: European and American exercise
- **Monte Carlo**: exact GBM sampling with antithetic variates and a control variate

The goal is to learn options pricing by implementing each model from first principles and
to verify it the way a quant-dev team would: published reference values, model-free
properties checked by property-based tests, cross-model convergence, and an independent
oracle (QuantLib) used only in tests.

Units, day count, edge cases and Greeks conventions are in
[docs/conventions.md](docs/conventions.md).

## Stack

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/), `src/` layout
- Runtime: numpy, scipy
- Dev: pytest, hypothesis, ruff, mypy (strict), matplotlib, jupyter
- Test-only oracle: QuantLib (`oracle` dependency group, never imported from `src/`)
- CI: GitHub Actions runs ruff, mypy and pytest on every push and pull request

## Setup

```sh
uv sync                    # runtime + dev dependencies
uv sync --group oracle     # optionally add QuantLib for tests/test_cross_model.py
uv run pytest
```

Lint and type-check:

```sh
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

## Usage

```python
from optionlab import Market, Option, OptionKind, price_bs

opt = Option(OptionKind.CALL, strike=40.0, expiry_years=0.5)
mkt = Market(spot=42.0, rate=0.10, vol=0.20)
price_bs(opt, mkt)  # per-share price
```

## Tests and the `pending` marker

The tests were written before the models. Tests that depend on an unimplemented model are
decorated with `@pending("<model>")` (defined in `tests/conftest.py`), which is
`pytest.mark.xfail(raises=NotImplementedError, strict=False)`. They show as `xfail` until
the model exists, so CI is green from day one.

The marker only tolerates `NotImplementedError`: once a model exists, any wrong answer
fails the run. **As you implement each model, delete its `@pending(...)` markers** so
passing tests report as passes rather than `XPASS`. Find them with
`grep -n 'pending("price_bs' tests/*.py`.

Reference values live in `tests/data/reference_values.json`. Entries marked `TODO` are
skipped until you fill them in from a textbook. The Hull example is marked
"verify against your edition".

## Milestones

- [ ] **M0 Setup**: tooling, CI, conventions, test scaffold
- [ ] **M1 Black-Scholes and Greeks**: `price_bs`, `greeks_bs`
- [ ] **M2 Implied volatility**: `implied_vol`
- [ ] **M3 Binomial tree**: `price_binomial`, European and American
- [ ] **M4 Monte Carlo**: `price_mc` with antithetic and control variates
- [ ] **M5 Cross-model study**: `notebooks/convergence.ipynb`, QuantLib comparison

Notes on what surprised me along the way: [docs/learning-log.md](docs/learning-log.md).

## Limitations

_To be written as the models are implemented. Candidates: flat rates and vol, continuous
dividends only, no discrete dividends, European-only Monte Carlo, no calendars or
business-day conventions._
