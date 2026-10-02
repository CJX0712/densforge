"""The one and only randomness entry point.

Author: 晨星 <CJX0712@users.noreply.github.com>

Why ``RandomState`` and not ``default_rng``
-------------------------------------------
NEP 19 freezes the legacy ``RandomState`` bit stream (MT19937) but explicitly
does **not** promise that ``Generator`` (``np.random.default_rng``) produces the
same stream across NumPy versions. Measured on this machine, the two are
completely different streams for the same seed::

    RandomState(42).rand(3)  -> [0.37454012 0.95071431 0.73199394]
    default_rng(42).random(3) -> [0.77395605 0.43887844 0.85859792]

Because the CI matrix spans Python 3.12 and 3.13, and therefore may span two
NumPy versions, a ``Generator``-based pipeline would let the published
performance thresholds drift silently between jobs. Every stochastic decision in
DensForge -- data generation, shuffling, baseline ``random_state`` -- therefore
goes through :func:`get_rng`, which returns a legacy ``RandomState``.

Two further rules, both from the design's determinism section:

* module-level ``np.random.rand`` / ``np.random.randn`` / ``np.random.seed`` are
  banned, because they mutate process-global state;
* ``sklearn`` estimators receive an **integer** ``random_state``, never a
  ``RandomState`` object. ``check_random_state`` returns an object unchanged, so
  passing one means the second call sees an already-consumed stream.
"""

from __future__ import annotations

import os
import random

import numpy as np

from .errors import ConfigRangeError

#: Environment variables that pin BLAS/OpenMP to a single thread.
#:
#: Multi-threaded reductions sum in nondeterministic order, which perturbs the
#: last bits of every floating point result. That is enough to flip a
#: bit-exactness test (I25) and to move a reported NLL in the 12th digit.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: Seeds used by the CI fast lane. The full report uses ``REPORT_SEEDS``.
CI_SEEDS: tuple[int, ...] = (0, 1, 2)

#: Seeds used for headline numbers (5 seeds, enough for a meaningful std).
REPORT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: Offset applied to the base seed to draw the *test* split.
#:
#: A fixed offset (rather than a fresh random seed) is what makes assertion A1
#: checkable: ``test_seed == train_seed + TEST_SEED_OFFSET`` proves the test set
#: is an independent draw and not a slice of the training pool.
TEST_SEED_OFFSET = 10_000

#: Offset applied to the base seed to draw the *validation* split.
VAL_SEED_OFFSET = 500

_last_seed: int | None = None


def get_rng(seed: int) -> np.random.RandomState:
    """Return a legacy ``RandomState`` for ``seed``.

    This is the *only* sanctioned way to obtain randomness in DensForge. It
    deliberately does not touch NumPy's global state, so calling it cannot
    perturb an unrelated part of the program (invariant I27).

    Parameters
    ----------
    seed:
        Non-negative integer below ``2**32``.

    Raises
    ------
    ConfigRangeError
        If ``seed`` is negative or above ``2**32 - 1``.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ConfigRangeError(f"seed must be an int, got {type(seed).__name__}")
    value = int(seed)
    if not 0 <= value < 2**32:
        raise ConfigRangeError(f"seed must satisfy 0 <= seed < 2**32, got {value}")
    return np.random.RandomState(value)


def pin_single_thread() -> None:
    """Force single-threaded BLAS/OpenMP.

    Must run *before* NumPy/SciPy import their backends to be fully effective,
    which is why :func:`set_all` is called at the top of every entry point. On
    an already-imported stack the variables are still set so that any library
    read at call time sees them.
    """
    for name in _THREAD_ENV_VARS:
        os.environ[name] = "1"


def set_all(seed: int) -> int:
    """Seed every stochastic source DensForge can reach, and pin threading.

    This is the single seed entry point named in the package's
    ``SEED_ENTRY_POINT``. It does three things:

    1. validates and records ``seed``;
    2. seeds the standard library's :mod:`random` and **NumPy's legacy global
       state** so that third-party code which reaches for the global generator
       still lands on a reproducible stream;
    3. pins BLAS/OpenMP to one thread.

    NumPy's *global* state is seeded here on purpose, at the process boundary.
    DensForge's own code never reads it -- it always builds a local
    ``RandomState(seed)`` via :func:`get_rng` -- so invariant I27 still holds for
    library code while third-party estimators that insist on the global
    generator remain reproducible.

    Returns
    -------
    int
        The seed, so callers can log it.

    Raises
    ------
    ConfigRangeError
        If ``seed`` is outside ``[0, 2**32)``.
    """
    global _last_seed

    value = int(seed)
    # Validate before mutating any global state: a bad seed must not leave the
    # process half-seeded.
    probe = get_rng(value)
    del probe

    pin_single_thread()
    random.seed(value)
    np.random.seed(value)
    _last_seed = value
    return value


def current_seed() -> int | None:
    """Return the seed most recently passed to :func:`set_all`, or ``None``."""
    return _last_seed


def derive_seed(base_seed: int, offset: int) -> int:
    """Return ``base_seed + offset`` with wraparound into the valid seed range.

    Split offsets must not be able to drift out of range, otherwise a large
    ``base_seed`` would raise deep inside a data generator instead of at the
    split boundary where the mistake actually is.
    """
    return (int(base_seed) + int(offset)) % (2**32)


__all__ = [
    "CI_SEEDS",
    "REPORT_SEEDS",
    "TEST_SEED_OFFSET",
    "VAL_SEED_OFFSET",
    "current_seed",
    "derive_seed",
    "get_rng",
    "pin_single_thread",
    "set_all",
]
