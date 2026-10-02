"""Determinism contracts (invariants I25, I26, I27).

Author: 晨星 <CJX0712@users.noreply.github.com>

The premise of every published threshold in this project is that a number
reproduced on another machine, on another Python version, is the *same* number.
That requires three separate guarantees, each checked here.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from densforge.core.seed import TEST_SEED_OFFSET, derive_seed, get_rng, set_all
from densforge.data.datasets import build_split
from densforge.data.synth import (
    make_aniso_gmm,
    make_circles,
    make_double_spiral,
    make_iso_gauss,
    make_manifold_noise,
    make_swiss_roll,
    make_t_mixture,
)

pytestmark = pytest.mark.invariant

ALL_GENERATORS = (
    make_aniso_gmm,
    make_swiss_roll,
    make_double_spiral,
    make_circles,
    make_t_mixture,
    make_manifold_noise,
    make_iso_gauss,
)


# ==================================================== I26  RandomState golden values
#: Exact values from the legacy MT19937 stream, each drawn from a *fresh*
#: ``RandomState(12345)`` on numpy 2.5.3.
#:
#: NEP 19 freezes ``RandomState`` but not ``Generator``, which is the entire reason
#: this project uses ``RandomState`` everywhere. These values turn that promise into
#: a test: if a NumPy upgrade ever changes the legacy stream, this fails loudly
#: instead of letting every published threshold drift silently between CI jobs.
#:
#: Full repr precision, not the 8-decimal form: ``assert_array_equal`` is exact, and
#: a truncated literal never matches.
GOLD_RAND_5 = np.array(
    [
        0.9296160928171479,
        0.3163755545817859,
        0.18391881167709445,
        0.2045602785530397,
        0.5677250290816866,
    ]
)
GOLD_RANDN_3 = np.array(
    [
        -0.20470765948471295,
        0.47894333805754824,
        -0.5194387150567381,
    ]
)
GOLD_RANDINT_7 = np.array([98, 29, 1, 36, 41, 34, 29], dtype=np.int32)
GOLD_SHUFFLE_10 = np.array([0, 7, 3, 9, 6, 4, 1, 8, 5, 2])
GOLD_STANDARD_T_2 = np.array([2.419494450164955, 0.0376932134878793])


def test_i26_randomstate_bit_stream_is_frozen() -> None:
    """I26: the legacy stream produces the documented values on this machine."""
    assert_array_equal(np.random.RandomState(12345).rand(5), GOLD_RAND_5)
    assert_array_equal(np.random.RandomState(12345).randn(3), GOLD_RANDN_3)
    assert_array_equal(np.random.RandomState(12345).randint(0, 100, 7), GOLD_RANDINT_7)

    shuffled = np.arange(10)
    np.random.RandomState(12345).shuffle(shuffled)
    assert_array_equal(shuffled, GOLD_SHUFFLE_10)

    assert_array_equal(np.random.RandomState(7).standard_t(3.0, size=2), GOLD_STANDARD_T_2)


def test_i26_generator_produces_a_different_stream() -> None:
    """I26: ``default_rng`` really is a different stream, which is why it is banned.

    Pinned so that a future NumPy cannot quietly make the two equivalent (or make
    this project's ban look unnecessary when it is not).
    """
    legacy = np.random.RandomState(42).rand(3)
    generator = np.random.default_rng(42).random(3)
    assert not np.allclose(legacy, generator), (
        "RandomState and default_rng now agree; re-check whether the "
        "generator-stability argument for banning default_rng still holds"
    )


@pytest.mark.parametrize("generator", ALL_GENERATORS)
def test_i26_generators_are_reproducible(generator) -> None:
    """I26: every generator is a pure function of ``(n, seed, knobs)``."""
    a = generator(120, 3)
    b = generator(120, 3)
    assert_array_equal(a, b)
    assert a.dtype == np.float64
    assert a.ndim == 2 and a.shape[0] == 120


@pytest.mark.parametrize("generator", ALL_GENERATORS)
def test_i26_different_seeds_give_different_data(generator) -> None:
    """I26 guard: a seed that changes nothing is a broken seed."""
    assert not np.allclose(generator(120, 3), generator(120, 4))


# ==================================================== I27  no global RNG pollution
def _global_state() -> tuple:
    """Snapshot NumPy's global ``RandomState`` internals.

    The design document's proposed I27 check is
    ``np.random.seed(0); a = np.random.rand(); make_split(...); b = np.random.rand();
    assert a == b`` -- which **can never pass**, because ``a`` and ``b`` are two
    consecutive draws from one stream and consecutive draws are never equal.
    Measured: 0.5488... then 0.7151.... The test as written would fail even for a
    perfectly clean generator, and the obvious "fix" of removing the second draw
    would make it vacuous.

    The property that actually matters is that the global stream's *state* is
    untouched, which is directly observable through ``get_state()``.
    """
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return name, keys.copy(), pos, has_gauss, cached


def _assert_global_state_unchanged(before: tuple) -> None:
    after = _global_state()
    assert before[0] == after[0], "global MT19937 generator was replaced"
    assert np.array_equal(before[1], after[1]), "global stream state advanced"
    assert before[2] == after[2], "global stream position advanced"
    assert before[3] == after[3], "global gaussian flag changed"


def test_i27_data_generation_does_not_touch_the_global_stream() -> None:
    """I27: generating data must not advance NumPy's global state.

    The global stream belongs to whoever called us. A generator that consumed from
    it would make an unrelated part of the caller's program non-reproducible, and
    the damage would surface far from the cause.
    """
    np.random.seed(0)
    before = _global_state()
    for generator in ALL_GENERATORS:
        generator(80, 11)
    build_split("aniso_gmm", n_train=100, n_test=50, seed=0)
    _assert_global_state_unchanged(before)


def test_i27_get_rng_does_not_touch_the_global_stream() -> None:
    """I27: :func:`get_rng` returns an independent stream, leaving the global one alone."""
    np.random.seed(0)
    before = _global_state()
    for seed in range(20):
        get_rng(seed).rand(10)
        get_rng(seed).shuffle(np.arange(10))
    _assert_global_state_unchanged(before)


def test_i27_control_detects_actual_pollution() -> None:
    """I27 control: the isolation check would notice real pollution.

    A test that passes for the wrong reason is worse than no test, so the detector
    is itself verified against a known-polluting operation: reading
    ``np.random.rand`` directly *does* advance the global stream.
    """
    np.random.seed(0)
    before = _global_state()
    np.random.rand()  # the forbidden operation
    with pytest.raises(AssertionError):
        _assert_global_state_unchanged(before)


def test_i27_set_all_does_seed_the_global_state_on_purpose() -> None:
    """I27: ``set_all`` *does* seed the global stream, and that is deliberate.

    It seeds it exactly once, at the process boundary, so that third-party
    estimators reaching for the global generator are still reproducible. DensForge's
    own code never reads the global stream -- it always builds a local
    ``RandomState(seed)`` -- so the library stays reproducible either way.
    """
    set_all(1234)
    first = np.random.rand()
    set_all(1234)
    assert np.random.rand() == first


# ==================================================== seed plumbing
def test_get_rng_rejects_out_of_range_seeds() -> None:
    """A bad seed must fail at the boundary, not deep inside a generator."""
    from densforge.core.errors import ConfigRangeError

    for bad in (-1, 2**32, 2**33):
        with pytest.raises(ConfigRangeError):
            get_rng(bad)
    for bad in (1.5, "3", None, True):
        with pytest.raises(ConfigRangeError):
            get_rng(bad)  # type: ignore[arg-type]
    assert get_rng(0) is not get_rng(0), "each call must return an independent stream"


def test_derive_seed_wraps_instead_of_overflowing() -> None:
    """Split offsets must not be able to drift out of the valid seed range."""
    assert derive_seed(0, TEST_SEED_OFFSET) == TEST_SEED_OFFSET
    assert derive_seed(2**32 - TEST_SEED_OFFSET, TEST_SEED_OFFSET) == 0
    assert 0 <= derive_seed(2**32 - 1, 1) < 2**32


def test_split_seeds_are_distinct_and_offset() -> None:
    """The three splits must use three distinct, related seeds."""
    split = build_split("circles", n_train=100, n_test=50, seed=0)
    assert split.train_seed == 0
    assert split.val_seed == 500
    assert split.test_seed == TEST_SEED_OFFSET
    assert len({split.train_seed, split.val_seed, split.test_seed}) == 3
