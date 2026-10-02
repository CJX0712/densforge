"""The leakage firewall: assertions A1 to A4.

Author: 晨星 <CJX0712@users.noreply.github.com>

The headline number is a held-out test NLL. It means something only if the model
never saw the test data, and that requires four independent checks, because each
one can be defeated by a plausible-looking refactor:

* **A1** seed isolation -- the three splits are independent *draws*, provable from
  their seeds;
* **A2** metric isolation -- the arrays touched during hyper-parameter selection are
  recorded, and the test array's id must not appear;
* **A3** model isolation -- a content fingerprint of everything the model was
  fitted on, checked before any test metric is published;
* **A4** shape overlap -- the minimum train-to-test distance must be positive.
"""

from __future__ import annotations

import numpy as np
import pytest

from densforge.core.errors import (
    LeakageError,
    LeakageModelNotTrainOnlyError,
    LeakageTestSetTouchedError,
    SeedCollisionError,
    ShapeMismatchError,
)
from densforge.core.seed import TEST_SEED_OFFSET
from densforge.core.types import array_fingerprint
from densforge.data.datasets import build_split, min_train_test_distance
from densforge.density.baselines import SkKernelDensity
from densforge.density.flagship import DensFuse
from densforge.eval.leakage import (
    ScoreRecorder,
    assert_no_overlap,
    assert_seed_isolation,
    assert_split_roles,
    assert_test_untouched,
    assert_train_only_model,
)
from densforge.training.fitter import TrainOnlyModel, fit_density


@pytest.fixture(scope="module")
def split():
    return build_split("circles", n_train=200, n_test=100, seed=0)


# ============================================================== A1  seed isolation
def test_a1_split_seeds_are_independent_draws(split) -> None:
    """A1: the test seed is the train seed plus a fixed offset, so the draws are
    provably separate."""
    assert_seed_isolation(split.train_seed, split.test_seed)
    assert split.test_seed == (split.train_seed + TEST_SEED_OFFSET) % (2**32)


def test_a1_identical_seeds_are_refused() -> None:
    """A1: a train/test seed collision is refused outright, not tolerated."""
    with pytest.raises(LeakageError, match="share seed"):
        assert_seed_isolation(7, 7)


def test_a1_unrelated_seeds_are_refused() -> None:
    """A1: a merely-different seed is not enough; the fixed offset is the proof.

    This is the assertion a ``train_test_split`` refactor would fail: it produces
    different seeds, but not seeds related by the documented offset.
    """
    with pytest.raises(LeakageError, match="independent sampling"):
        assert_seed_isolation(0, 12345)


def test_a1_seed_collisions_are_structurally_impossible_and_still_guarded() -> None:
    """A1: the fixed offsets make a collision unreachable, and the guard proves it.

    ``train = s``, ``val = s + 500``, ``test = s + 10_000`` mod ``2**32``. Three
    distinct constants can never coincide under a bijection, so the
    :class:`SeedCollisionError` check in :func:`build_split` is unreachable with the
    documented offsets. It is a defence against someone later changing an offset to
    a duplicate value, so the guard is exercised directly rather than through a
    scenario that cannot occur.
    """
    # Exhaustive over a wraparound window: no base seed can produce a collision.
    for base in range(2**32 - 10_500, 2**32):
        seeds = {base % 2**32, (base + 500) % 2**32, (base + 10_000) % 2**32}
        assert len(seeds) == 3, f"collision at base={base}"

    # And the guard fires when a collision is injected directly.
    from densforge.data import datasets as datasets_module

    original = datasets_module.TEST_SEED_OFFSET
    try:
        # A duplicate offset makes val and test share a seed.
        datasets_module.TEST_SEED_OFFSET = 500
        with pytest.raises(SeedCollisionError, match="collided"):
            datasets_module.build_split("circles", n_train=100, n_test=50, seed=0)
    finally:
        datasets_module.TEST_SEED_OFFSET = original


# ========================================================== A2  metric isolation
def test_a2_tuning_must_not_touch_the_test_array(split) -> None:
    """A2: the arrays scored during selection are recorded and checked."""
    recorder = ScoreRecorder()
    with recorder:
        SkKernelDensity(bandwidth=1.0).fit(split.train).score_samples(
            recorder.record(split.val)
        )
    assert id(split.val) in recorder.touched
    assert id(split.test) not in recorder.touched
    recorder.assert_clean(split.test)


def test_a2_detects_a_leak(split) -> None:
    """A2: scoring the test array during selection is caught.

    The subtlest failure in the whole project: a tuning loop that accidentally
    evaluates on test data produces entirely plausible numbers and a slightly
    too-good result, with nothing else wrong anywhere.
    """
    recorder = ScoreRecorder()
    with recorder:
        recorder.record(split.test)
    with pytest.raises(LeakageTestSetTouchedError, match="test array"):
        recorder.assert_clean(split.test)


def test_a2_assert_test_untouched_directly() -> None:
    """A2: the standalone assertion behaves the same way."""
    array = np.zeros((10, 2))
    assert_test_untouched({id(array) + 1}, id(array))
    with pytest.raises(LeakageTestSetTouchedError):
        assert_test_untouched({id(array)}, id(array))


def test_a2_recorder_is_inert_outside_its_context() -> None:
    """A2: the recorder only records while active, so a stray call is not miscounted."""
    recorder = ScoreRecorder()
    array = np.zeros((5, 2))
    recorder.record(array)
    assert recorder.touched == frozenset()


# ========================================================== A3  model isolation
def test_a3_model_fitted_on_test_is_refused(split) -> None:
    """A3: a model that has seen the test data cannot produce a test metric."""
    leaky = fit_density(SkKernelDensity(bandwidth=1.0), split.test)
    with pytest.raises(LeakageModelNotTrainOnlyError, match="same content"):
        assert_train_only_model(leaky, split.test)


def test_a3_a_copy_of_the_test_array_is_also_caught(split) -> None:
    """A3: detection is by content digest, not by object identity.

    ``X.copy()`` has a different ``id()`` and the same content, and is just as much
    of a leak. An identity-based check would wave it through.
    """
    disguised = np.array(split.test, copy=True)
    assert disguised is not split.test
    assert id(disclosed_fix(disguised)) != id(split.test)
    leaky = fit_density(SkKernelDensity(bandwidth=1.0), disguised)
    with pytest.raises(LeakageModelNotTrainOnlyError):
        assert_train_only_model(leaky, split.test)


def disclosed_fix(array: np.ndarray) -> np.ndarray:
    """Return the array unchanged; a named no-op so the identity check reads clearly."""
    return array


def test_a3_clean_model_passes(split) -> None:
    """A3: a train-only model is accepted."""
    model = DensFuse().fit(split.train)
    assert_train_only_model(model, split.test)
    assert array_fingerprint(split.train) in model.fitted_fingerprints


def test_a3_model_without_provenance_is_refused(split) -> None:
    """A3: a model that cannot prove its provenance must not produce a test metric.

    A vacuous leakage check is worse than none, because it looks like a pass.
    """

    class Opaque:
        def score_samples(self, X):
            return np.zeros(len(X))

    with pytest.raises(LeakageModelNotTrainOnlyError, match="cannot be verified"):
        assert_train_only_model(Opaque(), split.test)


def test_a3_fitter_records_exactly_what_it_was_fitted_on(split) -> None:
    """A3: the wrapper's record matches its input exactly."""
    wrapper = fit_density(SkKernelDensity(bandwidth=1.0), split.train)
    assert wrapper.fitted_on is split.train
    assert wrapper.fitted_fingerprints == frozenset({array_fingerprint(split.train)})
    assert wrapper.name == "SkKernelDensity"
    assert "bandwidth" in wrapper.params()


def test_a3_fingerprints_are_content_addressed() -> None:
    """A3: two equal arrays share a digest; a changed value changes it."""
    a = np.arange(10.0)
    b = a.copy()
    c = a.copy()
    c[0] = 99.0
    assert array_fingerprint(a) == array_fingerprint(b)
    assert array_fingerprint(a) != array_fingerprint(c)
    # Shape is part of the identity: a reshaped view is a different object.
    assert array_fingerprint(a) != array_fingerprint(a.reshape(2, 5))


# ========================================================== A4  shape overlap
def test_a4_train_and_test_share_no_point(split) -> None:
    """A4: the minimum train-to-test distance is strictly positive."""
    minimum = assert_no_overlap(split)
    assert minimum > 0.0
    assert min_train_test_distance(split) == pytest.approx(minimum, rel=1e-12)


def test_a4_exact_overlap_is_detected() -> None:
    """A4: a distance of exactly 0.0 means the splits were sliced from one pool."""
    from densforge.core.types import Split

    X = np.random.RandomState(0).randn(40, 2)
    carved = Split(
        train=X[:20],
        val=X[20:30],
        test=X[10:20],  # deliberately overlapping the training half
        dataset="synthetic",
        train_seed=0,
        val_seed=500,
        test_seed=TEST_SEED_OFFSET,
    )
    with pytest.raises(LeakageError, match="identical point"):
        assert_no_overlap(carved)


# ========================================================== combined
def test_all_assertions_pass_for_a_real_run(split) -> None:
    """The full firewall, exercised together as the pipeline uses it."""
    assert_split_roles(split)
    manifest = split.manifest()
    model = DensFuse().fit(split.train, manifest=manifest)
    assert_train_only_model(model, split.test)
    assert manifest.role(split.train) == "train"
    assert manifest.role(split.test) == "test"


def test_manifest_detects_a_tampered_array(split) -> None:
    """The manifest digests must match its arrays, or the split was edited after
    the manifest was built."""
    manifest = split.manifest()
    tampered = np.array(split.train, copy=True)
    tampered[0, 0] += 1.0
    from dataclasses import replace as dc_replace

    bad = dc_replace(
        manifest,
        fingerprints={**manifest.fingerprints, "train": array_fingerprint(tampered)},
    )
    with pytest.raises(LeakageError, match="does not match the array content"):
        assert_split_roles(split, bad)


def test_manifest_role_lookup_returns_none_for_unregistered(split) -> None:
    """An unregistered array has no role, which is what makes ``fit`` refuse it."""
    manifest = split.manifest()
    assert manifest.role(np.zeros((5, 2))) is None
    assert manifest.role(split.train) == "train"


def test_manifest_with_role_registers_a_new_array(split) -> None:
    """``with_role`` is how a test registers an extra array."""
    manifest = split.manifest()
    extra = np.zeros((5, 2))
    assert manifest.role(extra) is None
    assert manifest.with_role(extra, "train").role(extra) == "train"
    # The original is untouched: manifests are immutable.
    assert manifest.role(extra) is None


def test_split_rejects_inconsistent_dimensions() -> None:
    """A split whose parts disagree on dimensionality is refused."""
    with pytest.raises(ShapeMismatchError):
        build_split("circles", n_train=100, n_test=50, seed=0, n_val=0)


def test_wrapper_reports_a_readable_repr(split) -> None:
    """A provenance wrapper that cannot be read in a traceback is not much use."""
    wrapper = TrainOnlyModel(SkKernelDensity(bandwidth=1.0), split.train)
    text = repr(wrapper)
    assert "SkKernelDensity" in text
    assert "n=200" in text
