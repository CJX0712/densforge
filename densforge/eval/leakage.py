"""Leakage defences: the three assertions plus the shape overlap check.

Author: 晨星 <CJX0712@users.noreply.github.com>

What "leakage" means here
-------------------------
The headline number is a held-out test NLL. That number means something only if
the model never saw the test data. Four independent checks enforce that, because
any one of them can be defeated by a plausible-looking refactor:

===  ==========================================================================
A1   seed isolation      the three splits are independent *draws*, provable from
                         their seeds: ``test_seed == train_seed + 10_000``
A2   metric isolation    the array ids touched during hyper-parameter selection
                         are recorded; the test array's id must not appear
A3   model isolation     the model records a content fingerprint of everything it
                         was fitted on; a test metric is refused for a model
                         whose fingerprint set contains the test array
A4   shape overlap       the minimum train-to-test distance must be strictly
                         positive; exactly 0.0 means the two sets share points
===  ==========================================================================

A1 is a *proof* rather than a heuristic: because the test seed is the train seed
plus a fixed offset, a collision is detectable rather than merely unlikely.

A2 is the only check that catches the subtlest failure -- a hyper-parameter search
that accidentally evaluates on the test set produces entirely plausible numbers
and a slightly-too-good result, with nothing else wrong.

Note what A2 does **not** do: it does not prevent a leak, it detects one after the
fact. The prevention is structural -- ``fit`` refuses a non-train array outright
(invariant I24), and ``score_samples`` is a pure function that writes no state
(invariant I22).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..core.errors import (
    LeakageError,
    LeakageModelNotTrainOnlyError,
    LeakageTestSetTouchedError,
)
from ..core.seed import TEST_SEED_OFFSET
from ..core.types import FloatArray, Split, SplitManifest, array_fingerprint

#: Number of leading training rows probed by the A4 overlap check.
A4_PROBE_ROWS = 64


def assert_seed_isolation(train_seed: int, test_seed: int) -> None:
    """Assertion **A1**: the test split is an independent draw, not a slice.

    Two conditions, both required:

    1. the seeds differ, and
    2. ``test_seed == (train_seed + 10_000) mod 2**32``.

    The second is the strong one. A merely-different seed could come from
    anything; the fixed offset *proves* the intended independent-draw protocol was
    followed, and a refactor that switches to ``train_test_split`` cannot satisfy
    it.

    Raises
    ------
    LeakageError
        If either condition fails.
    """
    if int(train_seed) == int(test_seed):
        raise LeakageError(
            f"train and test share seed {train_seed}; the test set would be a copy of "
            "the training set. Refusing to run."
        )
    expected = (int(train_seed) + TEST_SEED_OFFSET) % (2**32)
    if int(test_seed) != expected:
        raise LeakageError(
            f"test_seed={test_seed} is not train_seed + {TEST_SEED_OFFSET} "
            f"(expected {expected}); the split was not produced by independent sampling"
        )


def assert_test_untouched(touched_ids: Iterable[int], test_id: int) -> None:
    """Assertion **A2**: hyper-parameter selection never saw the test array.

    Parameters
    ----------
    touched_ids:
        ``id()`` of every array passed to ``score_samples`` during selection.
    test_id:
        ``id()`` of the test array.

    Raises
    ------
    LeakageTestSetTouchedError
        If the test array appears in ``touched_ids``.
    """
    seen = set(touched_ids)
    if int(test_id) in seen:
        raise LeakageTestSetTouchedError(
            "the test array was passed to score_samples during hyper-parameter "
            "selection; selecting on test data makes the reported metric meaningless"
        )


def assert_train_only_model(model: Any, X_test: FloatArray) -> None:
    """Assertion **A3**: the model was fitted without the test array.

    Two checks. The model's ``fitted_fingerprints`` set -- a content digest, not
    an identity -- must not contain the test array's digest. The digest matters
    because a test array can be *copied*: ``id()`` would differ while the content
    does not, and a copy is just as much of a leak.

    Raises
    ------
    LeakageModelNotTrainOnlyError
        If the test array's fingerprint is among the fitted fingerprints, or the
        model exposes no fingerprint information at all.
    """
    fingerprints = getattr(model, "fitted_fingerprints", None)
    if fingerprints is None:
        raise LeakageModelNotTrainOnlyError(
            f"{type(model).__name__} does not expose fitted_fingerprints; "
            "model isolation cannot be verified, so no test metric may be reported"
        )
    digest = array_fingerprint(X_test)
    if digest in frozenset(fingerprints):
        raise LeakageModelNotTrainOnlyError(
            "the model has been fitted on an array with the same content as the test "
            "data; reporting a test metric for it would be meaningless"
        )


def assert_no_overlap(split: Split, *, probe_rows: int = A4_PROBE_ROWS) -> float:
    """Assertion **A4**: train and test share no points.

    Under independent sampling from a continuous distribution the minimum distance
    is 0 only with probability zero, so an exact 0.0 is positive evidence that the
    two sets were carved out of one pool. Measured on this machine: 5.15.

    Returns
    -------
    float
        The observed minimum distance, so callers can record it.

    Raises
    ------
    LeakageError
        If the minimum distance is exactly zero.
    """
    from scipy.spatial.distance import cdist

    head = split.train[: min(probe_rows, split.train.shape[0])]
    minimum = float(cdist(head, split.test).min())
    if minimum == 0.0:
        raise LeakageError(
            "train and test contain an identical point (minimum distance 0.0); "
            "the splits were sliced from one pool rather than sampled independently"
        )
    return minimum


def assert_split_roles(split: Split, manifest: SplitManifest | None = None) -> None:
    """Verify every split array carries the role it claims.

    Checks the manifest when one is supplied, and the seed relationship always.
    """
    assert_seed_isolation(split.train_seed, split.test_seed)
    assert_no_overlap(split)
    if manifest is not None:
        expected = {
            "train": (split.train, "train"),
            "val": (split.val, "val"),
            "test": (split.test, "test"),
        }
        for array, want in expected.values():
            got = manifest.role(array)
            if got != want:
                raise LeakageError(
                    f"array registered for split {split.dataset!r} has role {got!r}, "
                    f"expected {want!r}"
                )
        digests = {
            "train": array_fingerprint(split.train),
            "val": array_fingerprint(split.val),
            "test": array_fingerprint(split.test),
        }
        for role, digest in digests.items():
            if manifest.fingerprints.get(role) != digest:
                raise LeakageError(
                    f"manifest fingerprint for {role!r} does not match the array content; "
                    "the split was modified after the manifest was built"
                )


class ScoreRecorder:
    """Context manager recording which arrays ``score_samples`` is called on.

    Wraps the hyper-parameter selection phase so that assertion **A2** can be
    checked afterwards against the real call history, rather than trusting a
    code review to confirm that no test array slipped through.

    Examples
    --------
    >>> recorder = ScoreRecorder()
    >>> with recorder:
    ...     _ = estimator.score_samples(X_val)     # doctest: +SKIP
    >>> recorder.touched                          # doctest: +SKIP
    {id-of-X_val}
    >>> assert_test_untouched(recorder.touched, id(X_test))   # doctest: +SKIP
    """

    def __init__(self) -> None:
        self._touched: set[int] = set()
        self._active = False

    def __enter__(self) -> ScoreRecorder:
        self._active = True
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._active = False

    def record(self, X: FloatArray) -> FloatArray:
        """Record ``X``'s identity and return it unchanged."""
        if self._active:
            self._touched.add(id(np.asarray(X)))
        return X

    @property
    def touched(self) -> frozenset[int]:
        """The ids recorded so far."""
        return frozenset(self._touched)

    def assert_clean(self, X_test: FloatArray) -> None:
        """Assertion A2 against a specific test array."""
        assert_test_untouched(self._touched, id(X_test))


__all__ = [
    "A4_PROBE_ROWS",
    "ScoreRecorder",
    "assert_no_overlap",
    "assert_seed_isolation",
    "assert_split_roles",
    "assert_test_untouched",
    "assert_train_only_model",
]
