"""Typed records shared across every layer.

Author: 晨星 <CJX0712@users.noreply.github.com>

Design note
-----------
Every record here is a ``frozen=True`` dataclass. Two reasons:

1. Immutability makes the leakage assertions (I23: "select() must not mutate the
   fit state") checkable by value comparison rather than by convention.
2. ``status`` is a closed enumeration, so a ``skipped`` row can never be silently
   coerced into a number. That is the structural half of architecture §4.3
   ("skipped rows do not participate in aggregation, are not back-filled with 0
   and are not filled with nan").
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Canonical float dtype. Every array that crosses a layer boundary is float64.
FloatArray = NDArray[np.float64]


#: Status of a single (dataset, estimator) measurement.
class Status(StrEnum):
    """Lifecycle of one benchmark row."""

    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"


def array_fingerprint(x: FloatArray) -> str:
    """Return a stable SHA256 of an array's *content*.

    Used by the L3 fingerprint audit (architecture §5.2, assertion A3): the model
    records what it was fitted on, and reporting code refuses to publish a test
    metric for a model that has seen the test array.

    The digest covers shape, dtype and the raw C-contiguous bytes, so two arrays
    with identical values but different memory layout still agree, while any
    numeric difference changes the digest.
    """
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class Dataset:
    """Static description of a synthetic dataset."""

    name: str
    n_train: int
    n_test: int
    d: int
    intrinsic_dim: int
    seed: int
    summary: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Dataset.name must be a non-empty string")


@dataclass(frozen=True, slots=True)
class Split:
    """An independently sampled train / val / test triple plus its provenance.

    The three parts come from *three separate draws* with three different seeds,
    never from a random shuffle of one pool. See architecture §5.1.
    """

    train: FloatArray
    val: FloatArray
    test: FloatArray
    dataset: str
    train_seed: int
    val_seed: int
    test_seed: int

    @property
    def d(self) -> int:
        return int(self.train.shape[1])

    def manifest(self) -> SplitManifest:
        return SplitManifest(
            dataset=self.dataset,
            roles={id(self.train): "train", id(self.val): "val", id(self.test): "test"},
            seeds={"train": self.train_seed, "val": self.val_seed, "test": self.test_seed},
            fingerprints={
                "train": array_fingerprint(self.train),
                "val": array_fingerprint(self.val),
                "test": array_fingerprint(self.test),
            },
        )


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Role labels + fingerprints for every array in a split.

    ``roles`` maps ``id(array) -> role``. ``id()`` is used rather than a weak
    reference because the arrays are kept alive by the owning :class:`Split`, so
    the ids cannot be recycled while the manifest is in use.
    """

    dataset: str
    roles: dict[int, str]
    seeds: dict[str, int]
    fingerprints: dict[str, str]

    def role(self, x: FloatArray) -> str | None:
        """Return the recorded role of ``x``, or ``None`` if it is unregistered."""
        return self.roles.get(id(x))

    def with_role(self, x: FloatArray, role: str) -> SplitManifest:
        """Return a copy with one more array registered (used by tests)."""
        new_roles = dict(self.roles)
        new_roles[id(x)] = role
        return replace(self, roles=new_roles)


@dataclass(frozen=True, slots=True)
class Result:
    """One (dataset, estimator) row. Immutable so reports cannot be edited in place."""

    dataset: str
    estimator: str
    status: Status
    nll: float | None = None
    nll_std: float | None = None
    trustworthiness: float | None = None
    trustworthiness_std: float | None = None
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    seeds: tuple[int, ...] = ()

    @property
    def is_ok(self) -> bool:
        return self.status is Status.OK

    def aggregated(self) -> bool:
        """Whether this row may enter mean/std aggregation.

        ``skipped`` and ``failed`` rows are excluded: no zero-fill, no nan-fill
        (architecture §4.3). This predicate is the *only* gate used by
        :func:`densforge.eval.metrics.summarize`.
        """
        return self.status is Status.OK


@dataclass(frozen=True, slots=True)
class Report:
    """Aggregated benchmark output.

    Note the ``_report``-free attribute names: this class has no method named
    ``benchmark``/``nll``/``trustworthiness``, so no attribute shadows a method
    (architecture pitfall: instance attributes must never shadow methods).
    """

    results: tuple[Result, ...]
    n_seeds: int
    n_skipped: int
    n_failed: int
    meta: dict[str, Any] = field(default_factory=dict)

    def ok_results(self) -> tuple[Result, ...]:
        return tuple(r for r in self.results if r.aggregated())

    def skipped_datasets(self) -> tuple[str, ...]:
        return tuple(r.dataset for r in self.results if r.status is Status.SKIPPED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_seeds": self.n_seeds,
            "n_skipped": self.n_skipped,
            "n_failed": self.n_failed,
            "meta": dict(self.meta),
            "results": [
                {
                    "dataset": r.dataset,
                    "estimator": r.estimator,
                    "status": str(r.status),
                    "nll": r.nll,
                    "nll_std": r.nll_std,
                    "trustworthiness": r.trustworthiness,
                    "trustworthiness_std": r.trustworthiness_std,
                    "params": r.params,
                    "reason": r.reason,
                    "seeds": list(r.seeds),
                }
                for r in self.results
            ],
        }
