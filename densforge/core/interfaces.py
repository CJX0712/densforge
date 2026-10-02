"""Structural protocols: the contracts every estimator must satisfy.

Author: 晨星 <CJX0712@users.noreply.github.com>

The single invariant that makes cross-module comparison meaningful
---------------------------------------------------------------
``score_samples`` returns **log-density** and *larger means more likely*. Every
density number in this project is derived from that one convention:

    NLL = -mean(score_samples(X_test))          # in nats, lower is better

Returning a probability instead is not a style choice, it is a correctness
failure: measured on 500x2 Gaussian samples, shrinking the bandwidth to 1e-3
drives ``log p`` to -1.73e6, ``exp`` of that overflows to ``inf`` and the NLL
becomes ``nan`` -- with no exception raised. See architecture §2.2.
"""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

import numpy as np

from .types import FloatArray


@runtime_checkable
class DensityEstimator(Protocol):
    """Learns ``p(x)``. Primary metric: held-out test NLL (lower is better)."""

    def fit(self, X: FloatArray) -> Self:
        """Fit on training data only. Must never touch val or test data."""
        ...

    def score_samples(self, X: FloatArray) -> FloatArray:
        """Return ``log p(x)`` with shape ``(n,)``; larger means more likely.

        Three obligations:

        1. natural logarithm, never ``exp(log p)``;
        2. ``NLL == -mean(score_samples(X_test))`` is in nats;
        3. comparable only within one dataset, never across datasets.
        """
        ...

    @classmethod
    def available(cls) -> bool:
        """Whether the backing implementation is importable.

        Must be a pure boolean predicate: it must not construct an estimator,
        because the benchmark loop calls it once per row and construction can be
        expensive (architecture §4.1).
        """
        ...


@runtime_checkable
class ManifoldEmbedder(Protocol):
    """Recovers intrinsic geometry. Primary metric: trustworthiness (higher is better)."""

    def fit(self, X: FloatArray) -> Self:
        """Fit on training data only."""
        ...

    def transform(self, X: FloatArray) -> FloatArray:
        """Return embedding coordinates with shape ``(n, n_components)``."""
        ...

    @classmethod
    def available(cls) -> bool:
        """Whether the backing implementation is importable."""
        ...


@runtime_checkable
class FusedEstimator(Protocol):
    """The flagship: one object producing both a density and an embedding.

    Deliberately **not** a subclass of the two protocols above. It is their
    structural intersection, declared independently so that ``isinstance`` cannot
    return a misleading ``True`` when one of the methods is missing.
    """

    def fit(self, X: FloatArray) -> Self:
        """Fit on training data only."""
        ...

    def score_samples(self, X: FloatArray) -> FloatArray:
        """Return ``log p(x)``, shape ``(n,)``; larger means more likely."""
        ...

    def embed(self, X: FloatArray) -> FloatArray:
        """Return embedding coordinates, shape ``(n, n_components)``."""
        ...

    @classmethod
    def available(cls) -> bool:
        """Whether the flagship backend is importable."""
        ...


def check_log_density(name: str, logp: FloatArray) -> FloatArray:
    """Validate a log-density vector. Raises :class:`NumericalError` on garbage.

    Called at the boundary of every ``score_samples`` implementation so that a
    NaN can never travel silently into an aggregated benchmark table.
    """
    from .errors import NumericalError  # local import: keeps core.errors out of the hot path

    arr = np.asarray(logp, dtype=np.float64)
    if arr.ndim != 1:
        raise NumericalError(f"{name}.score_samples returned shape {arr.shape}, expected (n,)")
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise NumericalError(
            f"{name}.score_samples produced {n_bad}/{arr.size} non-finite values; "
            "density must be evaluated in log space via logsumexp"
        )
    return arr


__all__ = [
    "DensityEstimator",
    "FusedEstimator",
    "ManifoldEmbedder",
    "check_log_density",
]
