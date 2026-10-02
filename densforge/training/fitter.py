"""Fit bookkeeping and the train-only provenance record.

Author: 晨星 <CJX0712@users.noreply.github.com>

This layer exists to hold one fact that the leakage assertions need: **exactly what
was fitted on what.** Every estimator is wrapped so that the wrapper -- not the
estimator -- owns that record, and so that a metric cannot be reported for a model
whose provenance does not check out.

Layer position
--------------
L3. It references density and manifold estimators only through the
:class:`~densforge.core.interfaces` protocols, so the static dependency graph stays
acyclic: ``core`` is the sink, and resolving a concrete class happens at call time
via the ``available()`` factory.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..core.errors import FitFailedError
from ..core.types import FloatArray, array_fingerprint


@runtime_checkable
class FittedModel(Protocol):
    """An estimator plus the provenance assertion A3 depends on."""

    #: The exact array this model was fitted on. Compared by identity.
    fitted_on: FloatArray
    #: Content digests of everything fitted on. A *copy* of the test array has a
    #: different ``id()`` but the same digest, so the digest is the real check.
    fitted_fingerprints: frozenset[str]

    def score_samples(self, X: FloatArray) -> FloatArray:
        """Return ``log p(x)``, shape ``(n,)``; larger means more likely."""
        ...


class TrainOnlyModel:
    """Wraps an estimator and records what it was fitted on.

    The wrapper adds no mathematics. It exists so that provenance cannot be
    forgotten: a bare estimator that forgets to set ``fitted_on`` would make
    assertion A3 silently vacuous, and a vacuous leakage check is worse than none
    because it looks like a pass.

    Examples
    --------
    >>> from densforge.training.fitter import fit_density
    >>> from densforge.density.baselines import SkKernelDensity
    >>> import numpy as np
    >>> X = np.random.RandomState(0).randn(100, 2)
    >>> model = fit_density(SkKernelDensity(bandwidth=1.0), X)
    >>> model.fitted_on is X
    True
    >>> model.fitted_fingerprints
    frozenset({...})
    """

    __slots__ = ("_estimator", "_fingerprints", "_n_features", "fitted_on")

    def __init__(self, estimator: Any, X: FloatArray) -> None:
        self._estimator = estimator
        self.fitted_on = X
        self._fingerprints = frozenset({array_fingerprint(X)})
        self._n_features = int(X.shape[1])

    @property
    def estimator(self) -> Any:
        """The wrapped estimator."""
        return self._estimator

    @property
    def fitted_fingerprints(self) -> frozenset[str]:
        """Content digests of everything this model was fitted on."""
        return self._fingerprints

    @property
    def name(self) -> str:
        return type(self._estimator).__name__

    def score_samples(self, X: FloatArray) -> FloatArray:
        """Delegate to the wrapped estimator's log-density."""
        return self._estimator.score_samples(X)

    def transform(self, X: FloatArray | None = None) -> FloatArray:
        """Delegate to the wrapped estimator's embedding, when it has one."""
        if not hasattr(self._estimator, "transform"):
            raise FitFailedError(f"{self.name} is a density estimator and cannot embed")
        return self._estimator.transform() if X is None else self._estimator.transform(X)

    def params(self) -> dict[str, Any]:
        """Return the estimator's hyper-parameters, for report provenance."""
        getter = getattr(self._estimator, "params", None)
        return dict(getter()) if callable(getter) else {}

    def __repr__(self) -> str:
        return f"TrainOnlyModel({self.name}, n={self.fitted_on.shape[0]}, d={self._n_features})"


def fit_density(estimator: Any, X: FloatArray) -> TrainOnlyModel:
    """Fit a density estimator and wrap it with provenance.

    Parameters
    ----------
    estimator:
        Anything with ``fit`` and ``score_samples``. Its ``available()`` classmethod
        is consulted first; a class reporting ``False`` is refused here rather than
        at construction, so a missing backend is skipped with a reason instead of
        crashing mid-benchmark.
    X:
        Training samples.

    Returns
    -------
    TrainOnlyModel

    Raises
    ------
    BackendUnavailableError
        If ``available()`` is False.
    FitFailedError
        If the estimator has no ``fit``, or the fit produced no usable state.
    """
    from ..core.errors import BackendUnavailableError

    available = getattr(type(estimator), "available", None)
    if callable(available) and not available():
        raise BackendUnavailableError(
            f"{type(estimator).__name__}.available() returned False; "
            "mark the row as skipped instead of forcing construction"
        )
    if not hasattr(estimator, "fit"):
        raise FitFailedError(f"{type(estimator).__name__} has no fit() method")
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim != 2:
        raise FitFailedError(f"X must be 2-D, got shape {X.shape}")
    estimator.fit(X)
    if not hasattr(estimator, "score_samples"):
        raise FitFailedError(
            f"{type(estimator).__name__} has no score_samples(); it is not a density estimator"
        )
    return TrainOnlyModel(estimator, X)


def fit_embedder(estimator: Any, X: FloatArray) -> TrainOnlyModel:
    """Fit a manifold estimator and wrap it with provenance.

    Returns
    -------
    TrainOnlyModel
        The wrapper exposes ``transform`` rather than ``score_samples``; callers
        check with :func:`hasattr` rather than assuming both.
    """
    from ..core.errors import BackendUnavailableError

    available = getattr(type(estimator), "available", None)
    if callable(available) and not available():
        raise BackendUnavailableError(
            f"{type(estimator).__name__}.available() returned False; "
            "mark the row as skipped instead of forcing construction"
        )
    if not hasattr(estimator, "fit"):
        raise FitFailedError(f"{type(estimator).__name__} has no fit() method")
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim != 2:
        raise FitFailedError(f"X must be 2-D, got shape {X.shape}")
    estimator.fit(X)
    return TrainOnlyModel(estimator, X)


__all__ = ["FittedModel", "TrainOnlyModel", "fit_density", "fit_embedder"]
