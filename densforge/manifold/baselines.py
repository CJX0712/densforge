"""Manifold-learning baselines: scikit-learn adapters plus the Tier-1 fallbacks.

Author: 晨星 <CJX0712@users.noreply.github.com>

scikit-learn 1.9 API drift, verified on this machine
----------------------------------------------------
============================  ==========================================
class                        change
============================  ==========================================
``Isomap``                    has **no** ``random_state``. Passing one
                              raises ``TypeError: Isomap.__init__() got an
                              unexpected keyword argument 'random_state'``.
                              It does not need one: the underlying ARPACK
                              solver's default start vector is deterministic.
``TSNE``                      ``max_iter`` replaced ``n_iter``. The old name
                              raises ``TypeError``.
``MDS``                       ``normalized_stress`` is new in 1.9 and changes
                              the optimisation objective, so results are not
                              comparable with older defaults. Its ``n_init``
                              default measures **1** on this machine, not the
                              4 quoted in the design document.
``MDS`` cost                  the dominant budget item. It is O(n^2 * iters)
                              and is mitigated by ``n_init=1``, a low
                              ``max_iter``, and subsampling the training rows.
``SpectralEmbedding``         ``gamma=None`` defaults to ``1 / n_neighbors``,
                              which changes meaning whenever ``n_neighbors``
                              changes, making two experiments incomparable.
                              Always pass it explicitly.
``trustworthiness``           signature is ``(X, X_embedded, *, n_neighbors)``.
                              **Both** arguments are required. Calling
                              ``trustworthiness(Y)`` raises ``TypeError``.
============================  ==========================================

Every constructor call below is keyword-only, and every estimator receives
``n_jobs=1``: multi-threaded reductions sum in a nondeterministic order, which
perturbs the last bits and makes bit-exact reproducibility impossible.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.errors import FitFailedError, NumericalError
from ..core.types import FloatArray
from ..density.baselines import _BackendMixin

#: Neighbourhood size used for every trustworthiness number in the project.
#:
#: Fixed globally on purpose: the metric's value depends on ``n_neighbors``, so
#: two experiments with different ``k`` are not comparable. Reports must state it.
TRUSTWORTHNESS_K = 5


class BaseManifold(_BackendMixin):
    """Common scaffolding for manifold-learning estimators."""

    #: Embedding dimensionality produced by :meth:`transform`.
    n_components: int = 2

    def __init__(
        self,
        *,
        n_components: int = 2,
        n_neighbors: int = 10,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if n_neighbors < 2:
            raise ValueError(f"n_neighbors must be >= 2, got {n_neighbors}")
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.random_state = int(random_state)
        self.extra_params = dict(kwargs)
        self._estimator: Any = None
        self._embedding: FloatArray | None = None
        self.fitted_on: FloatArray | None = None
        self._n_features: int | None = None

    def _attach(self, X: FloatArray) -> FloatArray:
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise NumericalError(f"X must be 2-D (n, d), got shape {X.shape}")
        if X.shape[0] < 3:
            raise NumericalError(f"X needs at least 3 rows, got {X.shape[0]}")
        if not np.isfinite(X).all():
            raise NumericalError("X contains NaN or Inf")
        # A neighbourhood cannot exceed n-1 once the point itself is excluded.
        self.n_neighbors = int(min(self.n_neighbors, X.shape[0] - 1))
        self.fitted_on = X
        self._n_features = int(X.shape[1])
        return X

    def _check_query(self, X: FloatArray) -> FloatArray:
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise NumericalError(f"X must be 2-D (n, d), got shape {X.shape}")
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise NumericalError(
                f"X has {X.shape[1]} features but the model was fitted on {self._n_features}"
            )
        return X

    def _run(self, X: FloatArray) -> FloatArray:
        try:
            return np.asarray(self._estimator.fit_transform(X), dtype=np.float64)
        except Exception as exc:
            raise FitFailedError(f"{type(self).__name__}.fit failed: {exc}") from exc

    def transform(self, X: FloatArray) -> FloatArray:
        """Return embedding coordinates for ``X``, shape ``(n, n_components)``.

        Two distinct cases, deliberately *not* merged:

        * the training data is passed → return the cached ``fit_transform`` result,
          which is free and is what every metric in this project actually needs;
        * anything else → delegate to the backend's own ``transform``.

        Silently returning the training embedding for *new* points would be wrong:
        it would answer a different question and produce a plausible-looking number.
        Backends without a ``transform`` (scikit-learn 1.9's ``MDS``) override this
        method and raise instead.
        """
        if self._estimator is None:
            raise FitFailedError(f"{type(self).__name__} is not fitted")
        X = self._check_query(X)
        # Querying with the training data itself returns the cached embedding, which
        # is what the pipeline does and is cheaper than re-running the backend.
        if (
            self._embedding is not None
            and X.shape == self.fitted_on.shape
            and np.array_equal(X, self.fitted_on)
        ):
            return self._embedding
        delegate = getattr(self._estimator, "transform", None)
        if delegate is None:
            raise NumericalError(
                f"{type(self).__name__}: the scikit-learn backend has no transform(); "
                "out-of-sample coordinates are not available for this method"
            )
        try:
            return np.asarray(delegate(X), dtype=np.float64)
        except Exception as exc:
            raise FitFailedError(f"{type(self).__name__}.transform failed: {exc}") from exc

    def fit_transform(self, X: FloatArray) -> FloatArray:
        """Return the embedding produced by the preceding :meth:`fit`.

        ``X`` is accepted for interface symmetry with the scikit-learn estimators and
        validated, but the coordinates are already stored, so it is not re-used.
        """
        if self._embedding is None:
            raise FitFailedError(f"{type(self).__name__} is not fitted")
        self._check_query(X)
        return self._embedding

    @property
    def fitted_fingerprints(self) -> frozenset[str]:
        from ..core.types import array_fingerprint

        if self.fitted_on is None:
            return frozenset()
        return frozenset({array_fingerprint(self.fitted_on)})

    def params(self) -> dict[str, Any]:
        out = {
            "n_components": self.n_components,
            "n_neighbors": self.n_neighbors,
            "random_state": self.random_state,
        }
        out.update(self.extra_params)
        return out


class SkIsomap(BaseManifold):
    """Isomap: geodesic shortest paths followed by metric MDS.

    The strongest classical manifold baseline on genuinely folded geometry
    (measured trustworthiness 0.9997 on a swiss roll, against 0.9573 for PCA).
    Note that ``Isomap`` has no ``random_state`` on scikit-learn 1.9 and none is
    passed.
    """

    _BACKEND = ("sklearn.manifold", "Isomap")

    def __init__(
        self, *, n_components: int = 2, n_neighbors: int = 10, path_method: str = "auto"
    ) -> None:
        super().__init__(n_components=n_components, n_neighbors=n_neighbors)
        self.path_method = path_method

    def fit(self, X: FloatArray) -> SkIsomap:
        X = self._attach(X)
        cls = self.require()
        # No random_state: Isomap.__init__ has no such parameter on sklearn 1.9.
        self._estimator = cls(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            path_method=self.path_method,
            n_jobs=1,
        )
        self._embedding = self._run(X)
        return self

    def params(self) -> dict[str, Any]:
        out = super().params()
        out.pop("random_state", None)
        out["path_method"] = self.path_method
        return out


class SkLLE(BaseManifold):
    """Locally Linear Embedding.

    ``method='modified'`` is the default here: the standard formulation's local
    weights are only defined up to a null direction, and on low-rank neighbourhoods
    that produces degenerate eigenvalues with no useful embedding.
    """

    _BACKEND = ("sklearn.manifold", "LocallyLinearEmbedding")

    def __init__(
        self,
        *,
        n_components: int = 2,
        n_neighbors: int = 10,
        method: str = "modified",
        reg: float = 1e-3,
        random_state: int = 0,
    ) -> None:
        super().__init__(
            n_components=n_components, n_neighbors=n_neighbors, random_state=random_state
        )
        self.method = method
        self.reg = float(reg)

    def fit(self, X: FloatArray) -> SkLLE:
        X = self._attach(X)
        cls = self.require()
        self._estimator = cls(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            method=self.method,
            reg=self.reg,
            random_state=self.random_state,
            n_jobs=1,
        )
        self._embedding = self._run(X)
        return self

    def params(self) -> dict[str, Any]:
        out = super().params()
        out["method"] = self.method
        out["reg"] = self.reg
        return out


class SkSpectralEmbedding(BaseManifold):
    """Spectral embedding with an explicit affinity bandwidth.

    ``gamma`` defaults to ``1 / n_neighbors`` in scikit-learn, and that default
    *changes value* whenever ``n_neighbors`` changes, so two runs with different
    neighbourhoods are not comparable even though nothing else differs. It is set
    explicitly from the data's own median neighbour distance here, and recorded.

    This is also the closest baseline to the flagship's manifold half, which makes
    it the meaningful control for the density-balancing axis: it normalises the
    graph but has no density channel.
    """

    _BACKEND = ("sklearn.manifold", "SpectralEmbedding")

    def __init__(
        self,
        *,
        n_components: int = 2,
        n_neighbors: int = 10,
        affinity: str = "nearest_neighbors",
        gamma: float | None = None,
        random_state: int = 0,
    ) -> None:
        super().__init__(
            n_components=n_components, n_neighbors=n_neighbors, random_state=random_state
        )
        self.affinity = affinity
        self.gamma = gamma

    def _resolve_gamma(self, X: FloatArray) -> float:
        """Pick ``1 / (2 * median_neighbour_dist^2)``, the usual RBF scale."""
        if self.gamma is not None:
            return float(self.gamma)
        from sklearn.neighbors import NearestNeighbors

        k = int(min(self.n_neighbors + 1, X.shape[0]))
        dist, _ = NearestNeighbors(n_neighbors=k, n_jobs=1).fit(X).kneighbors(X)
        median = float(np.median(np.maximum(dist[:, 1:], 1e-12)))
        return 1.0 / (2.0 * median**2)

    def fit(self, X: FloatArray) -> SkSpectralEmbedding:
        X = self._attach(X)
        cls = self.require()
        self._resolved_gamma = self._resolve_gamma(X)
        self._estimator = cls(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            affinity=self.affinity,
            gamma=self._resolved_gamma,
            random_state=self.random_state,
            n_jobs=1,
        )
        self._embedding = self._run(X)
        return self

    def params(self) -> dict[str, Any]:
        out = super().params()
        out["affinity"] = self.affinity
        out["gamma"] = getattr(self, "_resolved_gamma", self.gamma)
        return out


class SkPCA(BaseManifold):
    """Linear baseline: the top principal components.

    Not a manifold learner, and included precisely because it is not. On data
    whose environment map is linear, PCA is *theoretically optimal* -- measured
    trustworthiness 0.9887 against Isomap's 0.9863 on a linearly mapped dataset.
    Any flagship claim that cannot beat PCA there is not a claim about manifolds.
    """

    _BACKEND = ("sklearn.decomposition", "PCA")

    def __init__(self, *, n_components: int = 2, svd_solver: str = "auto") -> None:
        super().__init__(n_components=n_components)
        self.svd_solver = svd_solver

    def fit(self, X: FloatArray) -> SkPCA:
        X = self._attach(X)
        cls = self.require()
        self._estimator = cls(n_components=self.n_components, svd_solver=self.svd_solver)
        self._embedding = self._run(X)
        return self

    def params(self) -> dict[str, Any]:
        out = super().params()
        out.pop("n_neighbors", None)
        out.pop("random_state", None)
        out["svd_solver"] = self.svd_solver
        return out


class SkMDS(BaseManifold):
    """Metric MDS on raw distances. The project's budget black hole.

    Cost is O(n^2 * max_iter). On this machine ``n_init=4`` at ``n=1500`` took
    15.84 s, enough on its own to blow a 60 s budget for the whole suite. Mitigated
    three ways: ``n_init=1`` (explicit, because the scikit-learn default is 1 on
    1.9 and relying on that is how a dependency bump quadruples the runtime),
    a low ``max_iter``, and subsampling the training rows before fitting.

    Two scikit-learn 1.9 facts this class works around:

    * ``MDS`` has **no** ``transform`` method on this version -- verified by
      ``hasattr``. Out-of-sample coordinates would require re-running SMACOF from
      the new point, which is expensive and is not implemented. :meth:`transform`
      therefore raises a typed error rather than delegating to a method that is
      not there.
    * ``max_iter`` must be an int; a float raises
      ``InvalidParameterError``.
    """

    _BACKEND = ("sklearn.manifold", "MDS")

    def __init__(
        self,
        *,
        n_components: int = 2,
        n_init: int = 1,
        max_iter: int = 100,
        normalized_stress: bool = False,
        subsample: int | None = 600,
        random_state: int = 0,
    ) -> None:
        super().__init__(n_components=n_components, n_neighbors=2, random_state=random_state)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.normalized_stress = bool(normalized_stress)
        self.subsample = subsample

    def fit(self, X: FloatArray) -> SkMDS:
        X = self._attach(X)
        cls = self.require()
        work = X
        if self.subsample is not None and X.shape[0] > self.subsample:
            # Deterministic stride, not a random draw: reproducibility is not
            # negotiable, and a fixed stride is also easier to reason about.
            step = max(X.shape[0] // self.subsample, 1)
            work = np.ascontiguousarray(X[::step][: self.subsample])
        self._fit_shape = tuple(work.shape)
        self._estimator = cls(
            n_components=self.n_components,
            n_init=self.n_init,  # explicit: the library default is 1 on 1.9
            max_iter=int(self.max_iter),
            normalized_stress=self.normalized_stress,  # new in 1.9; record it
            metric="euclidean",  # `dissimilarity` is deprecated in 1.9, removed in 1.10
            n_jobs=1,
            random_state=self.random_state,
        )
        self._embedding = self._run(work)
        return self

    def transform(self, X: FloatArray) -> FloatArray:  # noqa: ARG002 -- always raises
        """Always raises: scikit-learn 1.9's ``MDS`` has no ``transform``.

        Producing out-of-sample MDS coordinates means re-solving SMACOF with the
        new points held fixed, which is a different algorithm from the one this
        baseline is here to represent. Returning the training embedding for new
        inputs -- the tempting shortcut -- would silently answer a different
        question, so this is refused instead.
        """
        raise NumericalError(
            "MDS on scikit-learn 1.9 has no transform(); out-of-sample metric MDS is "
            "not implemented. Use fit_transform on the training data, or "
            "densforge.density.flagship.DensFuse.embed for out-of-sample coordinates"
        )

    def params(self) -> dict[str, Any]:
        out = super().params()
        out.pop("n_neighbors", None)
        out.update(
            {
                "n_init": self.n_init,
                "max_iter": self.max_iter,
                "normalized_stress": self.normalized_stress,
                "subsample": self.subsample,
            }
        )
        return out


class SkTSNE(BaseManifold):
    """t-SNE, as a visual reference only.

    Two reasons it is not an algorithm baseline. First, it does not preserve
    distances, so only the rank-based trustworthiness metric is meaningful on its
    output. Second, the Barnes-Hut approximation sums in a floating-point order
    that depends on the tree layout, so results are only reproducible with a fixed
    ``random_state`` **and** ``n_jobs=1``; both are set here.

    ``max_iter`` is the 1.9 spelling of the old ``n_iter``, and scikit-learn
    validates it against a floor of 250 -- passing less raises
    ``InvalidParameterError``, so the value is clamped rather than allowed to
    abort a run for a reason unrelated to the method's quality.
    """

    _BACKEND = ("sklearn.manifold", "TSNE")

    #: scikit-learn validates ``max_iter >= 250``; below that it raises.
    MIN_MAX_ITER = 250

    def __init__(
        self,
        *,
        n_components: int = 2,
        perplexity: float = 30.0,
        max_iter: int = 500,
        init: str = "pca",
        method: str = "exact",
        random_state: int = 0,
    ) -> None:
        super().__init__(n_components=n_components, n_neighbors=2, random_state=random_state)
        self.perplexity = float(perplexity)
        self.max_iter = int(max(max_iter, self.MIN_MAX_ITER))
        self.init = init
        self.method = method

    def _clamp_perplexity(self, n: int) -> float:
        """t-SNE requires ``perplexity < n``; clamp instead of raising."""
        return float(min(self.perplexity, max(5.0, (n - 1) / 3.0)))

    def fit(self, X: FloatArray) -> SkTSNE:
        X = self._attach(X)
        cls = self.require()
        self._estimator = cls(
            n_components=self.n_components,
            perplexity=self._clamp_perplexity(X.shape[0]),
            max_iter=self.max_iter,  # 1.9 name; `n_iter` no longer exists
            init=self.init,
            method=self.method,
            random_state=self.random_state,
            n_jobs=1,  # Barnes-Hut summation order depends on the thread count
        )
        self._embedding = self._run(X)
        return self

    def params(self) -> dict[str, Any]:
        out = super().params()
        out.pop("n_neighbors", None)
        out.update(
            {
                "perplexity": self.perplexity,
                "max_iter": self.max_iter,
                "init": self.init,
                "method": self.method,
            }
        )
        return out


#: Every manifold baseline, in reporting order.
MANIFOLD_BASELINES: tuple[type[BaseManifold], ...] = (
    SkPCA,
    SkIsomap,
    SkLLE,
    SkSpectralEmbedding,
    SkMDS,
    SkTSNE,
)

__all__ = [
    "MANIFOLD_BASELINES",
    "TRUSTWORTHNESS_K",
    "BaseManifold",
    "SkIsomap",
    "SkLLE",
    "SkMDS",
    "SkPCA",
    "SkSpectralEmbedding",
    "SkTSNE",
]
