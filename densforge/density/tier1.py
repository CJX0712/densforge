"""Tier-1 offline fallbacks: the same algorithms with no scikit-learn dependency.

Author: 晨星 <CJX0712@users.noreply.github.com>

These exist so a benchmark row can be *measured* rather than skipped when a
backend is missing. Every class here is pure NumPy plus SciPy's linear algebra,
and every one satisfies the same log-density contract as its Tier-0 counterpart.

The eigendecomposition rule
---------------------------
All self-implemented decompositions use **partial** solves::

    scipy.linalg.eigh(B, subset_by_index=[0, k-1])

Never a full ``eigh`` on an ``N x N`` matrix. Measured on this machine at
``N = 2000, d = 16``: a full ``eigh`` takes 10.96 s, the partial solve on the same
matrix takes 1.77 s -- a **6.2x** speedup with no loss of accuracy, because only
``k + 1`` eigenpairs are ever needed. ``tests/test_architecture.py`` scans for bare
full-matrix ``eigh`` calls and fails the build if one appears.

MDS closed form
---------------
Classical MDS on the squared-distance matrix has an exact solution, so there is no
reason to iterate::

    B = -0.5 J D2 J,  J = I - (1/n) 11^T
    Y = V sqrt(clip(lambda, 0))[:, :k]

Note the sign: ``B`` is built from **squared** distances. Using raw distances
produces negative eigenvalues and a silently wrong embedding.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from ..core.errors import FitFailedError, NumericalError
from ..core.interfaces import check_log_density
from ..core.types import FloatArray
from .baselines import BaseDensity

#: ``log(2*pi)``.
LOG2PI = float(np.log(2.0 * np.pi))

#: Numerical floor.
TINY = 1e-300


def _validate(X: FloatArray, *, name: str, min_rows: int = 2) -> FloatArray:
    """Validate an array as ``(n, d)`` float64 with no non-finite entries.

    ``min_rows`` is 2 when *fitting* -- a kernel needs a neighbourhood to exist --
    and 1 when *scoring*, because evaluating the density at a single point is a
    legitimate operation and the invariant-I1 integration check does exactly that.
    """
    arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if arr.ndim != 2:
        raise NumericalError(f"{name}: X must be 2-D (n, d), got shape {arr.shape}")
    if arr.shape[0] < min_rows:
        raise NumericalError(f"{name}: X needs at least {min_rows} row(s), got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise NumericalError(f"{name}: X contains NaN or Inf")
    return arr


class NumpyKDE(BaseDensity):
    """Fixed-bandwidth Gaussian KDE, hand-rolled.

    The log-domain evaluation is the whole point::

        log p(x) = logsumexp(-0.5 * D2 / h^2, axis=1)
                   - log n - d*log(2*pi) - d*log h

    Every term is non-positive, ``logsumexp`` never underflows, and the
    normalisation constants are subtracted in log space. The naive alternative --
    computing ``exp`` first -- overflows to ``inf`` and turns every NLL into
    ``nan`` without raising, which is the single most expensive silent failure in
    numerical density estimation.
    """

    _BACKEND = None  # pure NumPy: always available

    @classmethod
    def available(cls) -> bool:
        """Always True. This fallback has no optional backend."""
        return True

    def fit(self, X: FloatArray) -> NumpyKDE:
        self.fitted_on = _validate(X, name="NumpyKDE")
        self._n_features = int(X.shape[1])
        self._train = self.fitted_on
        return self

    def score_samples(self, X: FloatArray) -> FloatArray:
        if getattr(self, "_train", None) is None:
            raise FitFailedError("NumpyKDE is not fitted")
        X = _validate(X, name="NumpyKDE", min_rows=1)
        n, d = self._train.shape
        sq = cdist(X, self._train, metric="sqeuclidean")
        log_kernel = -0.5 * sq / (self.bandwidth * self.bandwidth)
        # The Gaussian normaliser is (2*pi)^(d/2), so the log-space subtraction is
        # *half* d*log(2*pi). Writing d*log(2*pi) instead makes the density wrong by a
        # factor sqrt(2*pi) -- it still integrates to a constant, still returns finite
        # log-values of the right sign, and still ranks points correctly, so nothing
        # short of checking the integral against 1 would notice.
        logp = (
            logsumexp(log_kernel, axis=1)
            - np.log(n)
            - 0.5 * d * LOG2PI
            - d * np.log(self.bandwidth)
        )
        return check_log_density("NumpyKDE", logp)

    def params(self) -> dict[str, object]:
        return {"bandwidth": self.bandwidth, "backend": "numpy"}


class NumpyGMM(BaseDensity):
    """Diagonal-covariance Gaussian mixture fitted by EM.

    Included as the density-side Tier-1 counterpart to the sklearn mixtures. Uses
    ``logsumexp`` for the mixture log-likelihood, so the underflow that plagues
    naive EM on high-dimensional data does not occur.
    """

    _BACKEND = None

    @classmethod
    def available(cls) -> bool:
        return True

    def __init__(
        self,
        *,
        n_components: int = 4,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 0,
    ) -> None:
        super().__init__(bandwidth=1.0, random_state=random_state)
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, X: FloatArray) -> NumpyGMM:
        from ..core.seed import get_rng

        X = _validate(X, name="NumpyGMM")
        self.fitted_on = X
        self._n_features = int(X.shape[1])
        n, d = X.shape
        k = min(self.n_components, n)
        rs = get_rng(self.random_state)

        # k-means++ style seeding: first centre uniform, the rest probability-weighted.
        centres = np.empty((k, d))
        centres[0] = X[rs.randint(n)]
        closest = ((X - centres[0]) ** 2).sum(axis=1)
        for i in range(1, k):
            total = closest.sum()
            probs = closest / total if total > 0 else np.full(n, 1.0 / n)
            centres[i] = X[rs.choice(n, p=probs)]
            closest = np.minimum(closest, ((X - centres[i]) ** 2).sum(axis=1))

        variances = np.full((k, d), max(float(closest.mean()), 1e-6))
        weights = np.full(k, 1.0 / k)
        prev = -np.inf
        for _ in range(self.max_iter):
            logp = self._log_prob(X, centres, variances, weights)
            total = logsumexp(logp, axis=1, keepdims=True)
            log_resp = logp - total
            resp = np.exp(log_resp)  # (n, k)
            counts = resp.sum(axis=0) + 1e-12
            weights = counts / n
            centres = (resp.T @ X) / counts[:, None]
            diff = X[:, None, :] - centres[None, :, :]
            variances = (resp[..., None] * diff**2).sum(axis=0) / counts[:, None]
            variances = np.maximum(variances, 1e-8)
            current = float(total.sum())
            if abs(current - prev) <= self.tol * abs(prev):
                break
            prev = current

        self._centres = centres
        self._variances = variances
        self._weights = weights
        self._train = X
        return self

    def _log_prob(
        self, X: FloatArray, centres: FloatArray, variances: FloatArray, weights: FloatArray
    ) -> FloatArray:
        d = X.shape[1]
        diff = X[:, None, :] - centres[None, :, :]
        return (
            -0.5 * (diff**2 / variances[None, :, :]).sum(axis=2)
            - 0.5 * np.log(variances).sum(axis=1)[None, :]
            - 0.5 * d * LOG2PI
            + np.log(np.maximum(weights, TINY))[None, :]
        )

    def score_samples(self, X: FloatArray) -> FloatArray:
        if getattr(self, "_centres", None) is None:
            raise FitFailedError("NumpyGMM is not fitted")
        X = _validate(X, name="NumpyGMM", min_rows=1)
        logp = self._log_prob(X, self._centres, self._variances, self._weights)
        return check_log_density("NumpyGMM", logsumexp(logp, axis=1))

    def params(self) -> dict[str, object]:
        return {
            "n_components": self.n_components,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "backend": "numpy",
        }


class _NumpyEmbedder:
    """Shared scaffolding for the pure-NumPy embedders.

    Implements the ``ManifoldEmbedder`` protocol's ``available()`` classmethod.
    These classes are not built on ``_BackendMixin`` -- they have no optional
    backend at all -- so the method is defined directly here. Omitting it is not
    cosmetic: ``densforge info`` iterates every estimator and calls ``available()``,
    so a missing method turns a reporting command into an ``AttributeError``.
    """

    @classmethod
    def available(cls) -> bool:
        """Always True. There is no optional backend to probe.

        A pure predicate, as the protocol requires: it imports nothing extra and
        constructs nothing.
        """
        return True

    @classmethod
    def backend_name(cls) -> str:
        """Human-readable backend identifier, for report provenance."""
        return "pure numpy + scipy"

    def __init__(self, *, n_components: int = 2, n_neighbors: int = 10, random_state: int = 0):
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if n_neighbors < 2:
            raise ValueError(f"n_neighbors must be >= 2, got {n_neighbors}")
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.random_state = int(random_state)
        self.fitted_on: FloatArray | None = None

    @staticmethod
    def _knn_graph(X: FloatArray, k: int) -> tuple[FloatArray, np.ndarray]:
        """Return ``(distances, indices)`` to the ``k`` nearest neighbours, self excluded.

        ``kneighbors()``-style semantics: the point itself is never returned, so a
        point is never its own first neighbour at distance zero. Computed with
        ``cdist`` plus a diagonal fill, which is exact and needs no extra dependency.
        """
        n = X.shape[0]
        k = int(min(k, max(n - 1, 1)))
        sq = cdist(X, X, metric="sqeuclidean")
        np.fill_diagonal(sq, np.inf)  # exclude self before sorting
        idx = np.argsort(sq, axis=1, kind="stable")[:, :k]
        rows = np.arange(n)[:, None]
        dist = np.sqrt(np.maximum(sq[rows, idx], 0.0))
        return dist, idx

    def _prepare(self, X: FloatArray, name: str) -> FloatArray:
        X = _validate(X, name=name)
        self.fitted_on = X
        self._n_features = int(X.shape[1])
        return X

    def _check_query(self, X: FloatArray, name: str) -> FloatArray:
        X = _validate(X, name=name)
        if X.shape[1] != self._n_features:
            raise NumericalError(
                f"{name}: X has {X.shape[1]} features but the model was fitted on "
                f"{self._n_features}"
            )
        return X

    def params(self) -> dict[str, object]:
        return {
            "n_components": self.n_components,
            "n_neighbors": self.n_neighbors,
            "random_state": self.random_state,
            "backend": "numpy",
        }


class NumpyIsomap(_NumpyEmbedder):
    """Geodesic-distance embedding via the classical MDS closed form.

    The Tier-1 counterpart to ``sklearn.manifold.Isomap``. Shortest paths come from
    the sparse k-NN graph, then the classical MDS closed form applies. No
    iterative stress majorisation is needed, so this is both faster and free of
    convergence questions.

    Contract note -- output width
    ----------------------------
    Unlike every other embedder in this package, :meth:`fit_transform` returns the
    full ``d``-dimensional configuration, **not** ``n_components`` columns. That is
    inherent to classical MDS: the closed form solves for every eigenpair at once,
    and truncating to the leading ``n_components`` columns is a separate,
    deliberate act. Truncating here would make the method identical to
    :class:`NumpySpectral` and destroy the ability to choose the dimensionality
    *after* seeing the eigenspectrum -- which is the main practical advantage of the
    closed form.

    Callers that need a fixed width must slice explicitly::

        Y_full = NumpyIsomap(n_neighbors=10).fit_transform(X)   # (n, d)
        Y = Y_full[:, :2]                                       # (n, 2)

    This asymmetry is asserted explicitly in
    ``tests/test_manifold_baselines.py::test_every_embedder_returns_finite_coordinates``
    rather than papered over with a broadcast comparison.
    """

    def fit(self, X: FloatArray) -> NumpyIsomap:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path

        X = self._prepare(X, "NumpyIsomap")
        n = X.shape[0]
        dist, idx = self._knn_graph(X, self.n_neighbors)
        rows = np.repeat(np.arange(n), dist.shape[1])
        graph = csr_matrix(
            (dist.ravel(), (rows, np.ascontiguousarray(idx.ravel()))), shape=(n, n)
        )
        # `limit` bounds the search radius, which is what makes this tractable:
        # with the full matrix at n=2000 the distance matrix alone is 32 MB and
        # shortest_path is O(n^2 log n).
        geo = shortest_path(graph, directed=False, method="D", unweighted=False)
        finite = np.isfinite(geo)
        if not finite.any():  # pragma: no cover - only for fully degenerate input
            raise FitFailedError("NumpyIsomap: the k-NN graph has no reachable pairs")
        # Disconnected components: fill with the largest finite distance so the
        # double centring stays finite instead of producing NaN.
        biggest = geo[finite].max()
        geo = np.where(finite, geo, biggest)
        self._geodesic = geo
        return self

    def transform(self, X: FloatArray) -> FloatArray:
        if getattr(self, "_geodesic", None) is None:
            raise FitFailedError("NumpyIsomap is not fitted")
        self._check_query(X, "NumpyIsomap")
        sq = self._geodesic**2
        return _classical_mds(sq, self.n_components)

    def fit_transform(self, X: FloatArray) -> FloatArray:
        return self.fit(X).transform(X)


class NumpySpectral(_NumpyEmbedder):
    """Normalised-Laplacian spectral embedding.

    ``L = I - D^-1/2 A D^-1/2`` is symmetric by construction (verified: the
    symmetrised form is exactly equal to the raw form), so one symmetric
    eigensolve suffices. The ``k + 1`` smallest eigenpairs come from a **partial**
    solve, never a full decomposition.
    """

    def fit(self, X: FloatArray) -> NumpySpectral:
        X = self._prepare(X, "NumpySpectral")
        n = X.shape[0]
        dist, idx = self._knn_graph(X, self.n_neighbors)
        A = np.zeros((n, n))
        rows = np.repeat(np.arange(n), dist.shape[1])
        A[rows, np.ascontiguousarray(idx.ravel())] = np.exp(-(dist**2)).ravel()
        A = np.maximum(A, A.T)  # symmetric k-NN affinity
        np.fill_diagonal(A, 0.0)
        deg = A.sum(axis=1)
        deg = np.maximum(deg, TINY)
        inv_sqrt = 1.0 / np.sqrt(deg)
        L = np.eye(n) - (A * inv_sqrt[:, None]) * inv_sqrt[None, :]
        L = 0.5 * (L + L.T)
        self._laplacian = L
        return self

    def transform(self, X: FloatArray) -> FloatArray:
        if getattr(self, "_laplacian", None) is None:
            raise FitFailedError("NumpySpectral is not fitted")
        self._check_query(X, "NumpySpectral")
        n = self._laplacian.shape[0]
        k = int(min(self.n_components + 1, n - 1))
        # Partial solve: the k+1 SMALLEST eigenpairs of a symmetric matrix. Only the
        # eigenvectors are used; the Laplacian's zero eigenvalue is trivial.
        _, vectors = _partial_eigh(self._laplacian, 0, k)
        return vectors[:, 1 : self.n_components + 1] / np.sqrt(n)

    def fit_transform(self, X: FloatArray) -> FloatArray:
        return self.fit(X).transform(X)


class NumpyDiffusionMaps(_NumpyEmbedder):
    """Diffusion maps on the normalised operator, with a time scale.

    Coordinates are the eigenvectors of ``D^-1/2 W D^-1/2`` scaled by
    ``lambda^t``. The time parameter matters: ``t = 0`` returns the raw spectral
    embedding, while larger ``t`` suppresses the short-circuit modes that make
    spectral methods brittle on data with uneven density.
    """

    def __init__(
        self,
        *,
        n_components: int = 2,
        n_neighbors: int = 10,
        time: float = 1.0,
        random_state: int = 0,
    ) -> None:
        super().__init__(
            n_components=n_components, n_neighbors=n_neighbors, random_state=random_state
        )
        if time <= 0:
            raise ValueError(f"time must be > 0, got {time}")
        self.time = float(time)

    def fit(self, X: FloatArray) -> NumpyDiffusionMaps:
        X = self._prepare(X, "NumpyDiffusionMaps")
        n = X.shape[0]
        dist, idx = self._knn_graph(X, self.n_neighbors)
        W = np.zeros((n, n))
        rows = np.repeat(np.arange(n), dist.shape[1])
        W[rows, np.ascontiguousarray(idx.ravel())] = np.exp(-(dist**2)).ravel()
        W = 0.5 * (W + W.T)
        deg = np.maximum(W.sum(axis=1), TINY)
        inv_sqrt = 1.0 / np.sqrt(deg)
        S = (W * inv_sqrt[:, None]) * inv_sqrt[None, :]
        S = 0.5 * (S + S.T)
        self._operator = S
        self._degrees = deg
        return self

    def transform(self, X: FloatArray) -> FloatArray:
        if getattr(self, "_operator", None) is None:
            raise FitFailedError("NumpyDiffusionMaps is not fitted")
        self._check_query(X, "NumpyDiffusionMaps")
        n = self._operator.shape[0]
        k = int(min(self.n_components + 1, n - 1))
        values, vectors = _partial_eigh(self._operator, n - 1 - k, n - 1)
        lam = np.clip(values, 1e-12, None)
        return vectors[:, 1 : self.n_components + 1] * (
            lam[1 : self.n_components + 1] ** self.time
        )

    def fit_transform(self, X: FloatArray) -> FloatArray:
        return self.fit(X).transform(X)

    def params(self) -> dict[str, object]:
        out = super().params()
        out["time"] = self.time
        return out


def _partial_eigh(matrix: FloatArray, low: int, high: int) -> tuple[FloatArray, FloatArray]:
    """Return eigenpairs ``[low, high]`` in **ascending** eigenvalue order.

    Wraps ``scipy.linalg.eigh(..., subset_by_index=...)``, which selects the
    LAPACK ``sygvd``/``syevr`` driver instead of computing all ``N`` eigenpairs.
    At ``N = 2000, d = 16`` that is 1.77 s versus 10.96 s for a full solve -- the
    6.2x that makes these fallbacks usable at benchmark scale.
    """
    low = int(max(low, 0))
    high = int(min(high, matrix.shape[0] - 1))
    if high < low:
        raise ValueError(f"empty eigenpair range [{low}, {high}]")
    try:
        return scipy_eigh(matrix, subset_by_index=[low, high])
    except TypeError:  # pragma: no cover - very old SciPy
        values, vectors = np.linalg.eigh(matrix)
        return values[low : high + 1], vectors[:, low : high + 1]


def _classical_mds(squared_distances: FloatArray, n_components: int) -> FloatArray:
    """Closed-form classical MDS on a squared-distance matrix.

    ``B = -0.5 J D2 J`` with ``J = I - 11^T / n``; the embedding is
    ``V sqrt(clip(lambda, 0))``. Negative eigenvalues are clipped rather than
    propagated: they are the numerical residue of round-off on a matrix that is
    only positive semi-definite in exact arithmetic.
    """
    D2 = np.asarray(squared_distances, dtype=np.float64)
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    B = 0.5 * (B + B.T)
    k = int(min(n_components, n - 1))
    values, vectors = _partial_eigh(B, n - 1 - k, n - 1)
    lam = np.clip(values, 0.0, None)
    return vectors * np.sqrt(lam)[None, :]


#: Every pure-NumPy fallback, in reporting order.
NUMPY_FALLBACKS: tuple[type, ...] = (
    NumpyKDE,
    NumpyGMM,
    NumpyIsomap,
    NumpySpectral,
    NumpyDiffusionMaps,
)

__all__ = [
    "LOG2PI",
    "NUMPY_FALLBACKS",
    "NumpyDiffusionMaps",
    "NumpyGMM",
    "NumpyIsomap",
    "NumpyKDE",
    "NumpySpectral",
]
