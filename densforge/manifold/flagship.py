"""Manifold-side flagship: :class:`ManifoldFuse`.

Author: 晨星 <CJX0712@users.noreply.github.com`

Relationship to :class:`~densforge.density.flagship.DensFuse`
------------------------------------------------------------
``DensFuse`` is the full closed loop: it estimates a density *and* an embedding,
and the two feed each other. ``ManifoldFuse`` is the ablation that isolates the
manifold half -- it produces an embedding whose density-balancing term comes from
a leave-one-out self-tuning KDE rather than from the flagship's own density model.

Keeping it as a separate class rather than a flag on ``DensFuse`` is what makes the
comparison interpretable. It answers a specific question: *how much of the manifold
performance comes from the density channel, and how much from the balancing
mechanism alone?* A flag would conflate the two.

Like the flagship, this module imports no baseline class.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..core.errors import BackendUnavailableError, FitFailedError, ShapeMismatchError
from ..core.interfaces import ManifoldEmbedder
from ..core.types import FloatArray, array_fingerprint
from .diffusionmaps import (
    centre_and_clip_ell,
    density_balance,
    diffusion_coordinates,
)

EPS = 1e-12
TINY = 1e-300


class ManifoldFuse:
    """Density-balanced diffusion embedding without a parametric density model.

    The graph weights come from a self-tuning isotropic kernel
    ``exp(-d_ij^2 / (2 sigma_i^2))`` with ``sigma_i`` proportional to the local
    scale, the leave-one-out log density is formed from the same weights, and the
    balancing factor ``exp(-alpha/2 (ell_j - ell_i))`` is applied before
    normalisation. Setting ``alpha = 0`` reduces the method to a self-tuned
    spectral embedding, which is exactly the control the density-balancing axis
    needs.

    Examples
    --------
    >>> from densforge.manifold.flagship import ManifoldFuse
    >>> from densforge.data.synth import make_swiss_roll
    >>> X = make_swiss_roll(300, 0)
    >>> Y = ManifoldFuse(n_neighbors=12).fit_transform(X)
    >>> Y.shape
    (300, 2)
    """

    def __init__(
        self,
        *,
        n_components: int = 2,
        n_neighbors: int = 15,
        alpha: float = 0.5,
        c_logp_clip: float = 4.0,
        random_state: int = 0,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.alpha = float(alpha)
        self.c_logp_clip = float(c_logp_clip)
        self.random_state = int(random_state)
        self._state: dict[str, object] | None = None
        self.fitted_on: FloatArray | None = None

    @classmethod
    def available(cls) -> bool:
        """Whether the backend is importable. Predicate only, no construction."""
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import dijkstra
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            return False
        # Reference each binding, so a linter cannot delete an import that exists
        # only to make this probe meaningful.
        return all(obj is not None for obj in (csr_matrix, dijkstra, NearestNeighbors))

    def fit(self, X: FloatArray) -> Self:
        """Fit on training data only.

        Raises
        ------
        BackendUnavailableError
            If SciPy or scikit-learn is missing.
        ShapeMismatchError
            If ``X`` is not a finite 2-D array with at least three rows.
        """
        if not self.available():
            raise BackendUnavailableError("ManifoldFuse requires scipy and scikit-learn")
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ShapeMismatchError(f"X must be 2-D (n, d), got shape {X.shape}")
        if X.shape[0] < 3:
            raise ShapeMismatchError(f"X needs at least 3 rows, got {X.shape[0]}")
        if not np.isfinite(X).all():
            raise ShapeMismatchError("X contains NaN or Inf")

        n, d = X.shape
        k = int(min(self.n_neighbors, n - 1))
        nn = NearestNeighbors(n_neighbors=k, n_jobs=1).fit(X)
        dist, idx = nn.kneighbors()  # self excluded
        dist = np.maximum(dist, EPS)
        rho = dist[:, k - 1]
        rho1 = dist[:, 0]
        rho_tilde = float(np.median(rho))

        log_s0_sq = float(np.mean(np.log(rho1**2 + 1e-300)) + np.log(n) - np.log(d + 2.0))
        sigma0 = float(np.sqrt(np.exp(np.clip(log_s0_sq, -60.0, 60.0))))
        sigma = np.maximum(sigma0 * (rho / rho_tilde) ** (1.0 / (d + 4.0)), 1e-6 * rho_tilde)

        quadratic = (dist**2) / sigma[:, None] ** 2
        log_weights = -0.5 * quadratic
        directed = np.exp(log_weights)

        rows = np.repeat(np.arange(n), k)
        cols = np.ascontiguousarray(idx.ravel())
        W = np.zeros((n, n))
        W[rows, cols] = directed.ravel()
        W = 0.5 * (W + W.T)

        # Leave-one-out log density from the self-excluding neighbourhood.
        ell = np.log(directed.sum(axis=1) + TINY) - d * np.log(sigma)
        ell = centre_and_clip_ell(ell, self.c_logp_clip)

        W_balanced = density_balance(directed, rows, cols, ell, self.alpha)
        embedding, eigenvalues, degrees = diffusion_coordinates(
            W_balanced, self.n_components, drop_trivial=True
        )

        self.fitted_on = X
        self._n_features = int(d)
        self._state = {
            "X": X,
            "idx": idx,
            "sigma": sigma,
            "ell": ell,
            "W": W,
            "W_balanced": W_balanced,
            "embedding": embedding,
            "eigenvalues": eigenvalues,
            "degrees": degrees,
            "fingerprint": array_fingerprint(X),
        }
        return self

    def _require_state(self) -> dict[str, object]:
        if self._state is None:
            raise FitFailedError("ManifoldFuse is not fitted; call fit() first")
        return self._state

    def transform(self, X: FloatArray | None = None) -> FloatArray:
        """Return embedding coordinates, shape ``(n, n_components)``.

        Only the training embedding is available. Producing coordinates for unseen
        points would require an out-of-sample extension, and every such scheme
        introduces its own choices; rather than offer one silently, this method
        requires that the caller pass the training data (or nothing at all). Use
        :class:`~densforge.density.flagship.DensFuse` when out-of-sample
        coordinates are needed.
        """
        state = self._require_state()
        if X is None:
            return state["embedding"]  # type: ignore[return-value]
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        train = state["X"]
        assert isinstance(train, np.ndarray)
        if X.shape != train.shape:
            raise ShapeMismatchError(
                "ManifoldFuse.transform only returns the training embedding; pass the "
                f"training data itself (shape {train.shape}), got {X.shape}"
            )
        if not np.array_equal(X, train):
            raise ShapeMismatchError(
                "ManifoldFuse.transform only returns the training embedding; the supplied "
                "array differs from the fitted data"
            )
        return state["embedding"]  # type: ignore[return-value]

    def fit_transform(self, X: FloatArray) -> FloatArray:
        """Fit and return the training embedding."""
        return self.fit(X).transform()

    def score_samples(self, X: FloatArray) -> FloatArray:
        """Return the leave-one-out log density used for balancing.

        Provided so the object satisfies the density half of the
        :class:`~densforge.core.interfaces.FusedEstimator` shape. It is the
        *leave-one-out* density of the self-tuning kernel, not a full density
        model: unlike the flagship's, this quantity is not normalised, so it must
        not be used as ``log p`` for cross-method NLL comparison. It exists to
        expose the balancing channel for inspection.
        """
        state = self._require_state()
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.shape[1] != self._n_features:
            raise ShapeMismatchError(
                f"X has {X.shape[1]} features but the model was fitted on {self._n_features}"
            )
        train = state["X"]
        sigma = state["sigma"]
        assert isinstance(train, np.ndarray) and isinstance(sigma, np.ndarray)
        from scipy.spatial.distance import cdist

        sq = cdist(X, train, metric="sqeuclidean")
        return -0.5 * np.log(np.exp(-0.5 * sq / (sigma[None, :] ** 2)).sum(axis=1) + TINY)

    def params(self) -> dict[str, object]:
        return {
            "n_components": self.n_components,
            "n_neighbors": self.n_neighbors,
            "alpha": self.alpha,
            "c_logp_clip": self.c_logp_clip,
            "random_state": self.random_state,
        }

    @property
    def fitted_fingerprints(self) -> frozenset[str]:
        from ..core.types import array_fingerprint

        if self.fitted_on is None:
            return frozenset()
        return frozenset({array_fingerprint(self.fitted_on)})


def _assert_embedder(obj: object) -> bool:
    """Structural check that ``obj`` satisfies :class:`ManifoldEmbedder`."""
    return isinstance(obj, ManifoldEmbedder)


__all__ = ["ManifoldFuse"]
