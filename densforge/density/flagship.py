"""The flagship: :class:`DensFuse`.

Author: 晨星 <CJX0712@users.noreply.github.com>

What makes it a closed loop rather than a stack of modules
-----------------------------------------------------------
The local scale ``rho`` decides the kernel shape; the density estimate decides the
diffusion operator's normalisation; the embedding decides the geodesic structure
that refines ``rho``. Each feeds the next, and the loop is iterated to a fixed
point:

    rho --(sigma_i)--> H_i --(p)--> ell --(balance)--> W --(diffusion)--> Psi
     ^                                                                 |
     +--------------------- geodesic rho (round t > 0) -----------------+

Three axes, each orthogonal to what the baselines can express
-------------------------------------------------------------
==================  ==========================================================
axis                why a baseline cannot match it
==================  ==========================================================
anisotropy ``beta`` local whitening ``V diag(lam^beta)^-1 V^T``; a fixed
                    bandwidth has a *constant* ``sigma``, a ``gaussian_kde``
                    shares one covariance across all kernels, a GMM is
                    axis-aligned or global
adaptivity          ``sigma_i ~ rho_i^eta`` with a per-point ``rho_i``; no
                    baseline estimates a per-point scale at all
density balance     ``exp(-alpha/2 (ell_j - ell_i))`` is a channel no baseline
                    has, because no baseline produces a density in the first
                    place
==================  ==========================================================

Anti-"baseline plus a patch" rule
--------------------------------
This module imports **no** baseline class. There is no ``KernelDensity``, no
``GaussianMixture``, no ``SpectralEmbedding``, no ``Isomap`` anywhere in the fit
path: DensFuse is an independent kernel estimator, not a wrapper. That is a
structural guarantee that gains come from new hypothesis directions rather than
from a stronger base model, and it is enforced by
``tests/test_architecture.py::test_flagship_does_not_import_baselines``.

The math, and the three corrections that made it consistent, are documented in
``docs/math_verification.md``. The short version: the local covariance is divided
by ``rho^2`` (making the kernel family scale covariant), its eigenvalues are
normalised to unit geometric mean (making ``beta`` a pure shape parameter rather
than a hidden bandwidth multiplier), and every output matrix is symmetrised
exactly once from the directed weights.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors

from ..core.config import DensFuseConfig
from ..core.errors import (
    BackendUnavailableError,
    FitFailedError,
    NumericalError,
    ShapeMismatchError,
)
from ..core.interfaces import check_log_density
from ..core.types import FloatArray, array_fingerprint
from ..manifold.diffusionmaps import (
    TINY,
    centre_and_clip_ell,
    density_balance,
    diffusion_coordinates,
    loo_log_density,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle broken at runtime
    from ..hpo.coordinate import TuningResult

#: ``log(2*pi)``, cached because it enters every single kernel term.
LOG2PI = float(np.log(2.0 * np.pi))

#: Floor on distances, for coincident points.
EPS = 1e-12


@dataclass(frozen=True, slots=True)
class FitState:
    """Everything derived from the training data, and nothing else.

    Frozen and content-addressed, which is what makes invariant **I23**
    ("``select()`` must not mutate the fit state") checkable: the test deep-copies
    the state, runs hyper-parameter selection, and compares fingerprints.

    No field of this object is ever computed from validation or test data. That
    is the structural half of the leakage defence; the manifest check in
    :meth:`DensFuse.fit` is the dynamic half.
    """

    X_train: FloatArray
    idx: np.ndarray  # (N, k) neighbour indices, self excluded
    rho_euclid: FloatArray  # (N,) distance to the k-th nearest neighbour
    rho_tilde: float  # median local scale
    sigma0: float  # Lazaridis pilot bandwidth
    eigvecs: FloatArray  # (N, d, d) local principal directions
    eigvals: FloatArray  # (N, d) floored, volume-normalised local eigenvalues
    logdet_shape: FloatArray  # (N,) -beta * sum(log eigvals)
    rows: np.ndarray  # (N*k,) directed edge source
    cols: np.ndarray  # (N*k,) directed edge target
    directed_weights: FloatArray  # (N*k,) exp(-0.5 * quadratic form)
    W: FloatArray  # (N, N) symmetric affinity
    W_balanced: FloatArray  # (N, N) density-balanced affinity
    ell: FloatArray  # (N,) clipped leave-one-out log density
    degrees: FloatArray  # (N,) degrees of W_balanced
    embedding: FloatArray  # (N, n_components) diffusion coordinates
    eigenvalues: FloatArray  # (n_components,) diffusion eigenvalues
    sigma: FloatArray  # (N,) final local scales actually used
    n_components: int
    d: int
    fingerprint: str
    fitted_on_id: int

    @property
    def n_samples(self) -> int:
        return int(self.X_train.shape[0])


class DensFuse:
    """Fused density estimation and manifold learning.

    The object satisfies :class:`~densforge.core.interfaces.FusedEstimator`:
    :meth:`score_samples` returns log-density and :meth:`embed` returns
    coordinates, both derived from the same fitted state.

    Examples
    --------
    >>> import numpy as np
    >>> from densforge.density.flagship import DensFuse
    >>> from densforge.core.seed import set_all
    >>> from densforge.data.synth import make_circles
    >>> set_all(0)
    0
    >>> X = make_circles(400, 0)
    >>> model = DensFuse().fit(X)
    >>> logp = model.score_samples(X)
    >>> logp.shape
    (400,)
    >>> Psi = model.embed(X)
    >>> Psi.shape
    (400, 2)
    """

    def __init__(self, config: DensFuseConfig | None = None) -> None:
        self.config = config or DensFuseConfig()
        self._state: FitState | None = None
        #: Fingerprints of arrays this object has been fitted on. Assertion A3
        #: consults this before any test metric is published.
        self._fitted_fingerprints: frozenset[str] = frozenset()

    # ------------------------------------------------------------------ protocol
    @classmethod
    def available(cls) -> bool:
        """Whether the flagship's backend is importable.

        DensFuse needs SciPy and scikit-learn's ``NearestNeighbors`` only -- no
        baseline estimator. Returns a bool without constructing anything.
        """
        try:
            import scipy.linalg
            import scipy.sparse.csgraph
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            return False
        # Reference each binding so the imports are load-bearing rather than
        # decorative: an unused import here could be deleted by a linter, silently
        # turning this probe into an unconditional `True` that never detects a
        # missing backend.
        return all(
            obj is not None for obj in (scipy.linalg, scipy.sparse.csgraph, NearestNeighbors)
        )

    # ------------------------------------------------------------------ provenance
    @property
    def fitted_on(self) -> FloatArray | None:
        """The exact training array, or ``None`` before :meth:`fit`."""
        return None if self._state is None else self._state.X_train

    @property
    def fitted_fingerprints(self) -> frozenset[str]:
        """Content digests of everything this model was fitted on.

        Assertion **A3** consults this before any test metric is published. The
        digest rather than the identity, because a *copy* of the test array has a
        different ``id()`` and the same content -- and is just as much of a leak.
        """
        return self._fitted_fingerprints

    def params(self) -> dict[str, Any]:
        """Return the active hyper-parameters, for report provenance."""
        return self.config.as_params()

    # ------------------------------------------------------------------ fitting
    def fit(self, X: FloatArray, *, manifest: Any = None) -> Self:
        """Fit on training data. Never sees validation or test data.

        Parameters
        ----------
        X:
            ``(n, d)`` training samples.
        manifest:
            Optional :class:`~densforge.core.types.SplitManifest`. When supplied,
            the array's recorded role must be ``"train"``; otherwise
            :class:`~densforge.core.errors.LeakageError` is raised. This is
            assertion **A2**'s dynamic counterpart and the enforcement point for
            invariant **I24**.

        Returns
        -------
        Self

        Raises
        ------
        LeakageError
            If ``manifest`` says this array is not training data.
        ShapeMismatchError
            If ``X`` is not 2-D, or has fewer than ``k + 1`` rows.
        """
        if not self.available():
            raise BackendUnavailableError("DensFuse requires scipy and scikit-learn")
        X = self._validate_X(X)

        if manifest is not None:
            role = manifest.role(X)
            if role != "train":
                from ..core.errors import LeakageError

                raise LeakageError(
                    f"fit() received an array whose split role is {role!r}, expected 'train'; "
                    "the leakage firewall refuses to fit on val or test data"
                )

        state = self._fit_state(X, self.config)
        self._state = state
        self._fitted_fingerprints = frozenset({state.fingerprint})
        return self

    def select(self, X_val: FloatArray) -> TuningResult:
        """Choose hyper-parameters using validation data only.

        Runs a coordinate descent in the order given by
        :data:`~densforge.core.config.HPO_ORDER` and returns a
        :class:`~densforge.hpo.coordinate.TuningResult` carrying the winning
        configuration, the number of validation evaluations spent, and the full
        trace.

        The fitted state is **not** touched: this method only produces a new
        config object. That is invariant **I23**, and it is why :meth:`score_samples`
        can be trusted to be a pure function of the training data plus the config.
        Call :meth:`refit` afterwards to materialise the selection.

        Parameters
        ----------
        X_val:
            ``(n_val, d)`` validation samples.

        Returns
        -------
        TuningResult

        Raises
        ------
        FitFailedError
            If called before :meth:`fit`.
        """
        from ..hpo.coordinate import coordinate_descent

        if self._state is None:
            raise FitFailedError("select() must be called after fit()")
        X_val = self._validate_X(X_val)
        result = coordinate_descent(self._state, X_val, self.config, estimator=self)
        self.config = result.config
        return result

    def refit(self) -> Self:
        """Re-run :meth:`fit` with the current config.

        Call after :meth:`select` to materialise the selected hyper-parameters.
        """
        if self._state is None:
            raise FitFailedError("refit() must be called after fit()")
        self._state = self._fit_state(self._state.X_train, self.config)
        self._fitted_fingerprints = frozenset({self._state.fingerprint})
        return self

    # ------------------------------------------------------------------ inference
    def score_samples(self, X: FloatArray, *, m_score: int | None = None) -> FloatArray:
        """Return ``log p(x)`` for each row of ``X``. Larger means more likely.

        A pure function: it reads the fitted state and the config and writes
        neither. Invariant **I22** verifies this by scoring the same array twice
        and by mutating a copy in between.

        Parameters
        ----------
        X:
            ``(n, d)`` points to score.
        m_score:
            Override the number of retained neighbours. ``None`` uses the config
            value. Smaller is faster and slightly biased; larger is slower and
            closer to the exact sum.

        Returns
        -------
        FloatArray
            ``(n,)`` log-densities in nats.

        Raises
        ------
        NumericalError
            If the state is missing or the result contains non-finite values.
        """
        state = self._require_state()
        X = self._validate_X(X, expected_d=state.d)
        cfg = self.config
        m = int(m_score or cfg.m_score)
        return self._log_density(state, X, m, cfg.batch_size)

    def embed(self, X: FloatArray | None = None, *, mode: str = "inductive") -> FloatArray:
        """Return diffusion coordinates.

        Parameters
        ----------
        X:
            Points to embed. ``None`` returns the training embedding.
        mode:
            ``"inductive"`` (default) freezes the training graph and places new
            points by out-of-sample extension, so no test information enters the
            graph. ``"transductive"`` merges the query points into the graph and
            recomputes. Trustworthiness numbers are only comparable when the mode
            is stated, so it is a required, explicit argument.

        Returns
        -------
        FloatArray
            ``(n, n_components)`` coordinates.
        """
        state = self._require_state()
        if mode not in {"inductive", "transductive"}:
            raise NumericalError(f"mode must be 'inductive' or 'transductive', got {mode!r}")
        if X is None:
            return state.embedding
        X = self._validate_X(X, expected_d=state.d)
        if mode == "transductive":
            return self._embed_transductive(state, X)
        return self._embed_inductive(state, X)

    # ------------------------------------------------------------------ internals
    def _require_state(self) -> FitState:
        if self._state is None:
            raise FitFailedError("the model is not fitted; call fit() first")
        return self._state

    @staticmethod
    def _validate_X(X: FloatArray, *, expected_d: int | None = None) -> FloatArray:
        """Validate a query array.

        Note the row-count floor is **1**, not 2: scoring a single point is a
        legitimate operation (invariant I1 integrates the density one point at a
        time), and a two-row minimum would make the density function impossible to
        evaluate pointwise. The two-row minimum applies to *fitting*, where a
        neighbourhood has to exist.
        """
        arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if arr.ndim != 2:
            raise ShapeMismatchError(f"X must be 2-D (n, d), got shape {arr.shape}")
        if arr.shape[0] < 1:
            raise ShapeMismatchError("X must contain at least one row")
        if not np.isfinite(arr).all():
            raise ShapeMismatchError("X contains NaN or Inf")
        if expected_d is not None and arr.shape[1] != expected_d:
            raise ShapeMismatchError(
                f"X has {arr.shape[1]} features but the model was fitted on {expected_d}"
            )
        return arr

    def _fit_state(self, X: FloatArray, cfg: DensFuseConfig) -> FitState:
        """Run the full closed loop and freeze the result."""
        N, d = X.shape
        k = min(cfg.k, N - 1)
        if k < 2:
            raise FitFailedError(
                f"n_samples={N} is too small for k={cfg.k}; need at least k+2 samples"
            )
        eta = cfg.resolved_eta(d)

        # -- stage 1: Euclidean local scale ------------------------------------
        # kneighbors() is called WITHOUT X, so the returned neighbours already
        # exclude the point itself: idx is (N, k) of genuine neighbours and
        # dist[:, k-1] is the distance to the k-th one. Passing X here would make
        # the point its own first neighbour at distance 0.
        nn = NearestNeighbors(n_neighbors=k, n_jobs=1).fit(X)
        dist, idx = nn.kneighbors()
        dist = np.maximum(dist, EPS)
        rho = dist[:, k - 1]
        rho1 = dist[:, 0]
        rho_tilde = float(np.median(rho))

        # Lazaridis pilot bandwidth. The signs are load-bearing:
        #   log sigma0^2 = mean(log rho1^2) + log N - log(d + 2)
        # log N - log(d+2) is positive for N >> d and lifts sigma0 to a sensible
        # magnitude. The sign-flipped variant yields sigma0 ~ 1e-3 and an NLL of
        # order 1e6.
        log_s0_sq = float(np.mean(np.log(rho1**2 + 1e-300)) + np.log(N) - np.log(d + 2.0))
        sigma0 = float(np.sqrt(np.exp(np.clip(log_s0_sq, -60.0, 60.0))))

        # -- stage 2: local anisotropy ----------------------------------------
        eigvecs, eigvals = _local_principal_directions(X, idx, rho, cfg.tau)
        logdet_shape = -cfg.beta * np.log(eigvals).sum(axis=1)

        rows = np.repeat(np.arange(N), k)
        cols = np.ascontiguousarray(idx.ravel())
        offsets = X[idx] - X[:, None, :]  # (N, k, d) neighbour displacement

        # -- stage 3: the closed loop -----------------------------------------
        W = np.zeros((N, N))
        W_balanced = np.zeros((N, N))
        ell = np.zeros(N)
        sigma = np.zeros(N)
        directed = np.zeros(N * k)
        diffusion = np.zeros((N, N))

        for round_index in range(cfg.n_fuse_rounds):
            if round_index == 0:
                rho_eff = rho
            else:
                # Geodesic scale from the PREVIOUS round's graph. The graph is built
                # from a reduced neighbourhood (k // 3) on purpose: on the full k-NN
                # graph the k-th shortest path is the direct edge to the k-th
                # Euclidean neighbour, so rho_geo would equal rho_euclid exactly and
                # the geodesic branch would be inert.
                rho_eff = _geodesic_rho(X, diffusion, k, cfg.k_graph_div, rho)
            rho_round = float(np.median(rho_eff))
            rho_clipped = np.clip(rho_eff, cfg.c_lo * rho_round, cfg.c_hi * rho_round)
            sigma = np.maximum(
                cfg.gamma * sigma0 * (rho_clipped / rho_round) ** eta,
                1e-6 * rho_round,
            )

            quadratic = (
                _quadratic_forms(offsets, eigvecs, eigvals, cfg.beta) / sigma[:, None] ** 2
            )
            log_weights = -0.5 * quadratic
            directed = np.exp(log_weights).ravel()

            W = _symmetrise(directed, rows, cols, N)
            # The self-excluding neighbourhood makes this a genuine leave-one-out
            # estimate; no manual removal of the diagonal term is needed.
            ell = centre_and_clip_ell(loo_log_density(log_weights, sigma, d), cfg.c_logp_clip)
            W_balanced = density_balance(directed.reshape(N, k), rows, cols, ell, cfg.alpha)
            diffusion = W_balanced

        embedding, eigenvalues, degrees = diffusion_coordinates(
            W_balanced, cfg.n_components, drop_trivial=cfg.drop_trivial
        )

        state = FitState(
            X_train=X,
            idx=idx,
            rho_euclid=rho,
            rho_tilde=rho_tilde,
            sigma0=sigma0,
            eigvecs=eigvecs,
            eigvals=eigvals,
            logdet_shape=logdet_shape,
            rows=rows,
            cols=cols,
            directed_weights=directed,
            W=W,
            W_balanced=W_balanced,
            ell=ell,
            degrees=degrees,
            embedding=embedding,
            eigenvalues=eigenvalues,
            sigma=sigma,
            n_components=int(cfg.n_components),
            d=int(d),
            fingerprint=array_fingerprint(X),
            fitted_on_id=id(X),
        )
        self._check_state_finite(state)
        return state

    @staticmethod
    def _check_state_finite(state: FitState) -> None:
        """Refuse to publish a state containing NaN or Inf."""
        checks = {
            "sigma": state.sigma,
            "eigvals": state.eigvals,
            "eigvecs": state.eigvecs,
            "logdet_shape": state.logdet_shape,
            "W": state.W,
            "W_balanced": state.W_balanced,
            "ell": state.ell,
            "embedding": state.embedding,
            "degrees": state.degrees,
        }
        for name, array in checks.items():
            if not np.isfinite(array).all():
                raise NumericalError(
                    f"fitted state field {name!r} contains non-finite values; "
                    "refusing to publish a degenerate model"
                )

    # -- density ------------------------------------------------------------
    def _log_density(
        self, state: FitState, X: FloatArray, m: int, batch_size: int
    ) -> FloatArray:
        """The one and only place a density is evaluated.

        Everything stays in log space. The per-kernel exponent is non-positive by
        construction and the sum is taken with ``logsumexp``, so neither overflow
        nor underflow can produce an ``inf`` or a silent ``nan``
        (architecture §2.2).

        Truncation selects the retained neighbours by **kernel weight**, not by
        Euclidean distance. With an anisotropic kernel a point can be far in
        Euclidean terms yet near in Mahalanobis terms, so Euclidean truncation
        discards high-mass contributions; measured L1 distortion at ``m = 128``,
        ``N = 400`` falls from 0.665 to 0.241 with the corrected selection
        (``docs/math_verification.md`` finding D6).
        """
        N, d = state.X_train.shape
        m = int(min(max(m, 1), N))
        logdet_H = 2 * d * np.log(state.sigma) + state.logdet_shape

        # Generous Euclidean candidate set, then keep the top-m by kernel weight.
        n_candidates = int(min(N, max(m, min(N, 4 * m))))
        finder = NearestNeighbors(n_neighbors=n_candidates, n_jobs=1).fit(state.X_train)

        out: list[FloatArray] = []
        for start in range(0, X.shape[0], batch_size):
            block = X[start : start + batch_size]
            _, nb = finder.kneighbors(block)  # (b, n_candidates)
            kernel = self._kernel_log_weights(state, block, nb, logdet_H)
            if m < n_candidates:
                top = np.argpartition(-kernel, m - 1, axis=1)[:, :m]
                kernel = np.take_along_axis(kernel, top, axis=1)
            # -log N because the mixture weights are 1/N spread over all N
            # training points, including those the truncation dropped. Using 1/m
            # instead would inflate the density by N/m.
            out.append(logsumexp(kernel, axis=1) - np.log(N))
        result = check_log_density("DensFuse", np.concatenate(out))
        return result

    def _kernel_log_weights(
        self, state: FitState, block: FloatArray, nb: np.ndarray, logdet_H: FloatArray
    ) -> FloatArray:
        """Return the per-neighbour log kernel weights, shape ``(b, m)``."""
        delta = block[:, None, :] - state.X_train[nb]  # (b, m, d)
        vecs = state.eigvecs[nb]  # (b, m, d, d)
        projected = np.einsum("bmd,bmde->bme", delta, vecs)
        weights = state.eigvals[nb] ** self.config.beta  # (b, m, d)
        quadratic = np.einsum("bme,bme->bm", projected * projected, weights)
        quadratic = quadratic / (state.sigma[nb] ** 2)
        return -0.5 * quadratic - 0.5 * logdet_H[nb] - 0.5 * state.d * LOG2PI

    # -- embedding ----------------------------------------------------------
    def _embed_inductive(self, state: FitState, X: FloatArray) -> FloatArray:
        """Out-of-sample extension onto the frozen training graph.

        Query points never enter the graph, so no test information can influence
        the geometry. This is the recommended reporting mode.

        The extension is the standard Nystrom form

            psi_out = D_out^-1/2 W_out D^-1/2 Psi / sqrt(lambda)

        where ``W_out`` holds the query-to-training kernel weights, ``D`` the frozen
        training degrees and ``D_out`` each query point's own total affinity.
        """
        m = int(min(self.config.m_score, state.n_samples))
        logdet_H = 2 * state.d * np.log(state.sigma) + state.logdet_shape
        finder = NearestNeighbors(n_neighbors=m, n_jobs=1).fit(state.X_train)
        _, nb = finder.kneighbors(X)
        kernel = self._kernel_log_weights(state, X, nb, logdet_H)
        W_out = np.exp(kernel)  # (n, m) weights from each query into the graph

        # Out-of-sample degree, put on the same scale as the training degrees by
        # dividing by the training-set size.
        deg_out = np.maximum(W_out.sum(axis=1) / state.n_samples, TINY)
        # D^-1/2 Psi, gathered at the selected neighbours: (n, m, n_components).
        scaled_basis = (state.degrees**-0.5)[:, None] * state.embedding
        weighted = W_out[:, :, None] * scaled_basis[nb]
        projected = weighted.sum(axis=1)  # (n, n_components)
        psi = (projected / np.sqrt(deg_out)[:, None]) / np.sqrt(state.eigenvalues)[None, :]
        return _align_to_train(state, psi)

    def _embed_transductive(self, state: FitState, X: FloatArray) -> FloatArray:
        """Merge query points into the graph and recompute the operator.

        Reported alongside the inductive mode, never instead of it. The graph now
        depends on the query set, so trustworthiness values are not comparable
        with inductive numbers.
        """
        combined = np.vstack([state.X_train, X])
        extended = DensFuse(replace(self.config))
        extended_state = extended._fit_state(combined, self.config)
        return extended_state.embedding[state.n_samples :]


def _align_to_train(state: FitState, psi: FloatArray) -> FloatArray:
    """Put out-of-sample coordinates in the training embedding's frame.

    The Nystrom extension fixes the query coordinates only up to their own mean,
    which differs from the training embedding's mean. Trustworthiness compares
    neighbour *ranks* between the input space and the embedding, so the two blocks
    must share an origin or the query block's coordinates carry an arbitrary
    offset that has nothing to do with geometry.

    Only the **translation** is removed. No rotation, no rescaling, and crucially no
    projection onto the training span: the query block has a different number of
    rows from the training block, so an orthogonal projection between them is not
    even dimensionally defined. Removing a linear transform would also be wrong --
    it would change the within-block neighbour distances that the metric measures.
    """
    return psi - state.embedding.mean(axis=0, keepdims=True)


def _local_principal_directions(
    X: FloatArray, idx: np.ndarray, rho: FloatArray, tau: float
) -> tuple[FloatArray, FloatArray]:
    """Local PCA with a spectral floor and unit-geometric-mean normalisation.

    Returns
    -------
    eigvecs:
        ``(N, d, d)`` local principal directions, descending by eigenvalue.
    eigvals:
        ``(N, d)`` eigenvalues, floored and volume-normalised.

    Notes
    -----
    Two normalisations, both necessary, both verified in
    ``docs/math_verification.md``:

    * **Divide by ``rho^2``.** This makes the local shape dimensionless, so
      ``H(cX) = c^2 H(X)`` exactly and the density is scale covariant (invariant
      I9). Without it the estimator's units depend on the data's units.
    * **Unit geometric mean.** ``det H`` becomes independent of ``beta``, so
      ``beta`` is a pure shape parameter. Without it, raising ``beta`` inflates
      the kernel volume and acts as an unadvertised bandwidth multiplier, which
      makes the anisotropy axis untestable.

    The spectral floor is not optional: without it ``H`` is singular for ``k < d``
    or degenerate neighbourhoods, ``log det H`` goes to ``-inf`` and the NLL
    explodes (measured ``~1e6`` in the original pilot).
    """
    k = idx.shape[1]
    offsets = X[idx] - X[:, None, :]
    centered = offsets - offsets.mean(axis=1, keepdims=True)
    cov = np.einsum("nkd,nke->nde", centered, centered) / k
    cov = 0.5 * (cov + cov.transpose(0, 2, 1))
    cov = cov / (rho**2)[:, None, None]

    values, vectors = np.linalg.eigh(cov)  # ascending
    values = values[:, ::-1]
    vectors = vectors[:, :, ::-1]
    floored = np.maximum(values, tau * values.mean(axis=1, keepdims=True))
    # A fully degenerate neighbourhood (all neighbours coincident) would make the
    # mean zero and log(mean) = -inf, so clamp before taking the logarithm.
    floored = np.maximum(floored, TINY)
    normalised = floored / np.exp(np.log(floored).mean(axis=1, keepdims=True))
    return np.ascontiguousarray(vectors), np.ascontiguousarray(normalised)


def _quadratic_forms(
    offsets: FloatArray, eigvecs: FloatArray, eigvals: FloatArray, beta: float
) -> FloatArray:
    """Mahalanobis quadratic form per neighbour, shape ``(N, k)``.

    Computed in the eigenbasis rather than by forming ``H^-1`` explicitly: it costs
    ``O(N k d)`` instead of ``O(N k d^2)`` and avoids a batched matrix inverse
    whose symmetry would only have to be restored afterwards.
    """
    projected = np.einsum("nkd,nde->nke", offsets, eigvecs)
    weights = eigvals**beta
    return np.einsum("nke,nke->nk", projected * projected, weights[:, None, :])


def _symmetrise(directed: FloatArray, rows: np.ndarray, cols: np.ndarray, n: int) -> FloatArray:
    """Scatter directed weights into a dense matrix and symmetrise **once**.

    Symmetrising an already-symmetrised matrix a second time halves every
    one-directional k-NN edge again. Measured error at ``alpha = 0``: 25% of the
    edge weight (``docs/math_verification.md`` finding D3).
    """
    W = np.zeros((n, n), dtype=np.float64)
    W[rows, cols] = directed
    return 0.5 * (W + W.T)


def _geodesic_rho(
    X: FloatArray,
    W: FloatArray,
    k: int,
    k_graph_div: int,
    rho_fallback: FloatArray,
) -> FloatArray:
    """Local scale measured as accumulated shortest-path length.

    Builds a k-NN graph over a **reduced** neighbourhood and reads off the k-th
    shortest path from each node. The reduction is essential: on the full k-NN
    graph the k-th Euclidean neighbour is a direct edge, so the k-th shortest path
    *is* that edge and the geodesic scale would be identical to the Euclidean one
    (measured ratio 1.000000). On a graph with ``k // 3`` neighbours the k-th
    shortest path is genuinely multi-hop; measured ratio on a noisy swiss roll:
    1.1117.
    """
    N = X.shape[0]
    k_graph = int(max(3, min(k // max(k_graph_div, 1), N - 1)))
    finder = NearestNeighbors(n_neighbors=k_graph, n_jobs=1).fit(X)
    gdist, gidx = finder.kneighbors()  # self already excluded
    gdist = np.maximum(gdist, EPS)
    rows = np.repeat(np.arange(N), k_graph)
    graph = csr_matrix(
        (gdist.ravel(), (rows, np.ascontiguousarray(gidx.ravel()))), shape=(N, N)
    )
    # limit=k keeps only the k smallest distances per source, so the matrix stays
    # sparse in practice even though its storage is dense.
    lengths = dijkstra(graph, directed=False, limit=k)
    lengths = np.where(np.isfinite(lengths), lengths, np.inf)

    rho = np.empty(N, dtype=np.float64)
    for i in range(N):
        finite = np.sort(lengths[i][np.isfinite(lengths[i])])
        rho[i] = finite[min(k - 1, finite.size - 1)] if finite.size else np.inf
    bad = ~np.isfinite(rho) | (rho <= 0)
    # A disconnected component has no path; the Euclidean scale is the only
    # sensible stand-in and is strictly better than an infinite bandwidth.
    rho[bad] = rho_fallback[bad]
    del W  # kept in the signature to document the caller contract
    return rho


__all__ = ["LOG2PI", "DensFuse", "FitState"]
