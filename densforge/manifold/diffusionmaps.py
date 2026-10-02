"""Diffusion maps: density-balanced operator and biorthogonal coordinates.

Author: 晨星 <CJX0712@users.noreply.github.com>

Pipeline
--------
1. **Density balancing.** A symmetric affinity ``W`` built from a non-uniform
   sample approximates a ``p``-weighted operator, so its leading eigenvectors
   describe the *sampling density* as much as the geometry. Multiplying edge
   ``(i, j)`` by ``exp(-alpha/2 (ell_j - ell_i))`` and re-symmetrising gives

       Wbar_ij = W_ij * cosh(alpha/2 * log(p_j / p_i)) >= W_ij

   because ``cosh >= 1``. Edge weights therefore never systematically shrink, and
   the correction vanishes quadratically as ``alpha -> 0``. The factor is ``alpha/2``
   rather than ``alpha`` precisely because of that identity: it is a design
   argument, not a convention.
2. **Symmetric normalisation.** ``S = D^-1/2 W D^-1/2``.
3. **Biorthogonal coordinates.** ``Psi = psi_m / sqrt(lambda_m)``, then
   ``Psi <- Psi G^-1/2`` with ``G = Psi^T D Psi``, so that ``Psi^T D Psi = I``
   (invariant I17). Without this step diffusion coordinates are not
   distance-interpretable.

Note on the Markov bound
------------------------
Rows of ``S`` do **not** sum to at most 1; measured max row sum is 1.156. What
holds is ``max |S_ij| <= 1`` and ``eig(S) in [-1, 1]``. The row-stochastic object is
``P = D^-1 W``. See ``docs/math_verification.md`` finding D5.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from ..core.errors import NumericalError
from ..core.types import FloatArray

#: Guard for the log-determinant / degree floor.
TINY = 1e-300

#: Floor on diffusion eigenvalues. ``1 - lambda -> 0`` makes ``psi / sqrt(lambda)``
#: diverge, which is the numerical face of the alpha-explosion failure mode.
LAMBDA_FLOOR = 1e-12


def symmetric_normalize(W: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return ``(S, deg)`` with ``S = D^-1/2 W D^-1/2`` and ``deg = diag(D)``.

    Parameters
    ----------
    W:
        Symmetric non-negative ``(N, N)`` affinity matrix.

    Returns
    -------
    S:
        Symmetrically normalised operator, exactly symmetric.
    deg:
        Row sums of ``W``, floored at :data:`TINY`.

    Raises
    ------
    NumericalError
        If ``W`` is not square or contains negative entries.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise NumericalError(f"affinity matrix must be square, got shape {W.shape}")
    if W.min() < -1e-12:
        raise NumericalError(f"affinity matrix has negative entries (min={W.min():.3e})")
    deg = np.maximum(W.sum(axis=1), TINY)
    inv_sqrt = 1.0 / np.sqrt(deg)
    S = (W * inv_sqrt[:, None]) * inv_sqrt[None, :]
    S = 0.5 * (S + S.T)  # kill the last-bit asymmetry from the two scalings
    return S, deg


def diffusion_coordinates(
    W: FloatArray,
    n_components: int = 2,
    *,
    drop_trivial: bool = True,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute biorthogonal diffusion coordinates.

    Parameters
    ----------
    W:
        Symmetric non-negative affinity matrix.
    n_components:
        Number of coordinates to return.
    drop_trivial:
        Skip the leading eigenvector. For a connected affinity matrix that
        eigenvector is the all-ones direction of the random walk: it carries no
        geometry, so keeping it would spend one of ``n_components`` on a constant
        column.

    Returns
    -------
    Psi:
        ``(N, n_components)`` coordinates satisfying ``Psi^T D Psi = I``.
    eigenvalues:
        The selected diffusion eigenvalues, in descending order.
    deg:
        The degrees used for the normalisation.
    """
    S, deg = symmetric_normalize(W)
    # Full symmetric eigendecomposition: this is an N x N dense problem and the
    # operator is dense, so there is no sparse variant to exploit here. The
    # partial-decomposition optimisation (docs, finding R4) applies to the
    # Tier-1 self-implemented solvers, which factorise a structured matrix.
    values, vectors = eigh(S)
    # `values` is ascending; take the largest, with a stable tiebreak so that
    # degenerate eigenvalues always resolve to the same index.
    order = np.argsort(-values, kind="stable")
    if drop_trivial:
        order = order[1:]
    selected = order[:n_components]
    lam = np.clip(values[selected], LAMBDA_FLOOR, None)
    Psi = vectors[:, selected] / np.sqrt(lam)[None, :]

    # Biorthogonalisation: Psi <- Psi G^-1/2, computed as Psi L^-T for the
    # Cholesky factor L of G (G = L L^T). Solving with the Cholesky factor is
    # both faster and better conditioned than forming G^-1/2 explicitly.
    G = Psi.T @ (Psi * deg[:, None])
    G = 0.5 * (G + G.T) + 1e-10 * np.eye(Psi.shape[1])
    try:
        L = np.linalg.cholesky(G)
    except np.linalg.LinAlgError:
        # G is numerically indefinite only when Psi is rank deficient, which
        # happens when fewer independent components exist than requested.
        # Fall back to an eigenvalue-clipped inverse square root.
        G = 0.5 * (G + G.T)
        gvals, gvecs = eigh(G)
        gvals = np.clip(gvals, 1e-10, None)
        Psi = (gvecs * (gvals**-0.5)[None, :]) @ (gvecs.T @ Psi)
    else:
        Psi = np.linalg.solve(L, Psi.T).T
    return Psi, lam, deg


def density_balance(
    W_dir: FloatArray,
    rows: np.ndarray,
    cols: np.ndarray,
    ell: FloatArray,
    alpha: float,
) -> FloatArray:
    """Apply density balancing to a directed sparse weight set.

    Parameters
    ----------
    W_dir:
        ``(N, k)`` directed edge weights; entry ``[i, a]`` is the weight from
        ``rows[i]`` to ``cols[i]``.
    rows, cols:
        Flat index arrays of the directed edges, each of length ``N * k``.
    ell:
        ``(N,)`` clipped leave-one-out log densities.
    alpha:
        Balancing strength in ``[0, 1]``.

    Returns
    -------
    FloatArray
        The symmetrically balanced ``(N, N)`` matrix.

    Notes
    -----
    Symmetrisation happens **once**, on the directed weights. Symmetrising first
    and then re-weighting the already-symmetrised entries halves every
    one-directional k-NN edge a second time; see ``docs/math_verification.md``
    finding D3.
    """
    if not 0.0 <= alpha <= 1.0:
        raise NumericalError(f"alpha must lie in [0, 1], got {alpha}")
    N = int(ell.shape[0])
    correction = np.exp(-0.5 * alpha * (ell[cols] - ell[rows]))
    Wb = np.zeros((N, N), dtype=np.float64)
    Wb[rows, cols] = W_dir.ravel() * correction
    Wb = 0.5 * (Wb + Wb.T)
    return Wb


def loo_log_density(log_weights: FloatArray, sig: FloatArray, d: int) -> FloatArray:
    """Leave-one-out log density from a self-excluding neighbourhood.

    Parameters
    ----------
    log_weights:
        ``(N, k)`` log kernel weights for each point's ``k`` nearest neighbours.
        The neighbourhood must already exclude the point itself, which
        ``NearestNeighbors.kneighbors()`` does when called without ``X``.
    sig:
        ``(N,)`` local scales.
    d:
        Ambient dimensionality.

    Returns
    -------
    FloatArray
        ``(N,)`` raw (unclipped, uncentred) leave-one-out log densities.

    Notes
    -----
    Using the self-inclusive density here would give every point maximal weight
    from its own kernel, artificially suppressing every point's importance. The
    self-excluding neighbourhood is the structural guarantee; this function only
    aggregates it.
    """
    summed = np.log(np.exp(log_weights).sum(axis=1) + TINY)
    return summed - d * np.log(sig)


def centre_and_clip_ell(ell: FloatArray, c_logp_clip: float = 4.0) -> FloatArray:
    """Shift log densities to have zero log-mean, then clip.

    The shift makes ``ell`` a log *ratio*, which is the only thing the balancing
    factor consumes. The clip bounds the density ratio to
    ``[e^-2c, e^2c]``; at ``c = 4`` that is ``[3.4e-4, 2981]``. This is the single
    most important guard against alpha-driven spectral blow-up: without it a
    single very sparse point can capture the leading eigenvector.
    """
    from scipy.special import logsumexp

    n = ell.shape[0]
    centred = ell - (logsumexp(ell) - np.log(n))
    return np.clip(centred, -c_logp_clip, c_logp_clip)


__all__ = [
    "LAMBDA_FLOOR",
    "TINY",
    "centre_and_clip_ell",
    "density_balance",
    "diffusion_coordinates",
    "loo_log_density",
    "symmetric_normalize",
]
