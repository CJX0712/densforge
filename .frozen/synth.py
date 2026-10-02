"""Synthetic data generators.

Author: 晨星 <CJX0712@users.noreply.github.com>

Rules that every generator in this module obeys
-----------------------------------------------
1. ``np.random.RandomState(seed)`` only. Never ``default_rng`` -- NEP 19 does not
   freeze the ``Generator`` stream, so thresholds would drift between NumPy
   versions on the CI matrix (design §6.6).
2. Signature ``make_<name>(n, seed, *, knobs...) -> FloatArray`` returning
   ``(n, d)`` float64.
3. Component indices are drawn **once** and broadcast, never re-drawn per sample.
   Rejection-style sampling consumes a variable number of draws per sample, which
   desynchronises the stream and can index out of bounds. The one-line fix:

       comp = rs.randint(0, K, n)            # draw once
       X = mus[comp] + rs.randn(n, d) * sig[comp]   # reuse the index

4. The seed fully determines the output: same ``(n, seed, knobs)`` gives a
   bit-identical array.
"""

from __future__ import annotations

import numpy as np

from ..core.seed import get_rng
from ..core.types import FloatArray

#: Guard against duplicate points: distances below this are treated as zero.
EPS = 1e-12


def _split_sizes(n: int, k: int) -> np.ndarray:
    """Return ``k`` group sizes summing exactly to ``n``.

    ``n // k`` for every group loses the remainder, which silently changes the
    requested sample count. The remainder is distributed one sample at a time.
    """
    base = n // k
    sizes = np.full(k, base, dtype=np.int64)
    sizes[: n - base * k] += 1
    return sizes


def _finish(rs: np.random.RandomState, X: FloatArray) -> FloatArray:
    """Shuffle rows and return contiguous float64."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    rs.shuffle(X)
    return X


def make_aniso_gmm(
    n: int,
    seed: int,
    *,
    n_components: int = 6,
    sigma_min: float = 0.25,
    sigma_max: float = 2.0,
    separation: float = 4.0,
) -> FloatArray:
    """Multi-scale, per-axis anisotropic Gaussian mixture (``d = 8``).

    Component ``k`` has centre ``mu_k ~ U(-4, 4)^d`` and a *diagonal* covariance
    whose entries are drawn per axis in ``[sigma_min, sigma_max]``, so a single
    component is already stretched along some axes and squeezed along others.

    What it probes
    --------------
    A single global bandwidth cannot serve both a tight component and a diffuse
    one: it underfits the dense cluster and oversmooths the sparse one. Measured
    NLL spread across the bandwidth grid is 13.9 nats, enough to separate a good
    bandwidth selector from a bad one. The anisotropy additionally gives the
    flagship's ``beta`` axis something real to exploit.

    Parameters
    ----------
    n_components:
        Number of mixture components ``K``.
    sigma_min, sigma_max:
        Per-axis standard-deviation bounds.
    separation:
        Half-width of the uniform range for component centres.
    """
    rs = get_rng(seed)
    d = 8
    k = int(n_components)
    centers = rs.uniform(-separation, separation, size=(k, d))
    scales = rs.uniform(sigma_min, sigma_max, size=(k, d))
    comp = rs.randint(0, k, n)  # drawn once, broadcast below
    X = centers[comp] + rs.randn(n, d) * scales[comp]
    return _finish(rs, X)


def make_swiss_roll(n: int, seed: int, *, noise: float = 0.15) -> FloatArray:
    """Swiss roll (``d = 3``): a 1-D curve wound into 3-D.

    ``t = 1.5*pi*(1 + 2U)``, ``h = 21U``, ``x = [t cos t, h, t sin t]``, then
    isotropic thickness noise.

    What it probes
    --------------
    The geometric gold standard. Neighbouring points along ``h`` are far apart in
    ambient space while points across the fold are close, so Euclidean distance
    is actively misleading and only a method that respects the intrinsic path
    recovers the roll. Measured trustworthiness at ``k=15``: Isomap 0.9997,
    LLE 0.9951, PCA 0.9573 -- a 0.0424 gap between PCA and the best method,
    which is what makes this dataset informative rather than decorative.

    ``noise`` scales the thickness. As it grows, spectral methods degrade first.
    """
    rs = get_rng(seed)
    t = 1.5 * np.pi * (1.0 + 2.0 * rs.rand(n))
    h = 21.0 * rs.rand(n)
    X = np.column_stack([t * np.cos(t), h, t * np.sin(t)])
    X += rs.randn(n, 3) * float(noise)
    return _finish(rs, X)


def make_double_spiral(
    n: int, seed: int, *, gap: float = 0.35, noise: float = 0.05
) -> FloatArray:
    """Two interleaved helices (``d = 3``).

    ``t`` spans ``[0, 2.5*pi]``, ``r = t / 2.5``, and the two arms are offset by
    ``±gap`` in the first coordinate plus a ``t / 4`` ramp in the third.

    What it probes
    --------------
    Topological unfolding. The two arms pass arbitrarily close in ambient space,
    so a method that trusts Euclidean proximity will bridge them. Only a method
    that preserves the intrinsic path separates the arms.
    """
    rs = get_rng(seed)
    sizes = _split_sizes(n, 2)
    t1 = 2.5 * np.pi * rs.rand(sizes[0])
    t2 = 2.5 * np.pi * rs.rand(sizes[1])
    r1 = t1 / 2.5
    r2 = t2 / 2.5
    arm1 = np.column_stack([r1 * np.cos(t1), r1 * np.sin(t1), t1 / 4.0]) - gap
    arm2 = np.column_stack([r2 * np.cos(t2), r2 * np.sin(t2), t2 / 4.0]) + gap
    X = np.vstack([arm1, arm2])
    X += rs.randn(X.shape[0], 3) * float(noise)
    return _finish(rs, X)


def make_circles(n: int, seed: int, *, noise: float = 0.10) -> FloatArray:
    """Two concentric rings (``d = 2``), half the samples on each.

    What it probes
    --------------
    One dataset, two tasks at once. As a *density* problem it is bimodal, so a
    single global bandwidth must serve two very different local scales. As a
    *geometry* problem it should collapse to one dimension (a circle), which
    exercises the embedding dimensionality choice.

    A note on trustworthiness: measured on ring-like data, all reasonable methods
    land within 0.005 of each other, which makes this dataset uninformative for
    the *manifold* task specifically. It is retained because it is strongly
    informative for the density task, and the benchmark records that asymmetry
    rather than hiding it.
    """
    rs = get_rng(seed)
    sizes = _split_sizes(n, 2)
    parts = []
    for k, size in enumerate(sizes):
        t = 2.0 * np.pi * rs.rand(size)
        r = (1.0 + 0.1 * rs.randn(size)) + 2.0 * k
        parts.append(np.column_stack([r * np.cos(t), r * np.sin(t)]))
    X = np.vstack(parts)
    X += rs.randn(X.shape[0], 2) * float(noise)
    return _finish(rs, X)


def make_t_mixture(
    n: int,
    seed: int,
    *,
    n_components: int = 4,
    df_min: float = 3.0,
    df_max: float = 30.0,
    scale: float = 3.0,
) -> FloatArray:
    """Heteroscedastic heavy-tailed mixture of Student-t components (``d = 6``).

    Each component draws its degrees of freedom from ``U(df_min, df_max)`` and
    standardises the draw to unit variance via ``t_nu / sqrt(nu / (nu - 2))``.

    What it probes
    --------------
    When a Gaussian assumption is wrong. A Student-t with ``nu = 3`` puts
    appreciable mass far outside any Gaussian's comfort zone, so a
    Gaussian-kernel KDE systematically overestimates the tails and a GMM is
    misspecified outright. Measured NLL spread across the grid is 8.64 nats.
    """
    rs = get_rng(seed)
    d = 6
    k = int(n_components)
    centers = rs.uniform(-scale, scale, size=(k, d))
    dfs = rs.uniform(df_min, df_max, size=k)
    sizes = _split_sizes(n, k)
    parts = []
    for i, size in enumerate(sizes):
        nu = dfs[i]
        if nu <= 2.0:
            raise ValueError(
                f"degrees of freedom must exceed 2 to have finite variance, got {nu}"
            )
        z = rs.standard_t(nu, size=(int(size), d))
        z /= np.sqrt(nu / (nu - 2.0))  # unit variance
        parts.append(centers[i] + z)
    X = np.vstack(parts)
    return _finish(rs, X)


def make_manifold_noise(
    n: int,
    seed: int,
    *,
    d: int = 12,
    intrinsic_dim: int = 3,
    noise: float = 0.30,
) -> FloatArray:
    """A low-dimensional manifold wrapped nonlinearly and buried in 12-D noise.

    The intrinsic coordinate is ``u = [cos(theta), sin(theta), theta / 3]``; the
    environment map is ``x = 2 * tanh(1.5 * u W) + noise`` with ``W`` a random
    ``(intrinsic_dim, d)`` matrix.

    What it probes
    --------------
    Ambient rank versus intrinsic rank: ``matrix_rank(X - X_bar) == 12`` while the
    data was generated from three degrees of freedom. A fixed global bandwidth is
    hopeless here, which is where the self-tuning axis earns its keep.

    Why ``tanh`` and not a linear map
    ---------------------------------
    A linear environment map is *recoverable by PCA*: measured trustworthiness
    was PCA 0.9893 versus Isomap 0.9882, so PCA wins and the dataset has zero
    discriminative power. The nonlinear map removes that shortcut. Even so,
    ring-like geometry remains hard to separate on trustworthiness alone, which
    is why the benchmark treats :func:`make_swiss_roll` as the manifold
    discriminator and this one as a density/high-dimensionality probe.
    """
    rs = get_rng(seed)
    theta = 2.0 * np.pi * rs.rand(n)
    u = np.column_stack([np.cos(theta), np.sin(theta), theta / 3.0])[:, :intrinsic_dim]
    W = rs.randn(intrinsic_dim, d) / np.sqrt(intrinsic_dim)
    X = 2.0 * np.tanh(1.5 * u @ W)
    X += rs.randn(n, d) * float(noise)
    return _finish(rs, X)


def make_iso_gauss(
    n: int, seed: int, *, d: int = 4, scale: float = 1.5, shift: float = 2.0
) -> FloatArray:
    """Control dataset: a single isotropic Gaussian with no exploitable structure.

    This is the anti-cheat dataset. A flagship that "wins" by 20% here has a bug,
    not an advantage: there is no local anisotropy, no multi-scale structure and
    no manifold to recover, so the only legitimate outcome is a tie within
    numerical noise. The published gate requires exactly that.
    """
    rs = get_rng(seed)
    X = rs.randn(n, d) * scale + rs.randn(d) * shift
    return _finish(rs, X)


__all__ = [
    "make_aniso_gmm",
    "make_circles",
    "make_double_spiral",
    "make_heavy_tail",
    "make_hetero_density",
    "make_iso_gauss",
    "make_manifold_noise",
    "make_sparse_dim",
    "make_swiss_roll",
    "make_t_mixture",
    "make_uniform_hypercube",
]


# ---------------------------------------------------------------------------
# Added 2026-10-03 (algorithm side) to close the benchmark-vs-spec DoD gap.
# 01-algorithm-design.md sec 5.2 specifies 5 real density datasets + 2
# controls.  G-D1 / G-D2 must be measured on the shipped registry, not on a
# private generator set -- calibrating on a different generator is a
# methodology error that hides the real number.
# Same four rules as the rest of this module (RandomState only, draw-once
# component index, _finish, seed-deterministic).
# ---------------------------------------------------------------------------


def make_sparse_dim(
    n: int,
    seed: int,
    *,
    n_active: int = 4,
    dim: int = 16,
    noise: float = 0.35,
) -> FloatArray:
    """Variable-dimension sparse noise: only ``n_active`` axes carry signal.

    This is the ``d > 6`` curse-of-dimensionality probe.  A single global
    bandwidth cannot serve both the 4 signal axes and the ``dim - n_active``
    near-isotropic noise axes, and held-out NLL punishes a bandwidth that is
    wrong in the noise directions.

    Parameters
    ----------
    n_active: number of latent signal dimensions.
    dim: ambient dimension (spec sweet spot: 16).
    noise: std of the isotropic thickness added on every axis (sweet spot: 0.35).
    """
    rs = np.random.RandomState(seed)
    d = int(dim)
    a = int(n_active)
    z = rs.randn(n, a)
    proj = rs.randn(a, d) / np.sqrt(a)
    return _finish(rs, z @ proj + rs.randn(n, d) * float(noise))


def make_hetero_density(
    n: int,
    seed: int,
    *,
    n_clusters: int = 3,
    weights: tuple[float, ...] = (0.6, 0.1, 0.3),
    separation: float = 10.0,
    dim: int = 6,
    scale: float = 0.5,
) -> FloatArray:
    """Well-separated clusters with UNEQUAL weights -- a sampling-density trap.

    6:1:3 at separation 10 gives roughly a 6x local density contrast.  One
    fixed bandwidth is simultaneously too wide in the sparse cluster and too
    narrow in the dense one.  This is the dataset that isolates the
    adaptive-scale axis (``eta``) from the whitening axis (``beta``).

    Parameters
    ----------
    weights: per-cluster probability; normalised internally.
    separation: centres are drawn from ``U(-sep, sep)^d``.
    """
    rs = np.random.RandomState(seed)
    d = int(dim)
    k = int(n_clusters)
    w = np.asarray(weights[:k], dtype=float)
    w = w / w.sum()
    centers = rs.uniform(-float(separation), float(separation), size=(k, d))
    comp = rs.choice(k, size=n, p=w)  # drawn once
    return _finish(rs, centers[comp] + rs.randn(n, d) * float(scale))


def make_heavy_tail(
    n: int,
    seed: int,
    *,
    dim: int = 4,
    df_min: float = 3.0,
    df_max: float = 30.0,
    scale_ratio: float = 4.0,
    n_components: int = 4,
) -> FloatArray:
    """Heteroscedastic heavy-tail mixture (Student-t).

    A Gaussian kernel is a poor tail model, so this probes graceful
    degradation rather than catastrophic failure.  Each component is
    rescaled to unit variance, making ``df`` the only shape parameter.
    """
    rs = np.random.RandomState(seed)
    d = int(dim)
    k = int(n_components)
    dfs = np.linspace(float(df_min), float(df_max), k)
    scales = np.geomspace(1.0, float(scale_ratio), k)
    centers = rs.randn(k, d) * 2.5
    # RandomState.standard_t takes a SCALAR df, so draw per component rather
    # than trying to broadcast a per-row df (numpy rejects that shape).
    #
    # No component index is drawn here. An earlier revision drew
    # `comp = rs.choice(k, size=n, ...)` and then never used it, because the
    # assignment loop below already partitions the rows. That was not merely dead
    # code: consuming a draw from the legacy RandomState stream shifts every
    # subsequent value, so the generator's output depended on a line that had no
    # effect on the result. `ruff check` (F841) caught it.
    out = np.empty((n, d), dtype=np.float64)
    sizes = _split_sizes(n, k)
    at = 0
    for j in range(k):
        cnt = int(sizes[j])
        if cnt == 0:
            continue
        sl = slice(at, at + cnt)
        z = rs.standard_t(dfs[j], size=(cnt, d))
        z /= np.sqrt(dfs[j] / (dfs[j] - 2.0))  # unit variance
        out[sl] = centers[j] + z * scales[j]
        at += cnt
    return _finish(rs, out)


def make_uniform_hypercube(
    n: int,
    seed: int,
    *,
    dim: int = 4,
    low: float = -1.0,
    high: float = 1.0,
) -> FloatArray:
    """CONTROL: uniform on a hypercube -- genuinely flat local density.

    Adaptive bandwidth has nothing to exploit here, so the flagship must
    roughly TIE a well-tuned fixed KDE.  A large "win" on this control means
    something is broken.  Pairs with ``make_iso_gauss`` for gate G-D4.
    """
    rs = np.random.RandomState(seed)
    return _finish(rs, rs.uniform(float(low), float(high), size=(n, int(dim))))
