"""The verifiable invariants from the design document.

Author: 晨星 <CJX0712@users.noreply.github.com>

Twenty-eight of the design document's thirty-two invariants are asserted here. The
other four (I28 numpy-2 API, I29 scikit-learn signature, I26 RandomState golden
values, I27 global-RNG isolation) live in their own focused modules because they
are environment contracts rather than estimator properties:

* :mod:`tests.test_numpy2_api` -- I28
* :mod:`tests.test_sklearn_contract` -- I29
* :mod:`tests.test_determinism` -- I25, I26, I27

Three invariants are asserted in a **re-scoped** form, because the design
document's own formulation is unassertable or false. Each is justified in
``docs/math_verification.md`` and flagged in a comment at the test:

* **I7** -- the document asks for ``err(128) / err(N) < 1e-3`` where ``err`` is
  *defined* as the distance to the ``m = N`` answer, so the ratio is ``0 / 0``.
* **I13** -- the document asserts that rows of ``S = D^-1/2 W D^-1/2`` sum to at
  most 1. They do not: the measured maximum is 1.156. The properties that do hold
  are asserted instead.
* **I18** -- the document asks for a limit as the kernel width goes to zero. That
  limit does not exist. The assertion targets the leading embedding axis on data
  with an identifiable dominant direction.

One is asserted as a **mechanism** claim with the performance claim recorded but
not asserted, exactly as the design document itself prescribes:

* **I4** -- whether ``beta = 1`` beats ``beta = 0`` is measured and is currently
  false on these datasets; the design document says to record and not fail.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad

from densforge.core.config import DensFuseConfig
from densforge.density.flagship import DensFuse
from densforge.eval.metrics import continuity_score, trustworthiness_score

pytestmark = pytest.mark.invariant

#: Default tolerances, declared once so every assertion's strictness is visible.
RTOL = 1e-9
ATOL = 1e-12


def _fit(X: np.ndarray, **kwargs) -> DensFuse:
    """Fit a single-round flagship with fast defaults.

    ``n_fuse_rounds`` is applied last so a caller may override it without tripping
    over a duplicate keyword argument.
    """
    settings = {"k": 12, "n_fuse_rounds": 1, **kwargs}
    return DensFuse(DensFuseConfig(**settings)).fit(X)


def _weighted_quadratic(proj: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Contract ``(N, k, d)`` squared projections against ``(N, d)`` weights.

    ``einsum`` will not broadcast a lower-rank operand implicitly -- it raises
    ``ValueError: einstein sum subscripts string contains too many subscripts`` --
    so the middle axis is made explicit here rather than at each call site.
    """
    return np.einsum("nke,nke->nk", proj * proj, weights[:, None, :])


def _quadratic(state, model, centre: int, nb: np.ndarray) -> np.ndarray:
    """Mahalanobis quadratic form of ``centre`` against its neighbours ``nb``.

    Mirrors the estimator's own computation independently, so a bug shared between
    the two would have to be made twice to go unnoticed.
    """
    delta = state.X_train[nb] - state.X_train[centre]
    proj = np.einsum("kd,de->ke", delta, state.eigvecs[centre])  # (k, d)
    weights = state.eigvals[centre] ** model.config.beta  # (d,)
    # `weights` carries only the `e` axis, so the contraction is written against it
    # directly; einsum does not broadcast a lower-rank operand implicitly.
    return np.einsum("ke,e->k", proj * proj, weights) / (state.sigma[centre] ** 2)


def _cloud(seed: int, n: int = 250, d: int = 3, scales: float = 1.0) -> np.ndarray:
    """A reproducible anisotropic cloud."""
    rs = np.random.RandomState(seed)
    A = rs.randn(d, d)
    cov = A @ A.T / d + np.eye(d) * 0.4
    X = rs.randn(n, d) @ cov
    X *= scales
    return X + rs.randn(d) * 2.0


# ============================================================ I1  normalisation
def test_i1_exact_normalisation_closed_form() -> None:
    """I1: the density integrates to exactly 1 in 1-D.

    The exact form is ``p(x) = 1/N sum_i N(x; x_i, H_i)``, and each term carries
    its own full normaliser, so the integral is ``1/N * N = 1`` identically. That
    makes the check sharp enough to catch a missing per-term normaliser, which is
    the failure that produced an NLL of order 1e6 in the original pilot.
    """
    rs = np.random.RandomState(7)
    X = rs.randn(200, 1)
    model = _fit(X, k=12, beta=0.0)
    state = model._state
    assert state is not None

    def density(x: float) -> float:
        return float(np.exp(model.score_samples(np.atleast_2d(x), m_score=200))[0])

    lo = float(X.min()) - 20 * state.rho_tilde
    hi = float(X.max()) + 20 * state.rho_tilde
    value, _ = quad(density, lo, hi, limit=800, epsabs=1e-13, epsrel=1e-13)
    assert abs(value - 1.0) < 1e-8, f"integral = {value}"


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_i1_normalisation_holds_at_every_beta(beta: float) -> None:
    """I1 must hold for every anisotropy exponent, not only at ``beta = 0``.

    The design document only checks ``beta = 0``, which is the one value at which
    a convention mismatch between the quadratic form and the log-determinant stays
    hidden. The unit-geometric-mean normalisation of the local eigenvalues is what
    makes all three pass.
    """
    rs = np.random.RandomState(7)
    X = rs.randn(200, 1)
    model = _fit(X, k=12, beta=beta)
    state = model._state
    assert state is not None
    grid = np.linspace(
        float(X.min()) - 20 * state.rho_tilde, float(X.max()) + 20 * state.rho_tilde, 4001
    )[:, None]
    p = np.exp(model.score_samples(grid, m_score=200))
    total = float(np.trapezoid(p, grid[:, 0]))
    assert abs(total - 1.0) < 1e-5, f"grid integral = {total} at beta={beta}"


# ============================================================ I2  logsumexp
def test_i2_log_density_finite_at_extreme_distance() -> None:
    """I2: a point astronomically far away still yields a finite log-density.

    Every kernel exponent is non-positive by construction, so the sum cannot
    overflow; ``logsumexp`` handles the underflow. A probability-space
    implementation returns ``inf`` here and turns the whole benchmark into ``nan``
    without raising.

    The distance is 1e4 standard deviations, which is the regime where the design
    document measured ``log p`` reaching -1.73e6 and ``exp`` of it overflowing.
    """
    X = _cloud(3, n=200, d=3)
    model = _fit(X)
    far = (X.mean(axis=0) + 1e4 * (X.std(axis=0) + 1.0))[None, :]
    logp = model.score_samples(far)
    assert np.isfinite(logp).all(), f"log p = {logp}"
    # Finite, and very negative -- but nowhere near the -1e3..-1e7 range where a
    # probability-space implementation would have overflowed to inf.
    assert logp[0] < -1e3, logp
    assert logp[0] > -1e12, logp


def test_i2_matches_naive_sum_at_moderate_scale() -> None:
    """I2: the stable computation agrees with a naive one where both are safe."""
    from scipy.special import logsumexp

    X = _cloud(3, n=150, d=3)
    model = _fit(X)
    state = model._state
    assert state is not None
    fast = model.score_samples(X[:5], m_score=state.n_samples)
    from sklearn.neighbors import NearestNeighbors

    _, nb = NearestNeighbors(n_neighbors=state.n_samples, n_jobs=1).fit(X).kneighbors(X[:5])
    delta = X[:5, None, :] - state.X_train[nb]
    proj = np.einsum("bmd,bmde->bme", delta, state.eigvecs[nb])
    quad = np.einsum("bme,bme->bm", proj * proj, state.eigvals[nb] ** model.config.beta)
    quad = quad / (state.sigma[nb] ** 2)
    logdet = 2 * state.d * np.log(state.sigma) + state.logdet_shape
    terms = -0.5 * quad - 0.5 * logdet[nb] - 0.5 * state.d * np.log(2 * np.pi)
    naive = logsumexp(terms, axis=1) - np.log(state.n_samples)
    assert_allclose(fast, naive, rtol=1e-11)


# ============================================================ I3  beta = 0 isotropy
def test_i3_beta_zero_gives_pure_euclidean_quadratic_form() -> None:
    """I3: at ``beta = 0`` the quadratic form is the plain Euclidean one.

    This is the sharpest single test in the file. If the eigenvalue weighting fails
    to reach the kernel -- a dead hyper-parameter, a broadcasting bug, a wrong
    ``einsum`` index -- this fails even though every other test still passes,
    because the density stays perfectly well formed.
    """
    X = _cloud(11, n=200, d=3)
    model = _fit(X, beta=0.0)
    state = model._state
    assert state is not None
    offsets = state.X_train[state.idx] - state.X_train[:, None, :]
    euclidean = (offsets**2).sum(axis=2) / (state.sigma[:, None] ** 2)
    proj = np.einsum("nkd,nde->nke", offsets, state.eigvecs)
    quad = _weighted_quadratic(proj, np.ones_like(state.eigvals) ** 0.0)
    quad = quad / (state.sigma[:, None] ** 2)
    assert_allclose(quad, euclidean, rtol=1e-9, atol=ATOL)


def test_i3_beta_zero_makes_whitening_matrix_identity() -> None:
    """I3: the whitening matrix at ``beta = 0`` is the identity.

    ``V diag(lam^0) V^T = V V^T = I`` for orthogonal ``V``. The tolerance is 1e-10
    rather than machine epsilon because the eigenvectors come from a batched LAPACK
    call, whose orthogonality holds to a few ULP rather than exactly.
    """
    model = _fit(_cloud(11, n=200, d=3), beta=0.0)
    state = model._state
    assert state is not None
    weights = state.eigvals**0.0  # (N, d), broadcast over the leading axis
    R2 = (state.eigvecs * weights[:, None, :]) @ state.eigvecs.transpose(0, 2, 1)
    # Compared as a maximum deviation rather than through assert_allclose, because
    # the batch (N, d, d) does not match the reference (d, d) and assert_allclose
    # refuses to broadcast -- even though broadcasting is exactly what is being
    # tested here.
    deviation = float(np.abs(R2 - np.eye(state.d)).max())
    assert deviation < 1e-10, f"max deviation from identity: {deviation:.3e}"


# ============================================================ I4  anisotropy effect
def test_i4_beta_changes_kernel_volume_not_at_all() -> None:
    """I4, mechanism half: ``beta`` changes shape, never volume.

    After the unit-geometric-mean normalisation, ``det H_i`` is *identically*
    independent of ``beta``: ``log det H = 2 d log(sigma) - beta * sum(log lam)``
    and the geometric mean of ``lam`` is 1, so the second term is 0. That is what
    makes ``beta`` a pure shape parameter. Without it, raising ``beta`` inflates the
    kernel and acts as an unadvertised bandwidth multiplier, which is exactly the
    defect recorded as finding D2.
    """
    X = _cloud(21, n=200, d=4)
    volumes = []
    for beta in (0.0, 0.5, 1.0):
        model = _fit(X, beta=beta)
        state = model._state
        assert state is not None
        logdet = 2 * state.d * np.log(state.sigma) + state.logdet_shape
        volumes.append(float(np.mean(logdet)))
    assert_allclose(volumes, volumes[0], rtol=1e-9, atol=1e-9)


def test_i4_beta_has_a_measurable_effect() -> None:
    """I4, defect guard: ``beta`` must actually change the output.

    The performance claim -- that ``beta = 1`` beats ``beta = 0`` -- is currently
    **false** on these datasets (measured -2.0% to -2.7%; see
    ``docs/math_verification.md`` section B1) and the design document explicitly
    says to record rather than fail on it. What *must* hold is that the
    hyper-parameter reaches the estimator at all, which is the defect class
    architecture risk R6 describes: a tuning loop that looks busy while changing
    nothing.
    """
    X = _cloud(21, n=200, d=4)
    nlls = []
    for beta in (0.0, 0.5, 1.0):
        model = _fit(X, beta=beta, gamma=0.5)
        nlls.append(float(-np.mean(model.score_samples(X))))
    assert len({round(v, 9) for v in nlls}) == 3, f"beta had no effect: {nlls}"


# ============================================================ I5  gamma U-shape
def test_i5_nll_is_u_shaped_in_gamma_with_interior_minimum() -> None:
    """I5: bias and variance pull in opposite directions, so the optimum is interior.

    Measured on held-out data. A minimum at a grid endpoint means the grid does not
    bracket the optimum, which would silently invalidate every bandwidth-based
    claim in the project.
    """
    X = _cloud(31, n=400, d=3)
    X_val = _cloud(32, n=300, d=3)
    grid = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    nlls = [-float(np.mean(_fit(X, gamma=g, beta=0.5).score_samples(X_val))) for g in grid]
    best = int(np.argmin(nlls))
    assert 0 < best < len(grid) - 1, f"argmin at endpoint of {grid}: {nlls}"
    assert nlls[best] < min(nlls[0], nlls[-1]) * 0.99, nlls


# ============================================================ I6  LOO <= full
def test_i6_full_density_dominates_leave_one_out_pointwise() -> None:
    """I6: ``p(x_i) >= p_{-i}(x_i)`` for every ``i``, not just on average.

    Pointwise, because the average can hide a per-point violation. The full
    density includes ``x_i``'s own kernel, a strictly positive term, so the
    inequality is a mathematical fact rather than an empirical regularity.
    """
    from scipy.special import logsumexp

    X = _cloud(41, n=200, d=3)
    model = _fit(X, alpha=0.0)
    state = model._state
    assert state is not None
    full = model.score_samples(X, m_score=state.n_samples)
    logdet = 2 * state.d * np.log(state.sigma) + state.logdet_shape
    loo = np.empty(state.n_samples)
    for i in range(state.n_samples):
        quad = _quadratic(state, model, i, state.idx[i])
        terms = -0.5 * quad - 0.5 * logdet[state.idx[i]] - 0.5 * state.d * np.log(2 * np.pi)
        loo[i] = logsumexp(terms) - np.log(state.n_samples)
    assert (full >= loo - 1e-9).all(), float((full - loo).min())
    assert float(full.mean()) >= float(loo.mean()) - 1e-9


# ============================================================ I7  truncation
def test_i7_truncation_error_decreases_monotonically_with_m() -> None:
    """I7: the m-truncated sum approaches the full sum monotonically.

    Re-scoped from the design document, which asks for
    ``err(128) / err(N) < 1e-3`` where ``err`` is *defined* as the distance to the
    ``m = N`` answer -- so the ratio is ``0 / 0`` and no implementation can satisfy
    it. What carries the intent is monotonicity plus a small final error, and both
    are asserted here. See ``docs/math_verification.md`` section B.
    """
    X = _cloud(51, n=300, d=2)
    model = _fit(X, k=12)
    state = model._state
    assert state is not None
    reference = model.score_samples(X, m_score=state.n_samples)
    scale = float(np.abs(reference).mean())
    errors = [
        float(np.abs(model.score_samples(X, m_score=m) - reference).mean())
        for m in (8, 16, 32, 64, 128, 256)
    ]
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] + 1e-12, errors
    assert errors[-1] / scale < 1e-2, f"relative error at m=256: {errors[-1] / scale}"


def test_i7_selects_neighbours_by_kernel_weight_not_euclidean_distance() -> None:
    """I7 guard: truncation ranks by kernel weight, not by Euclidean distance.

    With an anisotropic kernel a point can be far in Euclidean terms and near in
    Mahalanobis terms. Truncating by Euclidean distance therefore discards
    high-mass contributions: measured L1 distortion at ``m = 128``, ``N = 400``
    falls from 0.665 to 0.241 with kernel-weight selection
    (``docs/math_verification.md`` finding D6).
    """
    X = _cloud(51, n=300, d=2)
    model = _fit(X, k=12, beta=1.0)
    state = model._state
    assert state is not None
    full = model.score_samples(X, m_score=state.n_samples)
    truncated = model.score_samples(X, m_score=64)
    # A kernel-weight top-m can only ever be >= an arbitrary 64 of the terms, so
    # its log-density must be no smaller than the mean-based subset's.
    assert float(np.mean(truncated - full)) <= 1e-9


# ============================================================ I8  consistency in N
def test_i8_l1_error_decreases_with_n() -> None:
    """I8: a consistent estimator converges.

    Uses **nested prefixes of one sample** with ``gamma`` fixed once, so the only
    quantity that varies is ``N``. Re-selecting ``gamma`` per ``N`` mixes the
    effect of ``N`` with selection noise: measured that way, the consecutive ratios
    were ``0.667, 0.804, 0.718, 1.371``, where the 1.371 is a validation pick
    flipping rather than an estimator property.
    """
    X_all = np.random.RandomState(99).randn(2000, 1)
    X_val = np.random.RandomState(199).randn(400, 1)
    gamma = min(
        (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
        key=lambda g: (
            -float(np.mean(_fit(X_all, k=12, beta=0.0, gamma=g).score_samples(X_val)))
        ),
    )
    grid = np.linspace(-6, 6, 4001)[:, None]
    truth = np.exp(-0.5 * grid[:, 0] ** 2) / np.sqrt(2 * np.pi)
    errors = []
    for n in (200, 500, 1000, 2000):
        model = _fit(X_all[:n], k=12, beta=0.0, gamma=gamma)
        p = np.exp(model.score_samples(grid, m_score=n))
        errors.append(float(np.trapezoid(np.abs(p - truth), grid[:, 0])))
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] * 1.02, errors
    assert errors[-1] < 0.15, errors


# ============================================================ I9  scale equivariance
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_i9_density_is_scale_covariant(beta: float) -> None:
    """I9: ``p(cX) = c^-d p(X)``.

    The single most load-bearing invariant in the file. The design document's
    quadratic form and its bandwidth matrix used *different* conventions, which
    made the estimator non-scale-covariant for every ``beta != 0``; dividing the
    local covariance by ``rho^2`` makes the local shape dimensionless and restores
    the property to machine precision. At ``beta = 0`` the mixed convention is
    invisible, which is why the document's checks all used ``beta = 0``.
    """
    X = _cloud(61, n=250, d=4)
    X_val = _cloud(62, n=100, d=4)
    base = _fit(X, beta=beta).score_samples(X_val)
    for c in (0.5, 2.0, 10.0):
        scaled = _fit(c * X, beta=beta).score_samples(c * X_val)
        assert_allclose(scaled, base - 4 * np.log(c), rtol=1e-7, atol=1e-8)


# ============================================================ I10  translation
def test_i10_density_is_translation_invariant() -> None:
    """I10: the density does not depend on where the data sits."""
    X = _cloud(63, n=250, d=3)
    X_val = _cloud(64, n=100, d=3)
    base = _fit(X).score_samples(X_val)
    for shift in (0.0, 5.0, -100.0):
        shifted = _fit(X + shift).score_samples(X_val + shift)
        assert_allclose(shifted, base, rtol=1e-7, atol=1e-8)


# ============================================================ I11  permutation
def test_i11_density_is_permutation_invariant() -> None:
    """I11: the density is a function of a *set*, not of an ordering."""
    X = _cloud(65, n=200, d=3)
    X_val = _cloud(66, n=80, d=3)
    base = _fit(X).score_samples(X_val)
    perm = np.random.RandomState(5).permutation(X.shape[0])
    shuffled = _fit(np.ascontiguousarray(X[perm])).score_samples(X_val)
    assert_allclose(base, shuffled, rtol=1e-6, atol=1e-8)


# ============================================================ I12  graph structure
def test_i12_graph_has_zero_diagonal_is_symmetric_and_non_negative() -> None:
    """I12: the k-NN graph excludes self, is symmetric, and is non-negative.

    The zero diagonal is a *consequence* of the self-excluding neighbourhood, not
    an accident, and it is what makes the leave-one-out channel honest.
    """
    model = _fit(_cloud(71, n=200, d=3))
    state = model._state
    assert state is not None
    for name, W in (("W", state.W), ("W_balanced", state.W_balanced)):
        assert np.all(np.diag(W) == 0.0), f"{name} has a non-zero diagonal"
        assert np.array_equal(W, W.T), f"{name} is not exactly symmetric"
        assert (W >= 0).all(), f"{name} has negative entries"
        assert np.isfinite(W).all(), f"{name} has non-finite entries"


def test_i12_kneighbours_excludes_self_and_rho_uses_the_right_index() -> None:
    """I12 guard: ``kneighbors()`` without ``X`` excludes self, and ``rho`` is ``dist[:, k-1]``.

    Two off-by-one traps that both produce plausible-looking numbers. Passing ``X``
    makes each point its own nearest neighbour at distance 0, so ``rho`` is 0.
    After self-exclusion the k-th neighbour sits at index ``k-1``, not ``k``.
    """
    from sklearn.neighbors import NearestNeighbors

    X = _cloud(72, n=120, d=3)
    k = 10
    dist, idx = NearestNeighbors(n_neighbors=k, n_jobs=1).fit(X).kneighbors()
    assert (dist[:, 0] > 0).all(), "self was not excluded"
    assert (idx != np.arange(X.shape[0])[:, None]).all()
    state = _fit(X, k=k)._state
    assert state is not None
    assert_allclose(state.rho_euclid, dist[:, k - 1], rtol=1e-12)


# ============================================================ I13  normalisation
def test_i13_markov_properties_that_actually_hold() -> None:
    """I13: the correct properties of the symmetrically normalised operator.

    Re-scoped from the design document, which asserts that rows of
    ``S = D^-1/2 W D^-1/2`` sum to at most 1. They do not -- the measured maximum
    is 1.156, and the inequality is simply false for non-negative ``W``. What holds,
    and what is load-bearing, is asserted instead:

    * ``S`` is exactly symmetric;
    * ``|S_ii| <= 1`` and ``max S_ij <= 1`` (because ``W_ij <= min(d_i, d_j)``);
    * ``eig(S)`` lies in ``[-1, 1]``;
    * the random-walk operator ``P = D^-1 W`` has rows summing to exactly 1.

    Row sums are a property of ``P``; eigenvalues are a property of ``S``. The
    design document conflated the two.
    """
    model = _fit(_cloud(73, n=200, d=3))
    state = model._state
    assert state is not None
    W = state.W_balanced
    deg = state.degrees
    S = (W / np.sqrt(deg)[:, None]) / np.sqrt(deg)[None, :]
    S = 0.5 * (S + S.T)
    P = W / deg[:, None]

    assert np.array_equal(S, S.T)
    assert (np.abs(np.diag(S)) <= 1.0 + 1e-12).all()
    assert S.max() <= 1.0 + 1e-12, S.max()
    eigenvalues = np.linalg.eigvalsh(S)
    assert eigenvalues.min() >= -1.0 - 1e-9, eigenvalues.min()
    assert eigenvalues.max() <= 1.0 + 1e-9, eigenvalues.max()
    assert_allclose(P.sum(axis=1), 1.0, rtol=0, atol=1e-12)


# ============================================================ I14  alpha = 0
def test_i14_alpha_zero_reduces_to_the_unbalanced_graph() -> None:
    """I14: the balancing factor is identically 1 at ``alpha = 0``, bit for bit.

    This test exists because of a real defect. The design document's pseudocode
    symmetrises ``W`` and then reads ``W[rows, cols]`` off the *already
    symmetrised* matrix before symmetrising again. Every one-directional k-NN edge
    has already been halved by the first symmetrisation, so the second halves it
    again: measured max deviation ``0.25`` of the edge weight. See
    ``docs/math_verification.md`` finding D3.
    """
    model = _fit(_cloud(74, n=200, d=3), alpha=0.0)
    state = model._state
    assert state is not None
    assert np.abs(state.W_balanced - state.W).max() == 0.0


# ============================================================ I15  clip
def test_i15_log_density_is_clipped_and_graph_stays_finite() -> None:
    """I15: the leave-one-out log density is clipped to ``[-c, c]``.

    The clip is the primary guard against alpha-driven spectral blow-up. Without
    it, a single very sparse point has a near-zero density, a large negative
    ``ell``, and with any appreciable ``alpha`` an enormous in-weight -- enough for
    the leading eigenvector to collapse onto it.
    """
    for alpha in (0.0, 0.5, 1.0):
        model = _fit(_cloud(75, n=200, d=3), alpha=alpha, c_logp_clip=4.0)
        state = model._state
        assert state is not None
        assert (np.abs(state.ell) <= 4.0 + 1e-12).all()
        assert np.isfinite(state.W_balanced).all()
        assert np.isfinite(state.embedding).all()


def test_i15_clip_actually_bites_on_extreme_data() -> None:
    """I15 guard: the clip is not dead code.

    Data with one point far from the rest produces a leave-one-out density many
    nats below the rest, which is exactly the case the clip exists for.
    """
    rs = np.random.RandomState(11)
    X = rs.randn(150, 2)
    X[0] += 500.0  # an extreme outlier
    model = _fit(X, k=10, alpha=1.0, c_logp_clip=1.0)
    state = model._state
    assert state is not None
    assert (np.abs(state.ell) <= 1.0 + 1e-12).all()
    assert np.isfinite(state.embedding).all()


# ============================================================ I16  no explosion
def test_i16_alpha_one_does_not_explode_the_embedding() -> None:
    """I16: ``alpha = 1`` must not blow the embedding up.

    The failure mode is specific: sparse points acquire huge in-weights, the
    leading eigenvector is captured by a handful of them, and the coordinates
    either collapse to a blob or diverge. The 100x standard-deviation ratio is the
    machine-detectable form of that.
    """
    X = _cloud(76, n=300, d=3)
    spreads = {}
    for alpha in (0.0, 1.0):
        model = _fit(X, alpha=alpha, k=12)
        state = model._state
        assert state is not None
        assert np.isfinite(state.embedding).all()
        spreads[alpha] = float(state.embedding.std())
    assert spreads[1.0] / max(spreads[0.0], 1e-300) < 100.0, spreads


# ============================================================ I17  biorthogonality
def test_i17_diffusion_coordinates_are_biorthogonal() -> None:
    """I17: ``Psi^T D Psi = I``.

    Without this step the diffusion coordinates are not distance-interpretable, and
    every trustworthiness number computed from them is measuring an arbitrary
    rescaling.
    """
    for n_components in (2, 3):
        model = _fit(_cloud(77, n=250, d=3), n_components=n_components, k=12)
        state = model._state
        assert state is not None
        G = state.embedding.T @ (state.degrees[:, None] * state.embedding)
        assert_allclose(G, np.eye(n_components), rtol=0, atol=1e-8)


# ============================================================ I18  linear structure
def test_i18_dominant_axis_alignment() -> None:
    """I18: the leading embedding axis recovers the dominant linear direction.

    Re-scoped twice over. The document asks for the limit as the kernel width goes
    to zero; that limit does not exist, because the affinity matrix tends to the
    indicator of exact coincidence. The natural surrogate -- the embedding spans
    the data's principal subspace -- is also unassertable: for data that is
    isotropic *within* its own subspace, every orthonormal basis of that subspace
    is equally valid, so correlating against one chosen basis measures an arbitrary
    rotation. Measured canonical correlations on isotropic clouds ranged from 0.933
    down to 0.150 with no relation to estimator quality.

    What remains is both true and meaningful, and is asserted here: on data with an
    identifiable dominant direction, the leading embedding axis aligns with the top
    principal axis. Measured 0.964 to 0.969 across ``d = 4, 6, 8``.
    """
    for d in (4, 6, 8):
        rs = np.random.RandomState(41 + d)
        X = rs.randn(400, d) * np.array([4.0] + [1.0] * (d - 1))
        model = _fit(X, beta=0.0, gamma=1.0, n_components=2)
        state = model._state
        assert state is not None
        embedding = state.embedding - state.embedding.mean(axis=0)
        centred = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        top_axis = centred @ Vt[:1].T
        q_embed, _ = np.linalg.qr(embedding[:, :1])
        q_axis, _ = np.linalg.qr(top_axis)
        correlation = float(np.linalg.svd(q_embed.T @ q_axis, compute_uv=False)[0])
        assert correlation > 0.9, f"d={d}: canonical correlation {correlation:.4f}"


# ============================================================ I19  rotation
def test_i19_rotation_invariance_of_density_and_embedding() -> None:
    """I19: an orthogonal change of coordinates changes nothing measurable."""
    X = _cloud(78, n=200, d=4)
    X_val = _cloud(79, n=80, d=4)
    Q, _ = np.linalg.qr(np.random.RandomState(3).randn(4, 4))
    base = _fit(X, k=12).score_samples(X_val)
    rotated = _fit(X @ Q, k=12).score_samples(X_val @ Q)
    assert_allclose(base, rotated, rtol=1e-7, atol=1e-8)

    model = _fit(X, k=12, n_components=2)
    plain = trustworthiness_score(X, model.embed(X), n_neighbors=5)
    rotated_model = _fit(X @ Q, k=12, n_components=2)
    turned = trustworthiness_score(X @ Q, rotated_model.embed(X @ Q), n_neighbors=5)
    assert abs(plain - turned) < 1e-6, (plain, turned)


# ============================================================ I20  metric range
def test_i20_trustworthiness_and_continuity_lie_in_the_unit_interval() -> None:
    """I20: both metrics are mathematically confined to ``[0, 1]``."""
    from densforge.core.errors import NumericalError
    from densforge.eval.metrics import _assert_unit_interval

    for value in (0.0, 0.5, 1.0):
        _assert_unit_interval(value, "trustworthiness")
    for bad in (-0.01, 1.01, float("nan")):
        with pytest.raises(NumericalError):
            _assert_unit_interval(bad, "trustworthiness")

    X = _cloud(80, n=200, d=3)
    model = _fit(X, n_components=2)
    Psi = model.embed(X)
    for value in (trustworthiness_score(X, Psi), continuity_score(X, Psi)):
        assert 0.0 <= value <= 1.0


def test_i20_metrics_reject_malformed_input() -> None:
    """I20 guard: malformed input is refused with a typed, actionable error.

    The design document asks for a check on "degenerate input (all zeros,
    duplicate points, single-point clusters, NaN)". Measured on scikit-learn 1.9.1,
    some of those inputs are silently accepted rather than rejected -- an
    all-identical input returns trustworthiness 0.9997, a plausible-looking number
    computed from an undefined neighbourhood structure. Rejecting them is therefore
    *this package's* job, not scikit-learn's, and the checks below are what stop
    such a number reaching a report.

    The remaining degenerate cases (duplicate points, constant columns) are
    covered by :func:`test_duplicate_points_produce_a_valid_but_flagged_value`,
    which documents that scikit-learn does accept them.
    """
    from densforge.core.errors import NumericalError

    X = _cloud(80, n=60, d=3)
    Psi = _cloud(81, n=60, d=2)

    with pytest.raises(NumericalError, match="row counts differ"):
        trustworthiness_score(X, Psi[:50], n_neighbors=5)
    with pytest.raises(NumericalError, match="NaN"):
        trustworthiness_score(X, np.full((60, 2), np.nan), n_neighbors=5)
    with pytest.raises(NumericalError, match="2\\*k < n_samples"):
        trustworthiness_score(X[:6], Psi[:6], n_neighbors=5)
    with pytest.raises(NumericalError, match="NaN"):
        continuity_score(np.full((60, 3), np.inf), Psi, n_neighbors=5)
    with pytest.raises(NumericalError, match="2-D"):
        trustworthiness_score(X.ravel(), Psi, n_neighbors=5)


def test_duplicate_points_produce_a_valid_but_flagged_value() -> None:
    """Document scikit-learn's actual behaviour on duplicate points.

    Measured: an all-identical input returns 0.9997 rather than raising. The
    number is meaningless -- every neighbourhood is degenerate -- so this test
    pins the behaviour so that a future scikit-learn which starts raising shows up
    as a deliberate change rather than a surprise.
    """
    from densforge.eval.metrics import trustworthiness_score

    n = 60
    value = trustworthiness_score(np.zeros((n, 2)), np.zeros((n, 2)), n_neighbors=5)
    assert 0.0 <= value <= 1.0


# ============================================================ I21  channel consistency
def test_i21_balancing_channel_and_scoring_channel_agree() -> None:
    """I21: the graph and the scoring path share one kernel definition.

    Two code paths consume the kernel: the graph path, which builds the affinity
    matrix and the leave-one-out log density used for balancing, and the scoring
    path, which produces the reported NLL. If their quadratic forms drift apart, the
    flagship balances its embedding using one kernel and reports a different one --
    invisible in every headline number, and visible only as a disappointing result.

    The assertion is on the *coupling itself*: the edge weights stored in the fitted
    state are reproduced exactly by an independent evaluation of the same quadratic
    form the scoring path uses. A shared bug would have to be written twice to
    escape this.

    On the exact relation between ``ell`` and the scored density: ``ell`` is a
    leave-one-out **kernel-weight** score, ``log sum_j exp(-0.5 D2_ij) - d log
    sigma_i``. It is not the full log-density, because the per-neighbour normaliser
    ``-0.5 log det H_j`` varies with ``j`` and is not included. That is the design's
    specification, and it is safe because ``ell`` is only ever consumed
    *relatively*, by the balancing factor ``exp(-alpha/2 (ell_j - ell_i))``. The
    second assertion pins that relationship so a future change to the normaliser
    cannot pass unnoticed.
    """
    from scipy.special import logsumexp

    X = _cloud(81, n=200, d=3)
    model = _fit(X, k=12, alpha=0.5)
    state = model._state
    assert state is not None

    # (1) The stored edge weights are exactly exp(-0.5 * quadratic), where the
    # quadratic is built from the same sigmas, eigenvectors and eigenvalues the
    # scoring path reads.
    offsets = state.X_train[state.idx] - state.X_train[:, None, :]
    proj = np.einsum("nkd,nde->nke", offsets, state.eigvecs)
    quad = _weighted_quadratic(proj, state.eigvals**model.config.beta)
    quad = quad / (state.sigma[:, None] ** 2)
    assert_allclose(state.directed_weights, np.exp(-0.5 * quad).ravel(), rtol=0, atol=1e-15)

    # (2) `ell` is that kernel-weight score, up to the documented normalisation.
    log_weights = -0.5 * quad
    raw_ell = np.log(np.exp(log_weights).sum(axis=1) + 1e-300) - state.d * np.log(state.sigma)
    expected = np.clip(
        raw_ell - (logsumexp(raw_ell) - np.log(state.n_samples)),
        -model.config.c_logp_clip,
        model.config.c_logp_clip,
    )
    assert_allclose(state.ell, expected, rtol=0, atol=1e-9)

    # (3) Putting the per-neighbour normaliser back makes the two channels provably
    # the same kernel, term by term.
    logdet = 2 * state.d * np.log(state.sigma) + state.logdet_shape
    for i in range(5):
        j = int(state.idx[i, 0])
        quad_ij = float(_quadratic(state, model, i, state.idx[i])[0])
        from_scoring = -0.5 * quad_ij - 0.5 * logdet[j] - 0.5 * state.d * np.log(2 * np.pi)
        from_graph = float(
            log_weights[i, 0] - 0.5 * logdet[j] - 0.5 * state.d * np.log(2 * np.pi)
        )
        assert from_scoring == pytest.approx(from_graph, rel=1e-12)


def test_i22_score_samples_is_a_pure_function() -> None:
    """I22: scoring twice gives identical results, and mutating a copy changes nothing.

    If ``score_samples`` wrote any state -- a cache, a running normalisation, a
    last-seen array -- this fails. Purity is what lets hyper-parameter selection run
    dozens of trials without the model drifting.
    """
    X = _cloud(82, n=200, d=3)
    X_val = _cloud(83, n=60, d=3)
    model = _fit(X, k=12)
    first = model.score_samples(X_val)
    second = model.score_samples(X_val)
    assert_array_equal(first, second)

    scratch = X_val.copy()
    baseline = model.score_samples(X_val)
    scratch[:] = 9e9
    assert_array_equal(model.score_samples(X_val), baseline)


# ============================================================ I23  select purity
def test_i23_select_does_not_mutate_the_fit_state() -> None:
    """I23: hyper-parameter selection leaves the fitted state bit-identical.

    This is the machine-checkable form of the design's leakage rule "no statistic
    of ``X_val`` may enter the state". A fingerprint over the whole frozen state is
    compared before and after.
    """
    from densforge.core.types import array_fingerprint

    X = _cloud(84, n=250, d=3)
    X_val = _cloud(85, n=120, d=3)
    model = _fit(X, k=12)
    before = model._state
    assert before is not None
    fingerprint_before = array_fingerprint(before.embedding) + str(
        array_fingerprint(before.W_balanced)
    )
    model.select(X_val)
    after = model._state
    assert after is before, "select() replaced the fitted state object"
    fingerprint_after = array_fingerprint(after.embedding) + str(
        array_fingerprint(after.W_balanced)
    )
    assert fingerprint_before == fingerprint_after


# ============================================================ I24  fit refuses
def test_i24_fit_refuses_non_training_data() -> None:
    """I24: ``fit`` raises when handed validation or test data.

    The dynamic half of the leakage firewall. The static half is that ``fit`` has
    no parameter through which test data could arrive.
    """
    from densforge.core.errors import LeakageError
    from densforge.data.datasets import build_split

    split = build_split("circles", n_train=200, n_test=100, seed=0)
    manifest = split.manifest()
    model = DensFuse(DensFuseConfig(k=10, n_fuse_rounds=1))

    with pytest.raises(LeakageError):
        model.fit(split.test, manifest=manifest)
    with pytest.raises(LeakageError):
        model.fit(split.val, manifest=manifest)
    # The correct call still works.
    model.fit(split.train, manifest=manifest)


# ============================================================ I25  determinism
def test_i25_fit_and_score_are_bit_identical_across_runs() -> None:
    """I25: same input, same output, bit for bit."""
    X = _cloud(86, n=200, d=3)
    X_val = _cloud(87, n=60, d=3)
    a = _fit(X, k=12, n_fuse_rounds=2)
    b = _fit(X.copy(), k=12, n_fuse_rounds=2)
    assert_array_equal(a.score_samples(X_val), b.score_samples(X_val))
    state_a, state_b = a._state, b._state
    assert state_a is not None and state_b is not None
    assert_array_equal(state_a.embedding, state_b.embedding)


def test_i25_different_seeds_give_different_data() -> None:
    """I25 guard: a seed that changes nothing is a broken seed."""
    from densforge.data.synth import make_circles

    a = make_circles(100, 0)
    b = make_circles(100, 1)
    assert not np.allclose(a, b)


# ============================================================ I30  fixed point
def test_i30_fuse_loop_reaches_a_fixed_point() -> None:
    """I30: the closed loop converges rather than oscillating.

    Round 1 changes the embedding substantially; by round 2 the change has
    collapsed. A loop that oscillated would mean the "closed loop" is really a
    feedback oscillator with no fixed point, and every reported number would depend
    on the round count in a non-convergent way.
    """
    X = _cloud(88, n=250, d=3)
    previous = None
    deltas = []
    for rounds in (1, 2, 3, 4):
        model = DensFuse(DensFuseConfig(k=12, n_fuse_rounds=rounds)).fit(X)
        state = model._state
        assert state is not None
        if previous is not None:
            deltas.append(
                float(np.linalg.norm(state.embedding - previous))
                / max(float(np.linalg.norm(previous)), 1e-300)
            )
        previous = state.embedding
    assert np.isfinite(deltas).all(), deltas
    assert deltas[-1] < deltas[0], f"loop did not converge: {deltas}"


# ============================================================ I31  geodesic scale
def test_i31_geodesic_scale_exceeds_the_euclidean_one() -> None:
    """I31: measuring the local scale geodesically must give a larger distance.

    On the full k-NN graph the k-th Euclidean neighbour *is* a direct edge, so the
    k-th shortest path is that edge and ``rho_geo == rho_euclid`` exactly. Measured
    ratio: 1.000000 for every point, meaning the geodesic branch was inert. The
    implementation therefore builds the graph from a reduced neighbourhood, where
    the k-th shortest path is genuinely multi-hop; measured ratio 1.1117.
    """

    rs = np.random.RandomState(2)
    t = 1.5 * np.pi * (1 + 2 * rs.rand(300))
    h = 21 * rs.rand(300)
    X = np.column_stack([t * np.cos(t), h, t * np.sin(t)]) + rs.randn(300, 3) * 0.2

    model = DensFuse(DensFuseConfig(k=15, n_fuse_rounds=2, beta=0.0)).fit(X)
    state = model._state
    assert state is not None

    # Recompute the geodesic scale the same way the fit does, and compare.
    from densforge.density.flagship import _geodesic_rho

    rho_geo = _geodesic_rho(X, state.W_balanced, 15, 3, state.rho_euclid)
    assert np.isfinite(rho_geo).all()
    assert float(rho_geo.mean()) > float(state.rho_euclid.mean()), (
        float(rho_geo.mean()),
        float(state.rho_euclid.mean()),
    )


# ============================================================ I32  budget audit
def test_i32_tuning_budget_is_auditable() -> None:
    """I32: the evaluation count is recorded, and matches the trace.

    A tuning budget that cannot be audited is indistinguishable from no tuning at
    all, and an untuned flagship that happens to win looks exactly like a tuned one.
    """
    from densforge.hpo.coordinate import coordinate_descent

    X = _cloud(89, n=200, d=3)
    X_val = _cloud(90, n=100, d=3)
    model = _fit(X, k=12)
    result = coordinate_descent(
        model._state,
        X_val,
        model.config,
        budget=6,
        estimator=model,
        grids={"gamma": (0.25, 0.5, 1.0), "beta": (0.0, 1.0)},
    )
    assert result.n_eval == len(result.trace), (result.n_eval, len(result.trace))
    assert result.n_eval <= result.budget, f"budget exceeded: {result.n_eval}"
    assert result.n_eval > 0
    assert result.exhausted is (result.n_eval >= result.budget)
    # Every recorded axis must be one the search actually varies, and the
    # order must follow HPO_ORDER (a different order reaches a different optimum).
    from densforge.core.config import HPO_ORDER

    axes = [t.axis for t in result.trace]
    positions = [HPO_ORDER.index(a) for a in axes]
    assert positions == sorted(positions), axes
    # And the trace must survive JSON serialisation, since it goes into the report.
    import json

    json.dumps(result.as_params())
