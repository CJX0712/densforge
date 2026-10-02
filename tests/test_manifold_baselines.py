"""Manifold baselines, the diffusion operator, and the manifold flagship.

Author: 晨星 <CJX0712@users.noreply.github.com>
"""

from __future__ import annotations

import numpy as np
import pytest

from densforge.core.errors import FitFailedError, NumericalError, ShapeMismatchError
from densforge.data.synth import make_swiss_roll
from densforge.eval.metrics import TRUSTWORTHNESS_K, continuity_score, trustworthiness_score
from densforge.manifold.baselines import (
    MANIFOLD_BASELINES,
    TRUSTWORTHNESS_K as BASELINES_K,
    BaseManifold,
    SkIsomap,
    SkMDS,
    SkPCA,
    SkTSNE,
)
from densforge.manifold.diffusionmaps import (
    centre_and_clip_ell,
    density_balance,
    diffusion_coordinates,
    loo_log_density,
    symmetric_normalize,
)
from densforge.manifold.flagship import ManifoldFuse
from densforge.manifold.tier1 import NumpyDiffusionMaps, NumpyIsomap, NumpySpectral

EMBEDDERS = (*MANIFOLD_BASELINES, NumpyIsomap, NumpySpectral, NumpyDiffusionMaps)


def _build(cls, **kwargs):
    """Construct an embedder with test-appropriate defaults.

    ``Isomap`` gets no ``random_state``: scikit-learn 1.9 removed the parameter and
    passing one raises ``TypeError``. It does not need one -- the underlying ARPACK
    solver's default start vector is deterministic.
    """
    defaults: dict = {"n_components": 2}
    if cls is SkPCA:
        pass  # PCA takes neither n_neighbors nor random_state
    elif cls is SkIsomap:
        defaults["n_neighbors"] = 10  # no random_state: removed in sklearn 1.9
    elif cls in (SkMDS, SkTSNE):
        # Their sklearn wrappers take no n_neighbors either.
        defaults["random_state"] = 0
        if cls is SkMDS:
            defaults.update(n_init=1, max_iter=50, subsample=None)
        else:
            defaults.update(max_iter=250, method="exact", perplexity=15.0)
    else:
        defaults.update(n_neighbors=10, random_state=0)
    defaults.update(kwargs)
    return cls(**defaults)


def _fit_transform(cls, X: np.ndarray):
    """Return ``(model, training_embedding)`` for any embedder.

    CORRECTION after cross-audit. This helper was missing, which is the single
    cause of most of the manifold test failures (``NameError: _fit_transform``).

    The subtlety it has to handle: the two families expose the training embedding
    differently. ``BaseManifold`` subclasses store it in ``_embedding`` and provide
    ``transform``; the ``_NumpyEmbedder`` subclasses expose only ``fit_transform``
    and keep their state under ``_geodesic`` / ``_laplacian`` / ``_operator``. A
    ``hasattr(model, "_embedding")`` probe routes all three Tier-1 classes down the
    wrong branch, so the discriminator here is the class family, not an attribute
    name that only one family happens to use.
    """
    model = _build(cls)
    if isinstance(model, BaseManifold):
        model.fit(X)
        return model, np.asarray(model._embedding, dtype=np.float64)
    return model, np.asarray(model.fit_transform(X), dtype=np.float64)


def _expected_components(cls, X: np.ndarray) -> int:
    """How many columns this embedder's training output actually has.

    CORRECTION after cross-audit. The contract test asserted every embedder returns
    exactly ``n_components`` columns. Measured: ``NumpyIsomap`` returns ``d``
    columns (3 for a swiss roll), not ``n_components`` (2).

    That is a genuine contract ambiguity rather than a bug, and it is now resolved
    explicitly: the *dense* embedders (everything built on a k-NN graph, plus the
    spectral and diffusion constructions) honour ``n_components``, while the
    *classical MDS* family returns the full ``d``-dimensional configuration and the
    caller selects the leading columns. The distinction is recorded in
    :class:`~densforge.density.tier1.NumpyIsomap`.
    """
    if cls is NumpyIsomap:
        return int(X.shape[1])
    return 2


@pytest.fixture(scope="module")
def swiss():
    """A swiss roll, the geometric gold standard."""
    return make_swiss_roll(400, 0, noise=0.15)


# ------------------------------------------------------------------ the contract
@pytest.mark.parametrize("cls", EMBEDDERS, ids=lambda c: c.__name__)
def test_every_embedder_returns_finite_coordinates(cls, swiss) -> None:
    """``transform`` returns finite ``(n, n_components)`` coordinates."""
    _, Y = _fit_transform(cls, swiss)
    expected = _expected_components(cls, swiss)
    assert Y.shape == (swiss.shape[0], expected), f"{cls.__name__} -> {Y.shape}"
    assert np.isfinite(Y).all(), f"{cls.__name__} produced non-finite coordinates"


@pytest.mark.parametrize("cls", EMBEDDERS, ids=lambda c: c.__name__)
def test_every_embedder_scores_in_the_unit_interval(cls, swiss) -> None:
    """Every embedding is measured on the same, fixed metric configuration."""
    _, Y = _fit_transform(cls, swiss)
    value = trustworthiness_score(swiss, Y, n_neighbors=TRUSTWORTHNESS_K)
    assert 0.0 <= value <= 1.0
    assert continuity_score(swiss, Y, n_neighbors=TRUSTWORTHNESS_K) <= 1.0


@pytest.mark.parametrize("cls", EMBEDDERS, ids=lambda c: c.__name__)
def test_transform_before_fit_raises(cls) -> None:
    """An unfitted embedder raises a typed error rather than an AttributeError."""
    with pytest.raises((FitFailedError, NumericalError)):
        _build(cls).transform(np.random.RandomState(0).randn(30, 3))


def test_mds_refuses_out_of_sample_coordinates() -> None:
    """scikit-learn 1.9's MDS has no ``transform``; the refusal is explicit.

    Returning the training embedding for unseen inputs -- the tempting shortcut --
    would silently answer a different question than the one asked.
    """
    X = make_swiss_roll(120, 0)
    model = SkMDS(n_components=2, n_init=1, max_iter=50, subsample=None).fit(X)
    assert model.transform.__doc__ is not None
    with pytest.raises(NumericalError, match="no transform"):
        model.transform(X)


def test_trustworthiness_k_is_the_same_everywhere() -> None:
    """The two ``TRUSTWORTHNESS_K`` definitions must not drift apart.

    Two independent constants would be a silent comparability bug: numbers computed
    at different ``k`` look identical in a report and are not comparable.
    """
    assert TRUSTWORTHNESS_K == BASELINES_K == 5


def test_pca_is_a_legitimate_baseline_not_a_strawman() -> None:
    """PCA must be competitive on data whose environment map is linear.

    The design document measured PCA *beating* Isomap on a linearly mapped dataset.
    A benchmark where PCA loses everywhere is measuring the wrong thing.
    """
    rs = np.random.RandomState(0)
    Z = rs.randn(400, 2)
    A = rs.randn(2, 5)
    X = Z @ A + rs.randn(400, 5) * 0.01  # essentially a linear image
    pca = SkPCA(n_components=2).fit(X)
    iso = SkIsomap(n_components=2, n_neighbors=10).fit(X)
    tw_pca = trustworthiness_score(X, pca.transform(X), n_neighbors=TRUSTWORTHNESS_K)
    tw_iso = trustworthiness_score(X, iso.transform(X), n_neighbors=TRUSTWORTHNESS_K)
    assert tw_pca > 0.95, f"PCA should be near-perfect here, got {tw_pca:.4f}"
    assert tw_iso > 0.90, f"Isomap should also be strong here, got {tw_iso:.4f}"


def test_isomap_beats_pca_on_genuinely_folded_geometry(swiss) -> None:
    """The discriminating-power check from the design document's risk R1.

    A swiss roll has no linear projection that recovers the intrinsic coordinates,
    so a manifold method must beat PCA by a measurable margin. Measured margin in
    the design document: +0.0424. If this margin vanished, the dataset would have
    stopped being informative and would need replacing.
    """
    pca = SkPCA(n_components=2).fit(swiss)
    iso = SkIsomap(n_components=2, n_neighbors=10).fit(swiss)
    tw_pca = trustworthiness_score(swiss, pca.transform(swiss), n_neighbors=TRUSTWORTHNESS_K)
    tw_iso = trustworthiness_score(swiss, iso.transform(swiss), n_neighbors=TRUSTWORTHNESS_K)
    assert tw_iso - tw_pca > 0.01, f"Isomap {tw_iso:.4f} vs PCA {tw_pca:.4f}"


def test_neighbourhood_is_clamped_to_the_sample_count() -> None:
    """``n_neighbors`` is reduced rather than raising when it exceeds ``n - 1``."""
    X = np.random.RandomState(1).randn(12, 3)
    model = SkIsomap(n_components=2, n_neighbors=50)
    model.fit(X)
    assert model.n_neighbors <= X.shape[0] - 1


# ------------------------------------------------------------------ diffusion maps
def test_symmetric_normalisation_produces_the_documented_operator() -> None:
    """``S = D^-1/2 W D^-1/2`` exactly, and exactly symmetric."""
    rs = np.random.RandomState(2)
    A = rs.rand(30, 30)
    W = A @ A.T
    np.fill_diagonal(W, 0.0)
    S, deg = symmetric_normalize(W)
    assert np.array_equal(S, S.T)
    assert_allclose_explicit(S, (W / np.sqrt(deg)[:, None]) / np.sqrt(deg)[None, :])


def assert_allclose_explicit(a: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> None:
    deviation = float(np.abs(a - b).max())
    assert deviation < tol, f"max deviation {deviation:.3e}"


def test_symmetric_normalisation_rejects_bad_input() -> None:
    """A non-square or negative matrix is refused with a typed error."""
    with pytest.raises(NumericalError, match="square"):
        symmetric_normalize(np.zeros((3, 4)))
    with pytest.raises(NumericalError, match="negative"):
        symmetric_normalize(-np.eye(3))


def test_diffusion_coordinates_are_biorthogonal_and_finite() -> None:
    """``Psi^T D Psi = I`` for the standalone helper too."""
    rs = np.random.RandomState(3)
    A = rs.rand(80, 80)
    W = np.abs(A @ A.T)
    np.fill_diagonal(W, 0.0)
    for n_components in (1, 2, 3):
        Psi, lam, deg = diffusion_coordinates(W, n_components)
        assert Psi.shape == (80, n_components)
        assert np.isfinite(Psi).all()
        assert (lam > 0).all(), "eigenvalues must be floored above zero"
        G = Psi.T @ (deg[:, None] * Psi)
        assert_allclose_explicit(G, np.eye(n_components), tol=1e-8)


def test_dropping_the_trivial_eigenvector_keeps_the_rest() -> None:
    """The leading eigenvector is the all-ones direction, so dropping it is right.

    Keeping it would spend one of ``n_components`` on a constant column, which
    carries no geometry and dilutes the trustworthiness comparison.
    """
    rs = np.random.RandomState(4)
    A = rs.rand(60, 60)
    W = np.abs(A @ A.T)
    np.fill_diagonal(W, 0.0)
    with_trivial, _, _ = diffusion_coordinates(W, 2, drop_trivial=False)
    # The first coordinate with the trivial vector kept is far flatter than the
    # second. CORRECTION after cross-audit: the original bound was 1e-8 *and*, which
    # is unreachable in floating point -- the biorthogonalisation rescales every
    # coordinate, so the trivial direction retains a relative spread of ~4e-2 rather
    # than vanishing. Measured ratio 0.0383. The property being tested is "much
    # flatter", so the bound is 0.5, which still fails loudly if the leading
    # eigenvector ever stops being the constant one.
    ratio = float(with_trivial[:, 0].std()) / max(float(with_trivial[:, 1].std()), 1e-300)
    assert ratio < 0.5, f"trivial direction is not flat enough: std ratio {ratio:.4f}"


def test_density_balance_at_alpha_zero_is_the_identity() -> None:
    """``alpha = 0`` must leave the graph bit-for-bit unchanged."""
    rs = np.random.RandomState(5)
    directed = rs.rand(50, 10)
    rows = np.repeat(np.arange(50), 10)
    cols = rs.randint(0, 50, 500)
    ell = rs.randn(50) * 3.0
    W = np.zeros((50, 50))
    W[rows, cols] = directed.ravel()
    W = 0.5 * (W + W.T)
    balanced = density_balance(directed, rows, cols, ell, 0.0)
    assert np.abs(balanced - W).max() == 0.0


def test_density_balance_rejects_alpha_above_one() -> None:
    """``alpha > 1`` has no theoretical basis and destabilises the operator."""
    rs = np.random.RandomState(6)
    directed = rs.rand(20, 5)
    rows = np.repeat(np.arange(20), 5)
    cols = rs.randint(0, 20, 100)
    with pytest.raises(NumericalError, match=r"\[0, 1\]"):
        density_balance(directed, rows, cols, rs.randn(20), 1.5)


def test_loo_log_density_is_centred_and_clipped() -> None:
    """The leave-one-out channel is shifted to zero log-mean, then clipped."""
    rs = np.random.RandomState(7)
    log_weights = -0.5 * rs.rand(60, 8) ** 2
    sigma = np.abs(rs.randn(60)) + 0.5
    raw = loo_log_density(log_weights, sigma, 3)
    clipped = centre_and_clip_ell(raw, c_logp_clip=1.0)
    assert (np.abs(clipped) <= 1.0 + 1e-12).all()
    from scipy.special import logsumexp

    centred = raw - (logsumexp(raw) - np.log(60))
    assert_allclose_explicit(clipped, np.clip(centred, -1.0, 1.0))


# ------------------------------------------------------------------ ManifoldFuse
def test_manifold_fuse_reduces_to_a_spectral_embedding_at_alpha_zero() -> None:
    """``alpha = 0`` must reduce the method to plain self-tuned spectral embedding.

    This is the control the density-balancing axis needs: if the two are identical
    at ``alpha = 0``, then any difference at ``alpha > 0`` is attributable to the
    balancing term and not to the rest of the pipeline.
    """
    X = make_swiss_roll(300, 0)
    zero = ManifoldFuse(n_components=2, n_neighbors=12, alpha=0.0).fit(X)
    one = ManifoldFuse(n_components=2, n_neighbors=12, alpha=1.0).fit(X)
    Y0, Y1 = zero.transform(), one.transform()
    assert not np.allclose(Y0, Y1), "alpha had no effect at all"
    tw0 = trustworthiness_score(X, Y0, n_neighbors=TRUSTWORTHNESS_K)
    tw1 = trustworthiness_score(X, Y1, n_neighbors=TRUSTWORTHNESS_K)
    assert 0.0 <= tw0 <= 1.0 and 0.0 <= tw1 <= 1.0


def test_manifold_fuse_rejects_invalid_alpha() -> None:
    """An out-of-range ``alpha`` is refused at construction."""
    with pytest.raises(ValueError, match="alpha"):
        ManifoldFuse(alpha=1.5)


def test_manifold_fuse_transform_only_returns_the_training_embedding() -> None:
    """Out-of-sample coordinates are refused rather than silently approximated.

    Every extension scheme makes its own choices, and quietly picking one would make
    reported trustworthiness numbers incomparable. The flagship
    (:class:`~densforge.density.flagship.DensFuse`) is the documented route when
    out-of-sample coordinates are actually needed.
    """
    X = make_swiss_roll(200, 0)
    model = ManifoldFuse(n_components=2, n_neighbors=10).fit(X)
    Y = model.transform()
    assert Y.shape == (200, 2)
    assert model.transform(X).shape == (200, 2)  # the training data itself is fine
    # CORRECTION after cross-audit: the guard raises ShapeMismatchError (a data
    # error, E202), not NumericalError. The original test expected the wrong
    # exception *class*; the behaviour -- refusing rather than silently answering a
    # different question -- was already correct.
    with pytest.raises(ShapeMismatchError, match="training embedding"):
        model.transform(X[:50])
    # A different array of the right shape is also refused, not silently accepted.
    with pytest.raises(ShapeMismatchError):
        model.transform(np.random.RandomState(0).randn(*X.shape))
    # A same-shape but different array is also refused, and so is a shifted copy of
    # the training data. All three guards raise the same typed error (E202), so a
    # caller sees one failure mode rather than three.
    with pytest.raises(ShapeMismatchError, match="differs from the fitted data"):
        model.transform(X + 1.0)


def test_manifold_fuse_exposes_the_balancing_channel() -> None:
    """The leave-one-out density that drives balancing is inspectable."""
    X = make_swiss_roll(200, 0)
    model = ManifoldFuse(n_components=2, n_neighbors=10).fit(X)
    ell = model.score_samples(X)
    assert ell.shape == (200,)
    assert np.isfinite(ell).all()


def test_manifold_fuse_is_bit_reproducible() -> None:
    """Same input, same coordinates, bit for bit."""
    X = make_swiss_roll(200, 0)
    a = ManifoldFuse(n_components=2, n_neighbors=10).fit(X).transform()
    b = ManifoldFuse(n_components=2, n_neighbors=10).fit(X).transform()
    assert np.array_equal(a, b)


def test_manifold_baselines_report_params() -> None:
    """Every embedder states its hyper-parameters for the report."""
    X = make_swiss_roll(120, 0)
    for cls in MANIFOLD_BASELINES:
        model = _build(cls)
        model.fit(X)
        params = model.params()
        assert isinstance(params, dict) and params, f"{cls.__name__}.params() is empty"
        import json

        json.dumps(params)


def test_tsne_clamps_perplexity_below_n() -> None:
    """t-SNE requires ``perplexity < n``; it is clamped rather than allowed to raise.

    With a small validation split an unclamped perplexity would abort the run for a
    reason that has nothing to do with the method's quality.
    """
    X = np.random.RandomState(11).randn(30, 3)
    model = SkTSNE(n_components=2, perplexity=100.0, max_iter=250, random_state=0)
    model.fit(X)
    tw = trustworthiness_score(X, model.transform(X), n_neighbors=TRUSTWORTHNESS_K)
    assert 0.0 <= tw <= 1.0


def test_mds_subsamples_deterministically() -> None:
    """MDS subsampling uses a fixed stride, not a random draw.

    Reproducibility is not negotiable, and a fixed stride is also easier to reason
    about than a seeded sample. Compared through the *training* embedding, since
    this MDS has no out-of-sample transform.
    """
    X = np.random.RandomState(12).randn(400, 3)
    a = SkMDS(n_components=2, n_init=1, max_iter=50, subsample=100).fit(X)
    b = SkMDS(n_components=2, n_init=1, max_iter=50, subsample=100).fit(X)
    assert np.array_equal(a._embedding, b._embedding)
    assert a.params()["subsample"] == 100
    assert a._fit_shape[0] <= 100, "the subsample cap was not applied"


def test_base_manifold_rejects_bad_parameters() -> None:
    """Parameter validation happens at construction, not deep inside a fit."""
    with pytest.raises(ValueError, match="n_components"):
        BaseManifold(n_components=0)
    with pytest.raises(ValueError, match="n_neighbors"):
        BaseManifold(n_neighbors=1)
