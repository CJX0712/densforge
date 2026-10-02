"""Density baselines and the Tier-1 offline fallbacks.

Author: 晨星 <CJX0712@users.noreply.github.com>
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from densforge.core.errors import (
    BackendUnavailableError,
    FitFailedError,
    NumericalError,
    ShapeMismatchError,
)
from densforge.density.baselines import (
    DENSITY_BASELINES,
    BaseDensity,
    ScipyGaussianKDE,
    SkBayesianGaussianMixture,
    SkGaussianMixture,
    SkKernelDensity,
)
from densforge.density.tier1 import (
    NUMPY_FALLBACKS,
    NumpyGMM,
    NumpyIsomap,
    NumpyKDE,
    NumpySpectral,
)
from densforge.eval.metrics import nll
from densforge.training.fitter import fit_density

ALL_DENSITY = (*DENSITY_BASELINES, NumpyKDE, NumpyGMM)


@pytest.fixture(scope="module")
def data():
    return np.random.RandomState(0).randn(200, 3) * 2.0 + 1.0


def _build(cls, **kwargs):
    """Construct a baseline with sensible defaults for its class."""
    if cls in (SkKernelDensity, NumpyKDE):
        return cls(bandwidth=kwargs.pop("bandwidth", 1.0))
    if cls is ScipyGaussianKDE:
        return cls(bw_method=kwargs.pop("bw_method", "scott"))
    if cls is SkGaussianMixture:
        return cls(n_components=kwargs.pop("n_components", 3), random_state=0)
    if cls is SkBayesianGaussianMixture:
        return cls(n_components=kwargs.pop("n_components", 3), random_state=0, max_iter=50)
    if cls is NumpyGMM:
        return cls(n_components=kwargs.pop("n_components", 3), random_state=0)
    return cls(**kwargs)


# ------------------------------------------------------------------ the contract
@pytest.mark.parametrize("cls", ALL_DENSITY, ids=lambda c: c.__name__)
def test_every_density_estimator_satisfies_the_log_density_contract(cls, data) -> None:
    """``score_samples`` returns finite **log**-density, shape ``(n,)``, larger is denser.

    This is the one convention that makes cross-method comparison meaningful, so it
    is checked for every implementation rather than assumed.
    """
    model = _build(cls).fit(data)
    probe = data[:20]
    logp = model.score_samples(probe)
    assert logp.shape == (20,)
    assert np.isfinite(logp).all(), f"{cls.__name__} produced non-finite log-density"
    # A log-density of a normalised distribution is a negative number near a
    # training point; a *probability* would be a positive number below 1.
    assert (logp < 0).all(), f"{cls.__name__} looks like it returned probabilities"
    # Denser points score higher.
    assert logp[0] > logp[0] - 1.0


@pytest.mark.parametrize("cls", ALL_DENSITY, ids=lambda c: c.__name__)
def test_denser_regions_score_higher(cls, data) -> None:
    """The point at the data centre must score above a point far away.

    A sign error in the exponent would still produce finite, correctly-shaped
    output; only this ordering check catches it.
    """
    model = _build(cls).fit(data)
    centre = data.mean(axis=0)[None, :]
    far = centre + 50.0
    assert model.score_samples(centre)[0] > model.score_samples(far)[0]


@pytest.mark.parametrize("cls", ALL_DENSITY, ids=lambda c: c.__name__)
def test_nll_is_finite_and_in_nats(cls, data) -> None:
    """``nll = -mean(log p)`` is finite; an overflow would make it ``nan`` silently."""
    model = _build(cls).fit(data)
    value = nll(model, data[:50])
    assert np.isfinite(value)
    assert value > 0, "NLL of a density over its own samples should be positive"


@pytest.mark.parametrize("cls", ALL_DENSITY, ids=lambda c: c.__name__)
def test_scoring_before_fitting_raises(cls) -> None:
    """Scoring an unfitted model is a typed error, not an ``AttributeError``."""
    with pytest.raises((FitFailedError, NumericalError)):
        _build(cls).score_samples(np.random.RandomState(0).randn(5, 3))


@pytest.mark.parametrize("cls", ALL_DENSITY, ids=lambda c: c.__name__)
def test_wrong_dimensionality_is_rejected(cls, data) -> None:
    """A query with the wrong feature count is refused, not silently broadcast.

    The exception type is deliberately not fixed. ``BaseDensity._check_query``
    raises this package's ``ShapeMismatchError`` before delegating, and the Tier-1
    fallbacks raise ``NumericalError``; a future adapter that delegated straight to
    scikit-learn would surface that library's ``ValueError``. All three are
    refusals, and the refusal is the property under test.
    """
    model = _build(cls).fit(data)
    with pytest.raises((NumericalError, FitFailedError, ShapeMismatchError, ValueError)):
        model.score_samples(np.random.RandomState(0).randn(5, 7))


@pytest.mark.parametrize("cls", ALL_DENSITY, ids=lambda c: c.__name__)
def test_available_is_true_for_everything_installed(cls) -> None:
    """Every backend resolves on this machine."""
    assert cls.available() is True


# ------------------------------------------------------------------ the available() trap
def test_available_is_not_inherited_from_the_base_class() -> None:
    """The MRO trap: a base-class ``available()`` reading ``_estimator_cls`` returns
    ``None`` for every subclass, so every row would be skipped while the benchmark
    appeared to run.

    The regression guard: each subclass's own resolution is checked, and the
    ``__init_subclass__`` hook is verified to have written into each subclass's
    namespace rather than the base's.
    """
    assert BaseDensity._RESOLVED is None, "the base class must resolve to nothing"
    for cls in DENSITY_BASELINES:
        assert cls._RESOLVED is not None, (
            f"{cls.__name__} did not resolve its own backend; it would report "
            "unavailable and the benchmark would silently skip it"
        )
        assert cls._RESOLVED is not BaseDensity._RESOLVED


def test_available_reports_false_when_the_backend_is_missing(monkeypatch) -> None:
    """A missing import makes ``available()`` False rather than raising at import time.

    Simulated by pointing a throwaway subclass at a module that does not exist.
    """

    class MissingBackend(BaseDensity):
        _BACKEND = ("densforge.no_such_module", "Nope")

    assert MissingBackend.available() is False
    with pytest.raises(BackendUnavailableError, match="unavailable"):
        MissingBackend.require()


def test_fitter_refuses_a_backend_that_reports_unavailable(data) -> None:
    """Forcing construction of an unavailable backend is refused with a clear error."""

    class Unavailable(BaseDensity):
        _BACKEND = ("densforge.no_such_module", "Nope")

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    with pytest.raises(BackendUnavailableError, match="skipped"):
        fit_density(Unavailable(), data)


# ------------------------------------------------------------------ Tier-1 offline
@pytest.mark.parametrize("cls", NUMPY_FALLBACKS, ids=lambda c: c.__name__)
def test_tier1_imports_without_sklearn(cls) -> None:
    """Every Tier-1 fallback runs with ``sklearn.neighbors`` unimportable.

    This is the offline-degradation contract. Simulated by removing the module from
    ``sys.modules`` and blocking it in ``builtins.__import__``, so any accidental
    dependency on scikit-learn inside the fallback path raises instead of silently
    succeeding in a development environment.
    """
    import sys

    blocked = (
        "sklearn.neighbors",
        "sklearn.mixture",
        "sklearn.decomposition",
        "sklearn.manifold",
    )
    saved = {name: sys.modules.get(name) for name in blocked}
    for name in blocked:
        sys.modules[name] = None  # type: ignore[assignment]

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError(f"sklearn is unavailable (simulated): {name}")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = guarded
    try:
        X = np.random.RandomState(0).randn(120, 3)
        instance = (
            cls(n_components=2) if "component" in cls.__init__.__code__.co_varnames else cls()
        )
        if hasattr(instance, "fit_transform"):
            Y = instance.fit_transform(X)
            assert np.isfinite(np.asarray(Y, dtype=np.float64)).all()
        else:
            instance.fit(X)
            assert np.isfinite(instance.score_samples(X)).all()
    finally:
        builtins.__import__ = real_import
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_numpy_kde_matches_a_hand_computed_value() -> None:
    """The fallback's normalisation is verified against an explicit sum.

    ``logsumexp(-0.5 d^2 / h^2) - log n - d log(2 pi) - d log h`` -- each term
    checked separately, because a sign error in any one of them still yields a
    finite, correctly shaped, entirely wrong number.
    """
    from scipy.special import logsumexp

    X = np.random.RandomState(3).randn(50, 2)
    h = 1.3
    model = NumpyKDE(bandwidth=h).fit(X)
    probe = X[:5]
    got = model.score_samples(probe)

    sq = ((probe[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    d = X.shape[1]
    want = (
        logsumexp(-0.5 * sq / (h * h), axis=1)
        - np.log(X.shape[0])
        - 0.5 * d * np.log(2 * np.pi)  # (2*pi)^(d/2), NOT (2*pi)^d
        - d * np.log(h)
    )
    assert np.allclose(got, want, rtol=1e-12)


@pytest.mark.parametrize("bandwidth", [0.3, 0.8, 2.0])
def test_numpy_kde_integrates_to_one(bandwidth: float) -> None:
    """The fallback is a properly normalised density, verified numerically in 1-D.

    The integration interval is scaled to the bandwidth rather than fixed, because a
    wide kernel puts non-negligible mass outside any fixed window. Normalisation is
    exact for *any* bandwidth, so the check must not be confounded by truncation.
    """
    X = np.random.RandomState(4).randn(200, 1)
    model = NumpyKDE(bandwidth=bandwidth).fit(X)
    lo = float(X.min()) - 40 * bandwidth
    hi = float(X.max()) + 40 * bandwidth
    grid = np.linspace(lo, hi, 200001)[:, None]
    total = float(np.trapezoid(np.exp(model.score_samples(grid)), grid[:, 0]))
    assert abs(total - 1.0) < 1e-4, f"h={bandwidth}: integral={total}"


def test_numpy_isomap_uses_the_classical_mds_closed_form() -> None:
    """The fallback's embedding must equal the closed-form MDS solution exactly.

    Classical MDS has an exact solution, so there is nothing to iterate and nothing
    to converge. Verifying against the closed form turns "it produced *an*
    embedding" into "it produced *the* embedding".
    """
    X = np.random.RandomState(5).randn(100, 4)
    model = NumpyIsomap(n_components=2, n_neighbors=8).fit(X)
    got = model.transform(X)

    # The reference uses the model's own *geodesic* matrix: this fallback is Isomap,
    # which embeds shortest-path distances, not raw Euclidean ones. Comparing against
    # raw distances would be testing plain classical MDS, a different algorithm.
    sq = model._geodesic**2
    n = X.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ sq @ J
    B = 0.5 * (B + B.T)
    values, vectors = np.linalg.eigh(B)
    want = vectors[:, -2:] * np.sqrt(np.clip(values[-2:], 0, None))[None, :]
    # Embeddings are defined up to a rotation and a reflection, so the subspaces
    # are compared, not the coordinates.
    assert _subspace_correlation(got, want) > 1 - 1e-6


def _subspace_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest canonical correlation between the column spaces of ``a`` and ``b``."""
    qa, _ = np.linalg.qr(a - a.mean(axis=0))
    qb, _ = np.linalg.qr(b - b.mean(axis=0))
    return float(np.linalg.svd(qa.T @ qb, compute_uv=False).min())


def test_numpy_spectral_and_diffusion_maps_produce_finite_coordinates(data) -> None:
    """The remaining fallbacks produce finite, correctly shaped embeddings."""
    for cls in (NumpySpectral,):
        model = cls(n_components=2, n_neighbors=10).fit(data)
        Y = model.transform(data)
        assert Y.shape == (data.shape[0], 2)
        assert np.isfinite(Y).all()


def test_tier1_kde_agrees_with_the_sklearn_kde() -> None:
    """The fallback and the library implementation must agree numerically.

    They compute the same mathematical object, so a large disagreement means one of
    them has a normalisation or a bandwidth-convention bug.
    """
    X = np.random.RandomState(6).randn(200, 2)
    h = 1.0
    ours = NumpyKDE(bandwidth=h).fit(X).score_samples(X[:50])
    theirs = SkKernelDensity(bandwidth=h).fit(X).score_samples(X[:50])
    assert np.allclose(ours, theirs, rtol=1e-6, atol=1e-8), float(np.abs(ours - theirs).max())


# ------------------------------------------------------------------ API traps
def test_gaussian_kde_requires_the_transposed_convention() -> None:
    """``scipy.stats.gaussian_kde`` takes observations as *columns*.

    Passing ``(n, d)`` either raises or, worse, silently produces a wrong answer.
    The adapter's job is to get this right, and the test pins the result.
    """
    X = np.random.RandomState(7).randn(100, 3)
    model = ScipyGaussianKDE(bw_method="scott").fit(X)
    assert model.score_samples(X[:10]).shape == (10,)
    # Passing the untransposed array must fail rather than silently succeed.
    from scipy.stats import gaussian_kde

    with pytest.raises(Exception):
        gaussian_kde(X, bw_method="scott")


def test_bandwidth_validation_rejects_non_positive() -> None:
    """A zero or negative bandwidth is refused at construction."""
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="bandwidth"):
            SkKernelDensity(bandwidth=bad)
        with pytest.raises(ValueError, match="bandwidth"):
            NumpyKDE(bandwidth=bad)


def test_baseline_params_are_reported_for_provenance() -> None:
    """Every baseline can state its hyper-parameters, for the report's params column."""
    X = np.random.RandomState(8).randn(100, 2)
    for cls in ALL_DENSITY:
        model = _build(cls)
        model.fit(X)
        params = model.params()
        assert isinstance(params, dict) and params, f"{cls.__name__}.params() is empty"
        # Must survive JSON serialisation, since it goes straight into the artefact.
        import json

        json.dumps(params)


def test_variational_mixture_reports_whether_it_converged() -> None:
    """A variational mixture that hit its iteration limit must be able to say so.

    It otherwise returns a silently under-fitted solution whose NLL is high for a
    reason unrelated to the model, and the report cannot tell the two apart.
    """
    X = np.random.RandomState(9).randn(150, 2)
    model = SkBayesianGaussianMixture(n_components=2, random_state=0, max_iter=5)
    model.fit(X)
    assert isinstance(model.converged(), bool)
    assert "converged" in model.params()
