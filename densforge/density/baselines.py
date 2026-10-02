"""Tier-0 density baselines: thin adapters over scikit-learn and SciPy.

Author: 晨星 <CJX0712@users.noreply.github.com>

The ``available()`` trap
-----------------------
A class-level lazy factory that reads a base-class attribute silently reports
"unavailable" for **every** subclass::

    class Base:
        _estimator_cls = None
        @classmethod
        def available(cls) -> bool:
            return cls._estimator_cls is not None     # reads Base's None, always False

``cls`` binds correctly, but attribute lookup walks the MRO and finds the *base*
class's ``None`` first when the subclass has not overridden it. Every algorithm
then reports False and the benchmark skips everything while appearing to run.

The fix used throughout this package is :meth:`_BackendMixin.available`, which
resolves the backend through an explicit registry populated by
``__init_subclass__`` at class-creation time. Each subclass therefore gets its
own resolved entry, and the base class is never asked to guess.

A companion rule from the same source: ``available()`` must be a pure boolean
predicate. It must not construct an estimator, because the benchmark calls it once
per row and construction can be expensive.

Authorisation to tune baselines
-------------------------------
Every baseline's hyper-parameter is selected on the validation set with the same
budget as the flagship. Tuning only the flagship and leaving baselines at library
defaults would manufacture the result: measured on a d=6 anisotropic mixture, a
val-tuned fixed-bandwidth KDE scores 16.69 against 33.92 for the library default
``gaussian_kde('scott')`` -- a factor of two. Any adaptive method would then
"beat the baseline by 50%" against a threshold that was never real.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar

import numpy as np

from ..core.errors import BackendUnavailableError, FitFailedError, ShapeMismatchError
from ..core.interfaces import check_log_density
from ..core.types import FloatArray


class _BackendMixin(abc.ABC):
    """Mixin providing a correct class-level lazy backend probe.

    Subclasses declare their backend by setting :attr:`_BACKEND` to an
    ``(module, attribute)`` pair. ``__init_subclass__`` resolves it once, at class
    creation, and stores the result in that subclass's own ``_RESOLVED`` slot.
    """

    #: ``(module_path, class_name)`` of the wrapped implementation.
    _BACKEND: ClassVar[tuple[str, str] | None] = None
    #: Populated per subclass by ``__init_subclass__``; never inherited.
    _RESOLVED: ClassVar[type | None] = None
    #: Populated per subclass by ``__init_subclass__``.
    _IMPORT_ERROR: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Resolve into THIS class's own namespace. Using setattr on cls (rather
        # than reading from the base) is what stops the MRO lookup problem.
        if cls._BACKEND is None:
            cls._RESOLVED = None
            cls._IMPORT_ERROR = None
            return
        module_path, attr = cls._BACKEND
        try:
            module = __import__(module_path, fromlist=[attr])
            cls._RESOLVED = getattr(module, attr)
            cls._IMPORT_ERROR = None
        except (ImportError, AttributeError) as exc:  # pragma: no cover - env dependent
            cls._RESOLVED = None
            cls._IMPORT_ERROR = f"{module_path}.{attr}: {exc}"

    @classmethod
    def available(cls) -> bool:
        """Whether the wrapped backend can be constructed here.

        Pure predicate: imports nothing extra and instantiates nothing.
        """
        return cls._RESOLVED is not None

    @classmethod
    def require(cls) -> type:
        """Return the backend class, or raise :class:`BackendUnavailableError`."""
        if cls._RESOLVED is None:
            detail = f" ({cls._IMPORT_ERROR})" if cls._IMPORT_ERROR else ""
            raise BackendUnavailableError(
                f"{cls.__name__} backend is unavailable{detail}; "
                "install scikit-learn/scipy or mark the row as skipped"
            )
        return cls._RESOLVED

    @classmethod
    def backend_name(cls) -> str:
        """Human-readable backend identifier, for report provenance."""
        return f"{cls._BACKEND[0]}.{cls._BACKEND[1]}" if cls._BACKEND else "none"


class BaseDensity(_BackendMixin):
    """Common scaffolding: fit bookkeeping plus the log-density contract."""

    def __init__(
        self,
        *,
        bandwidth: float = 1.0,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        if not bandwidth > 0.0:
            raise ValueError(f"bandwidth must be > 0, got {bandwidth}")
        self.bandwidth = float(bandwidth)
        self.random_state = int(random_state)
        self.extra_params = dict(kwargs)
        self._estimator: Any = None
        #: Arrays this model has been fitted on, for leakage assertion A3.
        self.fitted_on: FloatArray | None = None

    # -- helpers ---------------------------------------------------------
    def _attach(self, X: FloatArray) -> FloatArray:
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ShapeMismatchError(f"X must be 2-D (n, d), got shape {X.shape}")
        if X.shape[0] < 2:
            raise ShapeMismatchError(f"X needs at least 2 rows, got {X.shape[0]}")
        self.fitted_on = X
        self._n_features = int(X.shape[1])
        return X

    def _check_query(self, X: FloatArray) -> FloatArray:
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ShapeMismatchError(f"X must be 2-D (n, d), got shape {X.shape}")
        if X.shape[1] != self._n_features:
            raise ShapeMismatchError(
                f"X has {X.shape[1]} features but the model was fitted on {self._n_features}"
            )
        return X

    @property
    def fitted_fingerprints(self) -> frozenset[str]:
        """Content digests of everything this model was fitted on."""
        from ..core.types import array_fingerprint

        if self.fitted_on is None:
            return frozenset()
        return frozenset({array_fingerprint(self.fitted_on)})

    def params(self) -> dict[str, Any]:
        """Return the hyper-parameters, for the report's provenance column."""
        out = {"bandwidth": self.bandwidth, "random_state": self.random_state}
        out.update(self.extra_params)
        return out


class SkKernelDensity(BaseDensity):
    """Fixed-bandwidth Gaussian KDE via ``sklearn.neighbors.KernelDensity``.

    This is the **decisive baseline**: ``fixedkde(val-bw)`` is what the published
    thresholds are measured against, not the library default.

    ``atol``/``rtol`` are pinned to zero. They are early-stop thresholds for the
    kd-tree's branch pruning, so leaving them at their defaults would let
    floating-point comparison order decide a pruning decision and introduce
    ``~1e-3`` jitter into the NLL. A benchmark whose numbers wobble by a
    thousandth cannot distinguish two methods.
    """

    _BACKEND = ("sklearn.neighbors", "KernelDensity")

    def __init__(self, *, bandwidth: float = 1.0, random_state: int = 0) -> None:
        super().__init__(bandwidth=bandwidth, random_state=random_state)

    def fit(self, X: FloatArray) -> SkKernelDensity:
        X = self._attach(X)
        cls = self.require()
        try:
            self._estimator = cls(
                bandwidth=self.bandwidth,
                kernel="gaussian",
                algorithm="kd_tree",
                atol=0.0,
                rtol=0.0,
            ).fit(X)
        except Exception as exc:
            raise FitFailedError(f"SkKernelDensity.fit failed: {exc}") from exc
        return self

    def score_samples(self, X: FloatArray) -> FloatArray:
        if self._estimator is None:
            raise FitFailedError("SkKernelDensity is not fitted")
        X = self._check_query(X)
        return check_log_density("SkKernelDensity", self._estimator.score_samples(X))


class ScipyGaussianKDE(BaseDensity):
    """Single-bandwidth KDE via ``scipy.stats.gaussian_kde``.

    Two API traps, both live on this machine:

    * ``dataset`` takes the observations as **columns**, i.e. shape ``(d, n)``.
      Passing ``(n, d)`` either raises ``ValueError: Number of dimensions is
      greater than number of samples`` or, worse, silently produces a wrong
      answer.
    * ``bw_method`` changes meaning with its type: a string selects a rule
      (``'scott'``, ``'silverman'``), a float is a direct multiplier. Passing
      ``1.0`` where ``'scott'`` was meant gives the same multiplier Scott would
      have produced, not Scott's rule.

    The bandwidth is always recorded in the report so the two are never confused.
    """

    _BACKEND = ("scipy.stats", "gaussian_kde")

    def __init__(self, *, bw_method: str = "scott", random_state: int = 0) -> None:
        super().__init__(bandwidth=1.0, random_state=random_state)
        self.bw_method = str(bw_method)

    def fit(self, X: FloatArray) -> ScipyGaussianKDE:
        X = self._attach(X)
        cls = self.require()
        try:
            # (d, n): the transposed convention this function expects.
            self._estimator = cls(X.T, bw_method=self.bw_method)
        except Exception as exc:
            raise FitFailedError(f"ScipyGaussianKDE.fit failed: {exc}") from exc
        return self

    def score_samples(self, X: FloatArray) -> FloatArray:
        if self._estimator is None:
            raise FitFailedError("ScipyGaussianKDE is not fitted")
        X = self._check_query(X)
        return check_log_density("ScipyGaussianKDE", self._estimator.logpdf(X.T))

    def params(self) -> dict[str, Any]:
        return {"bw_method": self.bw_method, "factor": float(self._estimator.factor)}


class SkGaussianMixture(BaseDensity):
    """Gaussian mixture via ``sklearn.mixture.GaussianMixture``.

    A parametric counterpart to the kernel methods: ``K`` global components with
    shared-or-per-axis covariance, against the flagship's ``N`` local kernels.
    That difference is why the flagship can win structurally rather than by
    constant-factor tuning.

    ``n_components`` is keyword-only on scikit-learn 1.9. Passing it positionally
    raises ``TypeError: __init__() takes 1 positional argument but 2 positional
    arguments were given``.
    """

    _BACKEND = ("sklearn.mixture", "GaussianMixture")

    def __init__(
        self,
        *,
        n_components: int = 5,
        covariance_type: str = "full",
        n_init: int = 3,
        max_iter: int = 200,
        reg_covar: float = 1e-4,
        random_state: int = 0,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        super().__init__(bandwidth=1.0, random_state=random_state)
        self.n_components = int(n_components)
        self.covariance_type = str(covariance_type)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.reg_covar = float(reg_covar)

    def fit(self, X: FloatArray) -> SkGaussianMixture:
        X = self._attach(X)
        cls = self.require()
        try:
            self._estimator = cls(
                n_components=self.n_components,  # keyword-only on sklearn >= 1.9
                covariance_type=self.covariance_type,
                n_init=self.n_init,
                max_iter=self.max_iter,
                reg_covar=self.reg_covar,
                random_state=self.random_state,  # int, never a RandomState object
            ).fit(X)
        except Exception as exc:
            raise FitFailedError(f"SkGaussianMixture.fit failed: {exc}") from exc
        return self

    def score_samples(self, X: FloatArray) -> FloatArray:
        if self._estimator is None:
            raise FitFailedError("SkGaussianMixture is not fitted")
        X = self._check_query(X)
        return check_log_density("SkGaussianMixture", self._estimator.score_samples(X))

    def params(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "reg_covar": self.reg_covar,
            "random_state": self.random_state,
        }


class SkBayesianGaussianMixture(BaseDensity):
    """Variational Bayesian mixture via ``sklearn.mixture.BayesianGaussianMixture``.

    Retained as a robustness reference rather than as a contender: it introduces
    roughly seven prior hyper-parameters, converges slowly (500+ iterations is
    typical), and lets the priors dominate on small samples. It has never been the
    strongest baseline in calibration, and that is exactly why it is useful: it
    shows what happens when the mixture weights are inferred rather than fixed.

    ``n_components`` is keyword-only on scikit-learn 1.9 -- positional use raises
    ``TypeError`` (verified on this machine).
    """

    _BACKEND = ("sklearn.mixture", "BayesianGaussianMixture")

    def __init__(
        self,
        *,
        n_components: int = 5,
        covariance_type: str = "diag",
        n_init: int = 1,
        max_iter: int = 500,
        random_state: int = 0,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        super().__init__(bandwidth=1.0, random_state=random_state)
        self.n_components = int(n_components)
        self.covariance_type = str(covariance_type)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)

    def fit(self, X: FloatArray) -> SkBayesianGaussianMixture:
        X = self._attach(X)
        cls = self.require()
        try:
            self._estimator = cls(
                n_components=self.n_components,  # keyword-only
                covariance_type=self.covariance_type,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
            ).fit(X)
        except Exception as exc:
            raise FitFailedError(f"SkBayesianGaussianMixture.fit failed: {exc}") from exc
        return self

    def score_samples(self, X: FloatArray) -> FloatArray:
        if self._estimator is None:
            raise FitFailedError("SkBayesianGaussianMixture is not fitted")
        X = self._check_query(X)
        return check_log_density("SkBayesianGaussianMixture", self._estimator.score_samples(X))

    def converged(self) -> bool:
        """Whether EM reached its iteration limit.

        ``BayesianGaussianMixture`` silently returns an under-fitted solution when
        it hits ``max_iter``, which inflates the NLL for a reason that has nothing
        to do with the model. Reports must be able to distinguish "converged and
        bad" from "never converged".
        """
        if self._estimator is None:
            return False
        return bool(getattr(self._estimator, "converged_", False))

    def params(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "converged": self.converged(),
        }


#: Every density baseline, in reporting order.
DENSITY_BASELINES: tuple[type[BaseDensity], ...] = (
    SkKernelDensity,
    ScipyGaussianKDE,
    SkGaussianMixture,
    SkBayesianGaussianMixture,
)

__all__ = [
    "DENSITY_BASELINES",
    "BaseDensity",
    "ScipyGaussianKDE",
    "SkBayesianGaussianMixture",
    "SkGaussianMixture",
    "SkKernelDensity",
]
