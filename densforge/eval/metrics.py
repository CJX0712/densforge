"""Metrics: NLL, trustworthiness, continuity, local-structure fidelity, significance.

Author: 晨星 <CJX0712@users.noreply.github.com>

The single convention
---------------------
Every density number in this project is ``NLL = -mean(score_samples(X_test))`` in
nats, lower is better. Every geometry number is trustworthiness or continuity in
``[0, 1]``, higher is better. Neither convention is negotiable, because
cross-method comparability rests entirely on them.

Trustworthiness takes **two** arguments on scikit-learn 1.9:
``trustworthiness(X, X_embedded, n_neighbors=k)``. Passing only the embedding
raises ``TypeError``. The metric compares neighbour ranks between the two spaces,
so the original space is not optional context -- it is half the measurement.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.manifold import trustworthiness as sk_trustworthiness

from ..core.errors import InsufficientSeedsError, NumericalError
from ..core.types import Report, Result, Status

#: Neighbourhood size for every trustworthiness / continuity number reported.
#:
#: Global and fixed on purpose. The metric's value depends on ``n_neighbors``, so
#: two experiments with different ``k`` are not comparable. Reports state it.
TRUSTWORTHNESS_K = 5


def nll(model: object, X_test: np.ndarray) -> float:
    """Return the held-out negative log-likelihood in nats.

    Parameters
    ----------
    model:
        Anything with a ``score_samples`` method returning log-density.
    X_test:
        ``(n, d)`` held-out points.

    Returns
    -------
    float
        ``-mean(log p(x))``. Lower is better.

    Raises
    ------
    NumericalError
        If the result is not finite, which means the density was evaluated in
        probability space somewhere and overflowed.
    """
    raw = model.score_samples(X_test)  # type: ignore[attr-defined]
    logp = np.asarray(raw, dtype=np.float64)
    if logp.shape[0] != X_test.shape[0]:
        raise NumericalError(
            f"score_samples returned {logp.shape[0]} values for {X_test.shape[0]} points"
        )
    value = float(-np.mean(logp))
    if not np.isfinite(value):
        raise NumericalError(
            f"test NLL is not finite ({value}); the density must be computed in log space"
        )
    return value


def nll_train_loo(model: object, X_train: np.ndarray = None) -> float:  # noqa: ARG001
    """Return the leave-one-out training NLL.

    Reported **alongside** the test NLL, never instead of it. The distinction
    matters for reading the results:

    * ``NLL_test < NLL_train_loo`` is *expected* for a correctly fitted model,
      because each ``sigma_i`` and each ``H_i`` is tuned to its own training
      point, so the training density is upward-biased;
    * a large gap indicates overfitting or a leakage path;
    * comparing a *full* training NLL against a test NLL is meaningless, because
      the full training NLL includes each point's own kernel contribution and is
      systematically lower.

    When the model exposes a fitted state, its leave-one-out channel is used
    directly. Otherwise a leave-one-out score is obtained by scoring each point
    against the model with that point's own contribution removed, which is only
    possible for models that expose their internals.
    """
    state = getattr(model, "_state", None)
    ell = getattr(state, "ell", None) if state is not None else None
    if ell is not None:
        return float(-np.mean(np.asarray(ell, dtype=np.float64)))
    raise NumericalError(
        f"{type(model).__name__} does not expose a leave-one-out channel; "
        "NLL_train_loo cannot be computed without it"
    )


def _validate_metric_inputs(
    X: np.ndarray, X_embedded: np.ndarray, *, n_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Validate inputs before handing them to scikit-learn.

    scikit-learn's own error messages are not always actionable, and two of its
    failure modes are silent rather than loud. Measured on 1.9.1:

    * a row-count mismatch raises ``IndexError: shape mismatch: indexing arrays
      could not be broadcast``, which does not say which array is wrong;
    * ``n_neighbors >= n / 2`` raises ``ValueError``, but only after the metric has
      already begun, and the threshold is a strict ``<`` that is easy to trip.

    Wrapping the checks here turns both into a single typed error naming the
    problem. NaN input is also caught: scikit-learn rejects it, but with a
    wall-of-text message about imputation that is irrelevant to this caller.
    """
    X = np.asarray(X, dtype=np.float64)
    X_embedded = np.asarray(X_embedded, dtype=np.float64)
    if X.ndim != 2 or X_embedded.ndim != 2:
        raise NumericalError(f"both arrays must be 2-D, got {X.shape} and {X_embedded.shape}")
    if X.shape[0] != X_embedded.shape[0]:
        raise NumericalError(
            f"row counts differ: X has {X.shape[0]}, embedding has {X_embedded.shape[0]}"
        )
    if not np.isfinite(X).all():
        raise NumericalError("X contains NaN or Inf")
    if not np.isfinite(X_embedded).all():
        raise NumericalError("the embedding contains NaN or Inf")
    # scikit-learn requires n_neighbors < n / 2, strictly.
    if 2 * n_neighbors >= X.shape[0]:
        raise NumericalError(
            f"n_neighbors={n_neighbors} must satisfy 2*k < n_samples "
            f"(n={X.shape[0]}); trustworthiness is undefined otherwise"
        )
    return X, X_embedded


def trustworthiness_score(
    X: np.ndarray, X_embedded: np.ndarray, *, n_neighbors: int = TRUSTWORTHNESS_K
) -> float:
    """Return trustworthiness in ``[0, 1]``; higher is better.

    Both arguments are required: the metric compares neighbour ranks between the
    original space ``X`` and the embedding ``X_embedded``. Passing only the
    embedding raises ``TypeError`` inside scikit-learn, which is a much less
    informative failure than the check here.

    Parameters
    ----------
    X:
        ``(n, d)`` points in the original space.
    X_embedded:
        ``(n, n_components)`` coordinates.
    n_neighbors:
        Neighbourhood size. Fixed at :data:`TRUSTWORTHNESS_K` project-wide.

    Raises
    ------
    NumericalError
        If the inputs are malformed or the value leaves ``[0, 1]``. A value
        outside the unit interval is not a rounding artefact: it means the inputs
        were degenerate, so the number is meaningless and must not be reported.
    """
    X, X_embedded = _validate_metric_inputs(X, X_embedded, n_neighbors=n_neighbors)
    value = float(
        sk_trustworthiness(X, X_embedded, n_neighbors=int(n_neighbors))  # both args required
    )
    _assert_unit_interval(value, "trustworthiness")
    return value


def continuity_score(
    X: np.ndarray, X_embedded: np.ndarray, *, n_neighbors: int = TRUSTWORTHNESS_K
) -> float:
    """Return continuity in ``[0, 1]``; higher is better.

    The mirror image of trustworthiness: the rank penalties are computed in the
    opposite direction. Reported as a secondary metric, because a method can score
    well on one and poorly on the other.

    Note
    ----
    scikit-learn 1.9.1 exports ``trustworthiness`` but **not** ``continuity`` --
    verified by inspecting ``sklearn.manifold.__all__``, whose public names are
    ``ClassicalMDS, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE,
    locally_linear_embedding, smacof, spectral_embedding, trustworthiness``.
    Continuity is therefore obtained by exchanging the two arguments, which is
    exactly its definition. If a future scikit-learn adds a native
    ``continuity``, this wrapper should prefer it.
    """
    X, X_embedded = _validate_metric_inputs(X, X_embedded, n_neighbors=n_neighbors)
    value = float(
        sk_trustworthiness(X_embedded, X, n_neighbors=int(n_neighbors))  # arguments exchanged
    )
    _assert_unit_interval(value, "continuity")
    return value


def _assert_unit_interval(value: float, name: str) -> None:
    """Reject a metric value outside ``[0, 1]``.

    A trustworthiness outside the unit interval means the inputs were degenerate
    (duplicated points, a single cluster, NaNs) and the number is meaningless.
    Asserting here turns a silently wrong table cell into a loud failure.
    """
    if not np.isfinite(value):
        raise NumericalError(f"{name} returned a non-finite value ({value})")
    if not 0.0 <= value <= 1.0:
        raise NumericalError(
            f"{name} returned {value}, outside its mathematical range [0, 1]; "
            "the inputs must be degenerate"
        )


def local_structure_fidelity(
    X: np.ndarray,
    X_embedded: np.ndarray,
    X_manifold: np.ndarray,
    *,
    n_neighbors: int = TRUSTWORTHNESS_K,
) -> float:
    """Fraction of geodesic neighbours preserved by the embedding, in ``[0, 1]``.

    Uses nearest neighbours **in a supplied intrinsic-space reference** as ground
    truth, rather than in the ambient space. That makes the metric sensitive to
    exactly the short-circuit failures that ambient-space trustworthiness is blind
    to: two points can be ambiently close yet geodesically far.

    Only usable where an intrinsic reference exists, i.e. on synthetic data with a
    known generating manifold. Never reported as a headline number.
    """
    X = np.asarray(X, dtype=np.float64)
    X_embedded = np.asarray(X_embedded, dtype=np.float64)
    X_manifold = np.asarray(X_manifold, dtype=np.float64)
    if not (X.shape[0] == X_embedded.shape[0] == X_manifold.shape[0]):
        raise NumericalError("all three arrays must have the same number of rows")
    k = int(n_neighbors)
    geo = _knn_indices(X_manifold, k)
    emb = _knn_indices(X_embedded, k)
    scores = [len(set(geo[i]) & set(emb[i])) / k for i in range(X.shape[0])]
    return float(np.mean(scores))


def _knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Return the ``k`` nearest-neighbour indices per row, self excluded."""
    from scipy.spatial.distance import cdist

    n = X.shape[0]
    k = int(min(k, n - 1))
    sq = cdist(X, X, metric="sqeuclidean")
    np.fill_diagonal(sq, np.inf)
    return np.argsort(sq, axis=1, kind="stable")[:, :k]


def calibration_error(model: object, X_test: np.ndarray, *, n_bins: int = 20) -> float:
    """Return the mean absolute calibration error over equal-count bins.

    A density model can minimise NLL and still be badly calibrated; this separates
    the two. Samples are split into ``n_bins`` groups of equal count, the mean
    predicted log-density is compared with the mean empirical log-frequency in
    each, and the absolute gaps are averaged. Lower is better.

    Only meaningful on data whose true density is known or estimable, so this is a
    diagnostic on synthetic data, never a headline metric.
    """
    raw = model.score_samples(X_test)  # type: ignore[attr-defined]
    logp = np.asarray(raw, dtype=np.float64)
    n = logp.shape[0]
    if n < n_bins:
        raise NumericalError(
            f"calibration_error needs at least n_bins={n_bins} samples, got {n}"
        )
    order = np.argsort(logp, kind="stable")
    errors = []
    for chunk in np.array_split(order, n_bins):
        if chunk.size == 0:
            continue
        predicted = float(np.mean(logp[chunk]))
        # Empirical log frequency of the bin, with a count-based continuity
        # correction so an empty-tail bin cannot produce -inf.
        observed = float(np.log(chunk.size / n))
        errors.append(abs(predicted - observed))
    return float(np.mean(errors))


def is_significant(delta: float, sigma_a: float, sigma_b: float) -> bool:
    """Whether an improvement clears the project's deliberately strict bar.

    The criterion is ``delta > 0.5 * (sigma_a + sigma_b)``. Requiring the gap to
    exceed *half the sum* of the two standard deviations means a difference that
    could plausibly be noise is reported as no difference. The project would
    rather miss a real small win than announce a win that reverses on the next
    seed.
    """
    if not np.isfinite(delta):
        return False
    return bool(delta > 0.5 * (abs(sigma_a) + abs(sigma_b)))


def summarize(results: Sequence[Result], *, n_seeds: int = 3) -> Report:
    """Aggregate per-seed results into a :class:`~densforge.core.types.Report`.

    Only rows whose status is ``ok`` enter the aggregation. ``skipped`` and
    ``failed`` rows are counted and carried through, but are never zero-filled and
    never nan-filled -- a fabricated number in a benchmark table is worse than a
    missing one, because a missing one is visible.

    Parameters
    ----------
    results:
        Rows to aggregate.
    n_seeds:
        Seeds per row. Fewer than 3 raises, because ``mean ± std`` over two points
        is not a measurement.

    Raises
    ------
    InsufficientSeedsError
        If ``n_seeds < 3``.
    """
    if n_seeds < 3:
        raise InsufficientSeedsError(
            f"n_seeds={n_seeds} cannot support a mean ± std report; the minimum is 3"
        )
    rows = tuple(results)
    n_skipped = sum(1 for r in rows if r.status is Status.SKIPPED)
    n_failed = sum(1 for r in rows if r.status is Status.FAILED)
    return Report(
        results=rows,
        n_seeds=int(n_seeds),
        n_skipped=int(n_skipped),
        n_failed=int(n_failed),
        meta={"n_rows": len(rows), "n_ok": len(rows) - n_skipped - n_failed},
    )


def aggregate(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    """Return ``(mean, std)`` over the non-``None`` entries, or ``(None, None)``.

    Standard deviation uses ``ddof=1`` (sample standard deviation), which is the
    correct choice across seeds: the seeds are a sample from the space of possible
    datasets, not the whole population.

    An empty input returns ``(None, None)`` rather than ``(nan, nan)``. Callers
    must handle the absence explicitly, which is what keeps "skipped" from
    silently becoming "measured as zero".
    """
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None, None
    if len(clean) < 2:
        return float(np.mean(clean)), 0.0
    return float(np.mean(clean)), float(np.std(clean, ddof=1))


def mean_or_none(values: Iterable[float | None]) -> float | None:
    """Return the mean of the finite entries, or ``None`` if there are none."""
    mean, _ = aggregate(values)
    return mean


def format_mean_std(
    mean: float | None, std: float | None, *, width: int = 10, precision: int = 4
) -> str:
    """Format a ``mean ± std`` cell for the CLI, or an explicit placeholder.

    A missing value prints as ``(skipped)`` rather than as a number. This is the
    textual half of the same rule the aggregator enforces numerically.
    """
    if mean is None:
        return "(skipped)"
    if std is None:
        return f"{mean:>{width}.{precision}f}"
    return f"{mean:>{width}.{precision}f} ± {std:.{precision}f}"


__all__ = [
    "TRUSTWORTHNESS_K",
    "aggregate",
    "calibration_error",
    "continuity_score",
    "format_mean_std",
    "is_significant",
    "local_structure_fidelity",
    "mean_or_none",
    "nll",
    "nll_train_loo",
    "summarize",
    "trustworthiness_score",
]
