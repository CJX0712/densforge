"""Staged coordinate descent over the flagship's hyper-parameters.

Author: 晨星 <CJX0712@users.noreply.github.com>

Why coordinate descent
----------------------
The full grid is ``k(5) x gamma(6) x beta(3) x tau(4) x eta(3) x m(4) x alpha(5)
x rounds(3) = 43,200`` evaluations. At roughly 0.1 s per evaluation that is over
an hour per dataset per seed, which no CI budget survives. Coordinate descent
visits the axes in sequence, and the order is not arbitrary:

===================  =========================================================
axis                 why it is searched at this position
===================  =========================================================
``gamma``            dominates the bias/variance trade-off; every other axis
                     interacts with it, so it must be settled first
``beta``             the largest structural gain; shape rather than scale
``k``                determines the statistical quality of ``rho`` and of the
                     local covariance, which both other axes read
``alpha``            manifold-side; interacts with the graph, not the density
``m_score``          pure speed/bias trade-off at scoring time
``tau``, ``eta``     second-order numerical guards
``n_fuse_rounds``    last: it is the most expensive and least sensitive
===================  =========================================================

Budget accounting (invariant **I32**)
--------------------------------------
The number of validation evaluations is returned and logged, never inferred. A
tuning budget that cannot be audited is indistinguishable from no tuning at all,
and an un-tuned flagship that happens to win looks identical to a tuned one. The
comparison against the baselines' combined budget is explicit in
:func:`baseline_budget` so a report can state the ratio rather than assert fairness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..core.config import HPO_ORDER, DensFuseConfig
from ..core.errors import NumericalError
from ..core.types import FloatArray


@dataclass(frozen=True, slots=True)
class TuningTrace:
    """One audit record per validation evaluation."""

    axis: str
    value: object
    val_nll: float

    def as_dict(self) -> dict[str, object]:
        return {"axis": self.axis, "value": self.value, "val_nll": round(self.val_nll, 8)}


@dataclass(frozen=True, slots=True)
class TuningResult:
    """The outcome of a tuning run: a config, a budget count, and the evidence.

    The trace is returned as its own object rather than stashed on the config.
    ``DensFuseConfig`` is a ``slots=True`` frozen dataclass, so it cannot carry
    extra attributes at all -- and even if it could, a budget that lives on the
    object it describes can be dropped by a refactor without any test noticing.
    A first-class return value cannot.
    """

    config: DensFuseConfig
    n_eval: int
    trace: tuple[TuningTrace, ...]
    budget: int

    @property
    def exhausted(self) -> bool:
        """Whether the search hit its budget before covering every axis.

        A truncated search and a complete one select different configurations, so
        this flag belongs in the report next to the config.
        """
        return self.n_eval >= self.budget

    def as_params(self) -> dict[str, object]:
        """Provenance for a benchmark row."""
        return {
            "n_eval": self.n_eval,
            "budget": self.budget,
            "exhausted": self.exhausted,
            "trace": [t.as_dict() for t in self.trace],
        }


def default_grid(
    axis: str,
    config: DensFuseConfig,  # noqa: ARG001 -- kept for signature symmetry
    d: int,
) -> tuple[object, ...]:
    """Return the candidate values for one axis.

    ``eta``'s grid depends on the ambient dimension, so it is computed from ``d``.
    """
    if axis == "gamma":
        # Extended below the design document's 0.5 floor. The Lazaridis pilot is a
        # mean-shift bandwidth, calibrated to make the density gradient have a
        # well-defined mode, so it is systematically too wide for NLL minimisation:
        # measured on aniso_gmm (d=8, n=400), sigma0 = 10.79 against a
        # validation-optimal fixed bandwidth of 2.5, so the correcting factor is
        # gamma ~ 0.23 and a grid starting at 0.5 cannot reach the optimum at all.
        # See docs/math_verification.md, section B3.
        return (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
    if axis == "beta":
        return (0.0, 0.5, 1.0)
    if axis == "k":
        return (8, 12, 15, 20, 30)
    if axis == "alpha":
        return (0.0, 0.25, 0.5, 0.75, 1.0)
    if axis == "m_score":
        return (256, 384, 512)
    if axis == "tau":
        return (0.05, 0.1, 0.2, 0.3)
    if axis == "eta":
        return (1.0 / (d + 4.0), 1.0 / d, 1.0 / (2.0 * d))
    if axis == "n_fuse_rounds":
        return (1, 2, 3)
    raise NumericalError(f"no grid defined for axis {axis!r}")


def coordinate_descent(
    state: object,
    X_val: FloatArray,
    config: DensFuseConfig,
    *,
    budget: int = 120,
    estimator: object | None = None,
    grids: dict[str, Sequence[object]] | None = None,
) -> TuningResult:
    """Tune the flagship by staged coordinate descent on validation NLL.

    Parameters
    ----------
    state:
        The fitted state. Used to rebuild a model for each trial; the state itself
        is never mutated, which is what invariant **I23** requires.
    X_val:
        Validation samples. The only data this function may see.
    config:
        Starting configuration.
    budget:
        Maximum number of validation evaluations. Reaching it is not an error; it is
        reported via :attr:`TuningResult.exhausted`, because a truncated search and
        a complete one select different configurations.
    estimator:
        The :class:`~densforge.density.flagship.DensFuse` instance that produced
        ``state``. Required, since each trial needs a fresh fit.
    grids:
        Optional per-axis grid overrides, for tests and for ablations.

    Returns
    -------
    TuningResult
        The selected config, the evaluations spent, and the full trace.

    Raises
    ------
    NumericalError
        If ``estimator`` is missing, or if no axis had more than one admissible
        value so nothing could be evaluated.
    """
    if estimator is None:
        raise NumericalError("coordinate_descent needs the estimator that produced `state`")
    X_val = np.ascontiguousarray(np.asarray(X_val, dtype=np.float64))
    d = int(state.d)
    n_train = int(state.n_samples)

    current = config
    trace: list[TuningTrace] = []
    n_eval = 0

    for axis in HPO_ORDER:
        values = grids.get(axis) if grids else None
        if values is None:
            values = default_grid(axis, current, d)
        # Values outside the state's own limits cannot be evaluated: `k` must not
        # exceed n_train - 1, and m_score cannot exceed n_train.
        values = tuple(v for v in values if _is_admissible(axis, v, n_train))
        if len(values) <= 1:
            continue

        best_value = getattr(current, axis)
        best_nll = np.inf
        for value in values:
            if n_eval >= budget:
                break
            trial = current.evolve(**{axis: _cast(axis, value)})
            model = _refit(estimator, trial)
            value_nll = _val_nll(model, X_val, n_train)
            n_eval += 1
            trace.append(TuningTrace(axis, value, value_nll))
            if np.isfinite(value_nll) and value_nll < best_nll:
                best_nll = value_nll
                best_value = value
        if np.isfinite(best_nll):
            current = current.evolve(**{axis: _cast(axis, best_value)})
        if n_eval >= budget:
            break

    if not trace:
        raise NumericalError(
            "coordinate descent performed no evaluations; every grid had a single "
            "admissible value"
        )
    return TuningResult(config=current, n_eval=n_eval, trace=tuple(trace), budget=int(budget))


def _is_admissible(axis: str, value: object, n_train: int) -> bool:
    """Whether a candidate value can actually be evaluated on this dataset."""
    if axis == "k":
        return isinstance(value, int) and 2 <= value <= n_train - 1
    if axis == "m_score":
        return isinstance(value, int) and 1 <= value <= n_train
    if axis == "n_fuse_rounds":
        return isinstance(value, int) and value >= 1
    if axis == "eta":
        return isinstance(value, float) and value > 0.0
    return True


def _cast(axis: str, value: object) -> object:
    """Coerce a grid value to the type the config field expects."""
    if axis in {"k", "m_score", "n_fuse_rounds"}:
        return int(value)  # type: ignore[arg-type]
    return float(value)  # type: ignore[arg-type]


def _refit(estimator: object, config: DensFuseConfig) -> object:
    """Return a fresh estimator fitted with ``config`` on the same training data."""
    from ..density.flagship import DensFuse

    state = estimator._state
    assert state is not None
    trial = DensFuse(config)
    trial._state = trial._fit_state(state.X_train, config)
    return trial


def _val_nll(model: object, X_val: FloatArray, n_train: int) -> float:
    """Validation NLL, with a cheap scoring cap so tuning stays affordable.

    The cap is on *scoring neighbours*, not on the number of validation points. It
    is recorded in the report, because a tuning run that scored with ``m = 384``
    and one that scored with ``m = 64`` selected different bandwidths for reasons
    that have nothing to do with the hyper-parameter under test.
    """
    m = int(min(model.config.m_score, n_train))
    logp = model.score_samples(X_val, m_score=m)  # type: ignore[attr-defined]
    return float(-np.mean(logp))


def baseline_budget(bandwidth_grid: Sequence[float], mixture_components: Sequence[int]) -> int:
    """Total validation evaluations spent tuning the baselines.

    Published alongside the flagship's own count so a report can state the ratio
    as a measured fact rather than claiming parity.
    """
    return len(bandwidth_grid) + len(mixture_components) * 2


__all__ = [
    "TuningResult",
    "TuningTrace",
    "baseline_budget",
    "coordinate_descent",
    "default_grid",
]
