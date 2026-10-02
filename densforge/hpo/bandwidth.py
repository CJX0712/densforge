"""Fixed-bandwidth selection on the validation set.

Author: 晨星 <CJX0712@users.noreply.github.com>

This is the fair-play module. The published thresholds are measured against
``fixedkde(val-bw)``: a ``KernelDensity`` whose bandwidth was chosen on validation
data. Not the library default.

The difference is not cosmetic. Measured on a d=6 multi-scale anisotropic mixture,
``gaussian_kde('scott')`` scores 33.92 while a val-tuned fixed bandwidth scores
16.69 -- a factor of two. A benchmark that compares against the library default
hands every adaptive method a free 50% and calls the result a threshold. No
method in this project is tuned less than any other; see
:func:`densforge.hpo.coordinate.coordinate_descent` for the flagship's budget
accounting.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..core.errors import NumericalError
from ..core.types import FloatArray


def select_bandwidth(
    factory,
    X: FloatArray,
    X_val: FloatArray,
    grid: Sequence[float],
) -> tuple[float, float]:
    """Choose a bandwidth by validation NLL.

    Parameters
    ----------
    factory:
        Callable taking ``bandwidth`` and returning an unfitted estimator.
    X, X_val:
        Training and validation samples. **Never** pass test data here: selecting
        on test makes the reported metric a training metric.
    grid:
        Candidate bandwidths, all positive.

    Returns
    -------
    tuple[float, float]
        The winning bandwidth and its validation NLL. The NLL is returned so the
        caller can report the selection cost without refitting.

    Raises
    ------
    NumericalError
        If the grid is empty or contains a non-positive value.

    Notes
    -----
    The grid should be log-spaced. The optimal bandwidth scales as
    ``N^-1/(d+4)``, so a linear grid either wastes resolution at the top or misses
    the optimum at the bottom.
    """
    candidates = [float(h) for h in grid]
    if not candidates:
        raise NumericalError("bandwidth grid must not be empty")
    if any(h <= 0.0 for h in candidates):
        raise NumericalError(f"bandwidths must be positive, got {candidates}")

    best_h = candidates[0]
    best_nll = np.inf
    for h in candidates:
        model = factory(bandwidth=h)
        model.fit(X)
        value = float(-np.mean(model.score_samples(X_val)))
        if np.isfinite(value) and value < best_nll:
            best_nll = value
            best_h = h
    if not np.isfinite(best_nll):
        raise NumericalError(
            f"every bandwidth in {candidates} produced a non-finite validation NLL; "
            "check the data scale"
        )
    return best_h, best_nll


def log_spaced_grid(
    low: float = 0.1, high: float = 10.0, n_points: int = 8
) -> tuple[float, ...]:
    """Return a log-spaced bandwidth grid.

    Log spacing because the optimum scales as a power of ``N``: a linear grid
    wastes almost all its resolution far from the optimum.
    """
    if low <= 0.0 or high <= 0.0:
        raise NumericalError("log_spaced_grid bounds must be positive")
    if n_points < 2:
        raise NumericalError(f"n_points must be >= 2, got {n_points}")
    return tuple(float(v) for v in np.exp(np.linspace(np.log(low), np.log(high), n_points)))


__all__ = ["log_spaced_grid", "select_bandwidth"]
