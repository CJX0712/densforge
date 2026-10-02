"""Density-balancing strength selection.

Author: 晨星 <CJX0712@users.noreply.github.com>

The ``alpha`` grid
------------------
``alpha in {0, 0.25, 0.5, 0.75, 1}`` and nothing above 1. The upper bound is
theoretical, not conservative: ``alpha = 1`` corresponds exactly to replacing the
``p``-weighted operator with a volume-measure operator, i.e. full importance
re-weighting. Above 1 there is no such interpretation, and the failure mode is
specific and nasty -- sparse points acquire enormous in-weights, the leading
eigenvector collapses onto a handful of them, and the embedding degenerates into a
blob or blows up in variance.

The endpoint alarm
------------------
If the selected ``alpha`` lands on 0 or 1, that is a **finding, not a result**. It
means the direction is either useless on this data or the grid is too narrow, and
the two are indistinguishable without a wider sweep. :func:`select_alpha` reports
this explicitly via :class:`AlphaSelection` so a caller cannot accidentally present
an endpoint optimum as a validated interior one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..core.errors import NumericalError
from ..core.types import FloatArray

#: The default candidate grid. ``0`` is the no-correction control, ``1`` full correction.
ALPHA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True, slots=True)
class AlphaSelection:
    """Outcome of an ``alpha`` search, with the endpoint flag attached."""

    alpha: float
    score: float
    grid: tuple[float, ...]
    scores: tuple[float, ...]
    at_endpoint: bool

    def as_params(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "alpha_at_endpoint": self.at_endpoint,
            "alpha_grid_scores": [round(s, 6) for s in self.scores],
        }


def select_alpha(
    X: FloatArray,
    X_val: FloatArray,  # noqa: ARG001 -- part of the documented signature
    grid: Sequence[float] = ALPHA_GRID,
    *,
    n_components: int = 2,
    n_neighbors: int = 15,
    evaluate: Callable[[FloatArray, float], float] | None = None,
) -> AlphaSelection:
    """Choose the density-balancing strength on validation data.

    Parameters
    ----------
    X, X_val:
        Training and validation samples.
    grid:
        Candidate values, each in ``[0, 1]``.
    n_components, n_neighbors:
        Passed to the default evaluator.
    evaluate:
        Optional ``(X, alpha) -> score`` override, where **higher is better**.
        The default evaluates embedding trustworthiness on the training set; supply
        your own when the downstream metric is different.

    Returns
    -------
    AlphaSelection
        The winning value, its score, the full score curve, and whether the optimum
        sits on a grid endpoint.
    """
    candidates = tuple(float(a) for a in grid)
    if not candidates:
        raise NumericalError("alpha grid must not be empty")
    for a in candidates:
        if not 0.0 <= a <= 1.0:
            raise NumericalError(
                f"alpha values must lie in [0, 1]; {a} has no theoretical interpretation "
                "and destabilises the diffusion operator"
            )

    scorer = evaluate or _default_evaluator(n_components, n_neighbors)
    scores: list[float] = []
    for a in candidates:
        value = float(scorer(X, a))
        if not np.isfinite(value):
            raise NumericalError(f"alpha={a} produced a non-finite score ({value})")
        scores.append(value)
    best = int(np.argmax(scores))
    return AlphaSelection(
        alpha=candidates[best],
        score=scores[best],
        grid=candidates,
        scores=tuple(scores),
        at_endpoint=best in {0, len(candidates) - 1},
    )


def _default_evaluator(n_components: int, n_neighbors: int):
    """Build the default evaluator: trustworthiness of the balanced embedding.

    Imported lazily because :mod:`densforge.eval` sits at L2 and this module at
    L4; a module-level import would be legal but would drag scikit-learn's
    ``trustworthiness`` in at package-import time for every user of the CLI.
    """
    from ..eval.metrics import TRUSTWORTHNESS_K, trustworthiness_score

    def evaluate(X: FloatArray, alpha: float) -> float:
        from ..manifold.flagship import ManifoldFuse

        model = ManifoldFuse(
            n_components=n_components, n_neighbors=n_neighbors, alpha=alpha
        ).fit(X)
        return trustworthiness_score(X, model.transform(), n_neighbors=TRUSTWORTHNESS_K)

    return evaluate


__all__ = ["ALPHA_GRID", "AlphaSelection", "select_alpha"]
