"""L3 training -- fit orchestration and train-only provenance.

Author: 晨星 <CJX0712@users.noreply.github.com>
"""

from __future__ import annotations

from .fitter import FittedModel, TrainOnlyModel, fit_density, fit_embedder

__all__ = ["FittedModel", "TrainOnlyModel", "fit_density", "fit_embedder"]
