"""L5 density -- baselines, Tier-1 fallbacks, and the flagship.

Author: 晨星 <CJX0712@users.noreply.github.com>
"""

from __future__ import annotations

from .baselines import (
    DENSITY_BASELINES,
    BaseDensity,
    ScipyGaussianKDE,
    SkBayesianGaussianMixture,
    SkGaussianMixture,
    SkKernelDensity,
)
from .flagship import DensFuse, FitState
from .tier1 import (
    NUMPY_FALLBACKS,
    NumpyDiffusionMaps,
    NumpyGMM,
    NumpyIsomap,
    NumpyKDE,
    NumpySpectral,
)

__all__ = [
    "DENSITY_BASELINES",
    "NUMPY_FALLBACKS",
    "BaseDensity",
    "DensFuse",
    "FitState",
    "NumpyDiffusionMaps",
    "NumpyGMM",
    "NumpyIsomap",
    "NumpyKDE",
    "NumpySpectral",
    "ScipyGaussianKDE",
    "SkBayesianGaussianMixture",
    "SkGaussianMixture",
    "SkKernelDensity",
]
