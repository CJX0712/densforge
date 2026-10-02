"""L5 manifold -- baselines, the density-balanced diffusion operator, Tier-1 fallbacks.

Author: 晨星 <CJX0712@users.noreply.github.com>
"""

from __future__ import annotations

from .baselines import (
    MANIFOLD_BASELINES,
    TRUSTWORTHNESS_K,
    BaseManifold,
    SkIsomap,
    SkLLE,
    SkMDS,
    SkPCA,
    SkSpectralEmbedding,
    SkTSNE,
)
from .diffusionmaps import (
    centre_and_clip_ell,
    density_balance,
    diffusion_coordinates,
    loo_log_density,
    symmetric_normalize,
)
from .flagship import ManifoldFuse
from .tier1 import NumpyDiffusionMaps, NumpyIsomap, NumpySpectral

__all__ = [
    "MANIFOLD_BASELINES",
    "TRUSTWORTHNESS_K",
    "BaseManifold",
    "ManifoldFuse",
    "NumpyDiffusionMaps",
    "NumpyIsomap",
    "NumpySpectral",
    "SkIsomap",
    "SkLLE",
    "SkMDS",
    "SkPCA",
    "SkSpectralEmbedding",
    "SkTSNE",
    "centre_and_clip_ell",
    "density_balance",
    "diffusion_coordinates",
    "loo_log_density",
    "symmetric_normalize",
]
