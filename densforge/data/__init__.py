"""L2 data -- synthetic generators and the leakage-proof split protocol.

Author: 晨星 <CJX0712@users.noreply.github.com>
"""

from __future__ import annotations

from .datasets import (
    CONTROL_DATASET,
    HEADLINE_DATASETS,
    REGISTRY,
    DatasetSpec,
    available_datasets,
    build_split,
    get_spec,
    min_train_test_distance,
)
from .synth import (
    make_aniso_gmm,
    make_circles,
    make_double_spiral,
    make_heavy_tail,
    make_hetero_density,
    make_iso_gauss,
    make_manifold_noise,
    make_sparse_dim,
    make_swiss_roll,
    make_t_mixture,
    make_uniform_hypercube,
)

__all__ = [
    "CONTROL_DATASET",
    "HEADLINE_DATASETS",
    "REGISTRY",
    "DatasetSpec",
    "available_datasets",
    "build_split",
    "get_spec",
    "make_aniso_gmm",
    "make_circles",
    "make_double_spiral",
    "make_heavy_tail",
    "make_hetero_density",
    "make_iso_gauss",
    "make_manifold_noise",
    "make_sparse_dim",
    "make_swiss_roll",
    "make_t_mixture",
    "make_uniform_hypercube",
    "min_train_test_distance",
]
