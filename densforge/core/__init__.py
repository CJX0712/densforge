"""L1 core -- the dependency sink. Imports nothing from any other DensForge layer.

Author: 晨星 <CJX0712@users.noreply.github.com>
"""

from __future__ import annotations

from .config import Config, DensFuseConfig
from .errors import (
    ArtifactNotFoundError,
    BackendUnavailableError,
    ConfigError,
    ConfigRangeError,
    ConfigUnknownKeyError,
    ConfigValidationError,
    DataError,
    DatasetNotFoundError,
    DensForgeError,
    EvalError,
    FitFailedError,
    InsufficientSamplesError,
    InsufficientSeedsError,
    LeakageError,
    LeakageModelNotTrainOnlyError,
    LeakageTestSetTouchedError,
    ModelError,
    NumericalError,
    NumericalOverflowError,
    SeedCollisionError,
    SerializationError,
    ShapeMismatchError,
)
from .interfaces import DensityEstimator, FusedEstimator, ManifoldEmbedder
from .seed import get_rng, set_all
from .types import Dataset, FloatArray, Report, Result, Split, SplitManifest

__all__ = [
    "ArtifactNotFoundError",
    "BackendUnavailableError",
    "Config",
    "ConfigError",
    "ConfigRangeError",
    "ConfigUnknownKeyError",
    "ConfigValidationError",
    "DataError",
    "Dataset",
    "DatasetNotFoundError",
    "DensForgeError",
    "DensFuseConfig",
    "DensityEstimator",
    "EvalError",
    "FitFailedError",
    "FloatArray",
    "FusedEstimator",
    "InsufficientSamplesError",
    "InsufficientSeedsError",
    "LeakageError",
    "LeakageModelNotTrainOnlyError",
    "LeakageTestSetTouchedError",
    "ManifoldEmbedder",
    "ModelError",
    "NumericalError",
    "NumericalOverflowError",
    "Report",
    "Result",
    "SeedCollisionError",
    "SerializationError",
    "ShapeMismatchError",
    "Split",
    "SplitManifest",
    "get_rng",
    "set_all",
]
