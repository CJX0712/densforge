"""Graded exception hierarchy with stable error codes (architecture §7.2).

Author: 晨星 <CJX0712@users.noreply.github.com>

Every error carries a ``code`` string of the form ``E<segment><specific>``. The
code is part of the public contract: tests assert on codes, not on message text,
so message wording can improve without breaking callers.
"""

from __future__ import annotations


class DensForgeError(Exception):
    """Base class for every error DensForge raises deliberately.

    Parameters
    ----------
    message:
        Human readable description.
    code:
        Stable identifier. Defaults to the class-level ``default_code``.
    """

    default_code: str = "E000"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        self.code = code or self.default_code
        self.message = message
        super().__init__(f"{type(self).__name__}[{self.code}]: {message}")


# --------------------------------------------------------------------------- E1xx config
class ConfigError(DensForgeError):
    """Generic configuration problem."""

    default_code = "E100"


class ConfigValidationError(ConfigError):
    """Missing field or wrong type."""

    default_code = "E101"


class ConfigRangeError(ConfigError):
    """Numeric value outside its documented range."""

    default_code = "E102"


class ConfigUnknownKeyError(ConfigError):
    """An unrecognised ``ENV_DENSFORGE_*`` key.

    Raised instead of ignoring the variable: a typo that silently does nothing is
    a classic production incident (architecture §7.1).
    """

    default_code = "E103"


# ---------------------------------------------------------------------------- E2xx data
class DataError(DensForgeError):
    """Generic data problem."""

    default_code = "E200"


class DatasetNotFoundError(DataError):
    """The dataset registry has no entry under that name."""

    default_code = "E201"


class ShapeMismatchError(DataError):
    """``X.ndim != 2`` or train/test dimensionality disagrees."""

    default_code = "E202"


class InsufficientSamplesError(DataError):
    """``n < 2`` or ``n_neighbors >= n``."""

    default_code = "E203"


class SeedCollisionError(DataError):
    """train and test seeds coincide -- a leak; refuse to run."""

    default_code = "E204"


# ------------------------------------------------------------------------- E3xx model
class ModelError(DensForgeError):
    """Generic model problem."""

    default_code = "E300"


class BackendUnavailableError(ModelError):
    """``available()`` returned False but the caller forced construction."""

    default_code = "E301"


class FitFailedError(ModelError):
    """Fitting raised or failed to converge."""

    default_code = "E302"


class NumericalError(ModelError):
    """NaN / Inf / log underflow."""

    default_code = "E303"


class NumericalOverflowError(NumericalError):
    """Density computation overflowed.

    The gatekeeper for architecture §2.2: a density model that returns
    ``exp(log p)`` instead of ``log p`` overflows silently and turns every NLL
    into ``nan`` *without raising*. This error exists so that path is loud.
    """

    default_code = "E304"


# ------------------------------------------------------------------------- E4xx eval
class EvalError(DensForgeError):
    """Generic evaluation problem."""

    default_code = "E400"


class LeakageError(EvalError):
    """Seeds are not isolated (assertion A1)."""

    default_code = "E401"


class LeakageTestSetTouchedError(LeakageError):
    """The HPO phase touched the test array (assertion A2)."""

    default_code = "E402"


class LeakageModelNotTrainOnlyError(LeakageError):
    """Reporting a test metric for a model that has seen the test array (A3)."""

    default_code = "E403"


class InsufficientSeedsError(EvalError):
    """Fewer than 3 seeds, yet ``mean ± std`` was requested."""

    default_code = "E404"


# ---------------------------------------------------------------------------- E5xx IO
class IOErrorBase(DensForgeError):
    """Generic IO problem."""

    default_code = "E500"


class ArtifactNotFoundError(IOErrorBase):
    """``benchmark.json`` is missing."""

    default_code = "E501"


class SerializationError(IOErrorBase):
    """JSON write/read failed."""

    default_code = "E502"


class AtomicWriteError(IOErrorBase):
    """A write died part way through.

    Artifacts are written to a temporary file and then renamed, so a crash can
    never leave a half-written ``benchmark.json`` behind.
    """

    default_code = "E503"


__all__ = [
    "ArtifactNotFoundError",
    "AtomicWriteError",
    "BackendUnavailableError",
    "ConfigError",
    "ConfigRangeError",
    "ConfigUnknownKeyError",
    "ConfigValidationError",
    "DataError",
    "DatasetNotFoundError",
    "DensForgeError",
    "EvalError",
    "FitFailedError",
    "IOErrorBase",
    "InsufficientSamplesError",
    "InsufficientSeedsError",
    "LeakageError",
    "LeakageModelNotTrainOnlyError",
    "LeakageTestSetTouchedError",
    "ModelError",
    "NumericalError",
    "NumericalOverflowError",
    "SeedCollisionError",
    "SerializationError",
    "ShapeMismatchError",
]
