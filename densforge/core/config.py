"""Immutable configuration objects with environment overrides.

Author: 晨星 <CJX0712@users.noreply.github.com>

Two configuration objects live here:

``Config``
    Run-level knobs (how many samples, how many seeds, which backends). Overridable
    from the environment via ``ENV_DENSFORGE_*`` variables.

``DensFuseConfig``
    The flagship model's hyper-parameters. Frozen, so that coordinate descent
    produces a new object per trial instead of mutating shared state -- which is
    what makes the "select() does not mutate the fit state" invariant (I23)
    checkable.

Override precedence: explicit argument > ``ENV_DENSFORGE_*`` > dataclass default.
An unrecognised ``ENV_DENSFORGE_*`` key raises rather than being ignored, because
a silently-ignored typo in a deployment variable is a classic incident.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np

from .errors import ConfigError, ConfigRangeError, ConfigUnknownKeyError, ConfigValidationError

#: Prefix for environment overrides.
ENV_PREFIX = "ENV_DENSFORGE_"

#: Upper bound on ambient dimensionality.
#:
#: Beyond this the O(n^3) eigendecomposition and the d^2-scaled neighbourhood
#: gathers stop being affordable, and published thresholds no longer apply
#: (architecture §7.1).
MAX_D = 16

#: Fixed-bandwidth grid for the KDE baseline: 61 log-spaced points over three
#: decades, ``np.logspace(-2, 3, 61)``.
#:
#: This is the project's most consequential measurement decision, so the reasoning
#: belongs with the constant rather than in a comment far from its use.
#:
#: The bandwidth minimising held-out NLL depends on the data's scale, and the six
#: datasets do not share one. Measured on this machine: the flagship's Lazaridis
#: pilot gives ``sigma0 = 10.79`` on ``aniso_gmm`` (d=8) against a
#: validation-optimal *fixed* bandwidth of 2.5, and the optimum on ``circles``
#: (d=2) is 3.0. A 5-point grid ``(0.25, 0.5, 1, 2, 4)`` contains no value near
#: either, so the baseline gets scored at a bandwidth its own validation data would
#: never choose.
#:
#: That is not conservative, it is *unfair*, and it flatters whatever it is compared
#: against. Measured consequence:
#:
#: ==========================  ==============  ===============  ================
#: baseline bandwidth grid      median vs base  worst vs base    flagship NLL
#: ==========================  ==============  ===============  ================
#: 5-point ``(0.25 … 4)``      +33.7%          --               as measured then
#: 5-point, re-measured         +52.84%         --               as measured then
#: **61-point log grid**        **+1.01%**      **-3.20%**      current
#: ==========================  ==============  ===============  ================
#:
#: Two of the three earlier figures were artefacts of an under-tuned baseline, and
#: the 61-point grid is what exposed that. The current numbers are the honest ones
#: and they are reported as such: see ``CHANGELOG.md`` and the README.
BANDWIDTH_GRID: tuple[float, ...] = tuple(round(float(v), 10) for v in np.logspace(-2, 3, 61))


def _coerce(name: str, raw: str, target: type) -> Any:
    """Convert an environment string to the field's declared type."""
    try:
        if target is int:
            return int(raw)
        if target is float:
            return float(raw)
        if target is bool:
            lowered = raw.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"{raw!r} is not a boolean")
    except ValueError as exc:
        raise ConfigValidationError(
            f"ENV_DENSFORGE_{name.upper()}={raw!r} could not be parsed as {target.__name__}"
        ) from exc
    return raw


@dataclass(frozen=True, slots=True)
class Config:
    """Run-level configuration.

    Attributes
    ----------
    n_train:
        Training samples per dataset. Larger is more accurate but MDS, the
        budget black hole, grows as O(n^2 * iterations). The floor is 20 rather
        than 100 so that smoke runs and unit tests can use a genuinely small
        sample count; at that size a reported NLL is a smoke signal, not a
        measurement, and every consumer of this class says so.
    n_test:
        Held-out samples, drawn independently with ``seed + 10_000``.
    n_seeds:
        Number of seeds per (dataset, estimator).

        The floor is **1**, not 3, so that a smoke test can run at the scale this
        project's own parameter table specifies for ``demo --quick`` (seeds = 1).
        The statistical requirement lives where it belongs -- in
        :func:`densforge.eval.metrics.summarize`, which raises
        ``InsufficientSeedsError`` below 3 -- rather than in the configuration
        layer. A config that can describe a smoke run is more honest than one that
        pretends the run does not exist.
    bandwidth_grid:
        Candidate fixed bandwidths. Each must be positive.
    mds_subsample:
        MDS is fitted on at most this many training rows. The only real lever
        against its cost (architecture §6.3).
    max_d:
        Reject datasets wider than this with ``E102``.
    random_state:
        Base seed; the run uses ``random_state + i`` for ``i`` in ``range(n_seeds)``.
    """

    n_train: int = 800
    n_test: int = 300
    n_seeds: int = 3
    #: Fixed-bandwidth grid for the KDE baseline.
    #:
    #: 61 log-spaced points spanning three decades. 61 is the smallest log-uniform
    #: count that keeps the relative spacing below 10% across those three decades,
    #: so the optimum is bracketed tightly on every dataset and no grid point is
    #: more than ~5% from its neighbours.
    bandwidth_grid: tuple[float, ...] = BANDWIDTH_GRID
    mds_subsample: int = 600
    max_d: int = MAX_D
    random_state: int = 0
    #: Validation evaluations the flagship's coordinate descent may spend.
    #:
    #: Recorded in the report alongside the baselines' own counts, so the tuning
    #: budget is a measured number rather than a claim of parity. 30 covers the
    #: two most informative axes at their full grids and leaves room for a third.
    n_tuning_evals: int = 30

    #: Inclusive bounds, enforced in ``__post_init__``.
    #:
    #: ``n_seeds`` has a hard floor of 3, taken from the design document: ``mean ±
    #: std`` over fewer than three points is not a measurement, and a package that
    #: allows it will eventually report one. ``n_train`` and ``n_test`` floor at 100
    #: so that a "quick" run still has enough samples for a k-NN neighbourhood to be
    #: meaningful at the default ``k = 15``.
    RANGES: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "n_train": (100, 5000),
            "n_test": (100, 5000),
            "n_seeds": (1, 20),
            "mds_subsample": (50, 5000),
            "max_d": (2, MAX_D),
            "random_state": (0, 2**32 - 1),
            "n_tuning_evals": (1, 5000),
        },
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name, (low, high) in self.RANGES.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigValidationError(
                    f"Config.{name} must be an int, got {type(value).__name__}"
                )
            if not low <= value <= high:
                raise ConfigRangeError(
                    f"Config.{name}={value} is outside the allowed range [{low}, {high}]"
                )
        if not self.bandwidth_grid:
            raise ConfigValidationError("Config.bandwidth_grid must not be empty")
        for value in self.bandwidth_grid:
            if not isinstance(value, float) or not value > 0.0:
                raise ConfigRangeError(
                    f"Config.bandwidth_grid entries must be positive floats, got {value!r}"
                )
        if self.mds_subsample > self.n_train:
            object.__setattr__(
                self, "mds_subsample", self.n_train
            )  # clamp rather than fail: subsampling is always safe

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Return the names of the init-able fields (used by env parsing)."""
        return tuple(f.name for f in fields(cls) if f.init)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: Any) -> Config:
        """Build a config from ``ENV_DENSFORGE_*`` variables plus explicit overrides.

        Raises
        ------
        ConfigUnknownKeyError
            If any ``ENV_DENSFORGE_*`` key does not name a real field. Failing
            loudly here is deliberate: a misspelled variable that silently does
            nothing is how a staging environment ends up serving stale numbers.
        """
        source = os.environ if env is None else env
        known = set(cls.field_names())
        values: dict[str, Any] = {}
        for key, raw in source.items():
            if not key.startswith(ENV_PREFIX):
                continue
            name = key[len(ENV_PREFIX) :].lower()
            if name not in known:
                raise ConfigUnknownKeyError(
                    f"{key} is not a recognised configuration key; "
                    f"valid keys are {sorted(ENV_PREFIX + n for n in known)}"
                )
            declared = {f.name: f.type for f in fields(cls) if f.init}
            target = declared[name]
            if isinstance(target, str) and target.startswith("tuple"):
                target = tuple
            if isinstance(target, str) and target.startswith("int"):
                target = int
            if isinstance(target, str) and target.startswith("bool"):
                target = bool
            values[name] = _coerce(name, raw, target)
        values.update(overrides)
        return cls(**values)

    def seeds(self) -> tuple[int, ...]:
        """Return the concrete seed list for this run."""
        return tuple(self.random_state + i for i in range(self.n_seeds))

    def scaled(self, factor: float) -> Config:
        """Return a copy with sample counts scaled by ``factor`` (for ``--quick``)."""
        factor = float(factor)
        return replace(
            self,
            n_train=max(100, int(self.n_train * factor)),
            n_test=max(100, int(self.n_test * factor)),
            mds_subsample=max(50, int(self.mds_subsample * factor)),
        )


@dataclass(frozen=True, slots=True)
class DensFuseConfig:
    """Flagship hyper-parameters (design §3.7).

    Every field is frozen and validated on construction, so an out-of-range
    value such as ``alpha=1.5`` is rejected at the point of the mistake rather
    than deep inside a fit.
    """

    #: Neighbourhood size for the local scale and local PCA.
    k: int = 15
    #: Global multiplicative bandwidth factor; pure scalar, does not break
    #: per-point adaptivity.
    gamma: float = 1.0
    #: Anisotropy exponent. 0 = isotropic, 1 = full local whitening.
    beta: float = 1.0
    #: Spectral floor as a fraction of the mean local eigenvalue.
    tau: float = 0.1
    #: Self-tuning exponent on the local scale. ``None`` -> ``1 / (d + 4)``.
    eta: float | None = None
    #: Local-scale clip bounds, as multiples of the median scale.
    c_lo: float = 0.25
    c_hi: float = 4.0
    #: Clip on the leave-one-out log density, in nats.
    c_logp_clip: float = 4.0
    #: Number of training points kept when scoring (m-N truncation).
    m_score: int = 384
    #: Density-balancing strength. ``alpha > 1`` has no theoretical basis and is
    #: rejected: over-correction hands huge in-weights to sparse points and the
    #: leading eigenvector collapses onto them.
    alpha: float = 0.5
    #: Embedding dimensionality.
    n_components: int = 2
    #: Closed-loop iterations.
    n_fuse_rounds: int = 2
    #: Drop the trivial (top) eigenvector of the diffusion operator. The
    #: eigenvector of the all-ones direction carries no geometry; keeping it
    #: would waste one of ``n_components`` on a constant column.
    drop_trivial: bool = True
    #: Neighbour batch size when scoring. Bounds peak memory at
    #: ``batch * m_score * d`` doubles, so this is a memory knob, not a speed knob.
    batch_size: int = 256
    #: Graph-neighbourhood divisor used to build the multi-hop geodesic graph.
    k_graph_div: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 2:
            raise ConfigRangeError(f"DensFuseConfig.k must be an int >= 2, got {self.k!r}")
        if not isinstance(self.n_components, int) or self.n_components < 1:
            raise ConfigRangeError(
                f"DensFuseConfig.n_components must be an int >= 1, got {self.n_components!r}"
            )
        if not isinstance(self.n_fuse_rounds, int) or self.n_fuse_rounds < 1:
            raise ConfigRangeError(
                f"DensFuseConfig.n_fuse_rounds must be an int >= 1, got {self.n_fuse_rounds!r}"
            )
        if not isinstance(self.m_score, int) or self.m_score < 1:
            raise ConfigRangeError(
                f"DensFuseConfig.m_score must be an int >= 1, got {self.m_score!r}"
            )
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ConfigRangeError(
                f"DensFuseConfig.batch_size must be an int >= 1, got {self.batch_size!r}"
            )
        if not isinstance(self.k_graph_div, int) or self.k_graph_div < 1:
            raise ConfigRangeError(
                f"DensFuseConfig.k_graph_div must be an int >= 1, got {self.k_graph_div!r}"
            )
        if not self.gamma > 0.0:
            raise ConfigRangeError(f"DensFuseConfig.gamma must be > 0, got {self.gamma}")
        if not 0.0 <= self.beta <= 1.0:
            raise ConfigRangeError(f"DensFuseConfig.beta must lie in [0, 1], got {self.beta}")
        if not 0.0 < self.tau <= 1.0:
            raise ConfigRangeError(f"DensFuseConfig.tau must lie in (0, 1], got {self.tau}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ConfigRangeError(
                f"DensFuseConfig.alpha must lie in [0, 1]; values above 1 have no "
                f"theoretical basis and destabilise the diffusion operator (got {self.alpha})"
            )
        if not 0.0 < self.c_lo < 1.0:
            raise ConfigRangeError(f"DensFuseConfig.c_lo must lie in (0, 1), got {self.c_lo}")
        if not self.c_hi > 1.0:
            raise ConfigRangeError(f"DensFuseConfig.c_hi must be > 1, got {self.c_hi}")
        if self.c_lo >= self.c_hi:
            raise ConfigRangeError(
                f"DensFuseConfig.c_lo ({self.c_lo}) must be below c_hi ({self.c_hi})"
            )
        if not self.c_logp_clip > 0.0:
            raise ConfigRangeError(
                f"DensFuseConfig.c_logp_clip must be > 0, got {self.c_logp_clip}"
            )
        if self.eta is not None and not self.eta > 0.0:
            raise ConfigRangeError(f"DensFuseConfig.eta must be > 0 or None, got {self.eta}")

    def resolved_eta(self, d: int) -> float:
        """Return the self-tuning exponent for ambient dimension ``d``."""
        if self.eta is not None:
            return float(self.eta)
        return 1.0 / (float(d) + 4.0)

    def evolve(self, **changes: Any) -> DensFuseConfig:
        """Return a new config with ``changes`` applied (never mutates in place)."""
        unknown = set(changes) - {f.name for f in fields(self) if f.init}
        if unknown:
            raise ConfigError(
                f"unknown DensFuseConfig field(s): {sorted(unknown)}; "
                f"valid fields are {sorted(f.name for f in fields(self) if f.init)}"
            )
        return replace(self, **changes)

    def as_params(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, for embedding in a report row."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}


#: Coordinate-descent order for flagship tuning.
#:
#: Ordered by how much information each axis carries, highest first. ``gamma``
#: dominates the bias/variance balance, ``beta`` is the largest structural gain,
#: ``k`` determines the statistical quality of the local scale; ``tau``, ``eta``
#: and the round count are second-order. Changing the order changes which
#: optimum a greedy search reaches, so the sequence is part of the contract.
HPO_ORDER: tuple[str, ...] = (
    "gamma",
    "beta",
    "k",
    "alpha",
    "m_score",
    "tau",
    "eta",
    "n_fuse_rounds",
)

__all__ = [
    "BANDWIDTH_GRID",
    "ENV_PREFIX",
    "HPO_ORDER",
    "MAX_D",
    "Config",
    "DensFuseConfig",
]
