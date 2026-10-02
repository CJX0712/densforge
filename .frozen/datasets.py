"""Dataset registry and the leakage-proof split protocol.

Author: 晨星 <CJX0712@users.noreply.github.com>

Split protocol (architecture §5.1)
----------------------------------
::

    make(n_train,           seed)           -> train   # the only data fit() may see
    make(n_train // 5,      seed + 500)     -> val     # hyper-parameter selection only
    make(n_test,            seed + 10_000)  -> test    # final scoring only

Three **independent draws**, never a random shuffle of one pool. Splitting a
single pool would make the two halves share sampling noise, which biases test
NLL optimistically and makes the headline number unfalsifiable. Because the test
seed is the train seed plus a fixed offset, the independence is itself
assertable (assertion A1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..core.errors import DatasetNotFoundError, SeedCollisionError, ShapeMismatchError
from ..core.seed import TEST_SEED_OFFSET, VAL_SEED_OFFSET, derive_seed
from ..core.types import Dataset, FloatArray, Split
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

#: A generator is any callable matching ``(n, seed, **knobs) -> (n, d)``.
Generator = Callable[..., FloatArray]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """A named generator plus its static description."""

    generator: Generator
    dataset: Dataset
    #: Whether this dataset is expected to be discriminative for the *manifold*
    #: task. Ring-like geometry is not: all reasonable methods land within 0.005
    #: trustworthiness of each other, so the benchmark must not read a manifold
    #: win on it as evidence.
    discriminative_manifold: bool = True

    def make(self, n: int, seed: int, **knobs: float) -> FloatArray:
        """Draw ``n`` samples. Same arguments always give the same array."""
        return self.generator(n, seed, **knobs)


def _spec(
    name: str,
    generator: Generator,
    *,
    d: int,
    intrinsic_dim: int,
    summary: str,
    discriminative_manifold: bool = True,
) -> DatasetSpec:
    return DatasetSpec(
        generator=generator,
        dataset=Dataset(
            name=name,
            n_train=1000,
            n_test=400,
            d=d,
            intrinsic_dim=intrinsic_dim,
            seed=0,
            summary=summary,
        ),
        discriminative_manifold=discriminative_manifold,
    )


#: The registry. Insertion order is the benchmark's reporting order.
REGISTRY: dict[str, DatasetSpec] = {
    "sparse_dim": _spec(
        "sparse_dim",
        make_sparse_dim,
        d=16,
        intrinsic_dim=16,
        summary="4 active axes + isotropic thickness in 16-D: the d>6 curse probe",
    ),
    "hetero_density": _spec(
        "hetero_density",
        make_hetero_density,
        d=6,
        intrinsic_dim=6,
        summary="6:1:3 unequal-weight far clusters: the density trap",
    ),
    "heavy_tail": _spec(
        "heavy_tail",
        make_heavy_tail,
        d=4,
        intrinsic_dim=4,
        summary="Student-t mixture, df in [3, 30]: graceful tail degradation",
    ),
    "uniform_hypercube": _spec(
        "uniform_hypercube",
        make_uniform_hypercube,
        d=4,
        intrinsic_dim=4,
        summary="CONTROL: flat local density, adaptive bandwidth has nothing to exploit",
    ),
    "aniso_gmm": _spec(
        "aniso_gmm",
        make_aniso_gmm,
        d=8,
        intrinsic_dim=8,
        summary="6-component mixture, per-axis sigma in [0.25, 2.0]: scale + anisotropy",
    ),
    "swiss_roll": _spec(
        "swiss_roll",
        make_swiss_roll,
        d=3,
        intrinsic_dim=2,
        summary="1-D curve wound into 3-D: the geometric gold standard",
    ),
    "double_spiral": _spec(
        "double_spiral",
        make_double_spiral,
        d=3,
        intrinsic_dim=2,
        summary="two interleaved helices: topological unfolding under ambient proximity",
    ),
    "circles": _spec(
        "circles",
        make_circles,
        d=2,
        intrinsic_dim=1,
        summary="two concentric rings: bimodal density, 1-D geometry",
        discriminative_manifold=False,
    ),
    "t_mixture": _spec(
        "t_mixture",
        make_t_mixture,
        d=6,
        intrinsic_dim=6,
        summary="Student-t mixture, nu in [3, 30]: when the Gaussian assumption fails",
    ),
    "manifold_noise": _spec(
        "manifold_noise",
        make_manifold_noise,
        d=12,
        intrinsic_dim=3,
        summary="3-D manifold under a tanh environment map, ambient rank 12",
        discriminative_manifold=False,
    ),
    # Anti-cheat control. Not part of the six-dataset headline table; the gate
    # runs on it separately and requires a tie, not a win.
    "iso_gauss": _spec(
        "iso_gauss",
        make_iso_gauss,
        d=4,
        intrinsic_dim=4,
        summary="CONTROL: single isotropic Gaussian, no exploitable structure",
        discriminative_manifold=False,
    ),
}

#: The six datasets that make up the headline benchmark, in reporting order.
HEADLINE_DATASETS: tuple[str, ...] = (
    "aniso_gmm",
    "swiss_roll",
    "double_spiral",
    "circles",
    "t_mixture",
    "manifold_noise",
)

#: The anti-cheat control dataset.
CONTROL_DATASET = "iso_gauss"


def available_datasets() -> list[str]:
    """Return every registered dataset name, sorted."""
    return sorted(REGISTRY)


def get_spec(name: str) -> DatasetSpec:
    """Look up a dataset specification.

    Raises
    ------
    DatasetNotFoundError
        If ``name`` is not registered. The message lists the valid names, because
        a typo that silently fell back to a default would corrupt a whole run.
    """
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise DatasetNotFoundError(
            f"unknown dataset {name!r}; available: {', '.join(available_datasets())}"
        ) from exc


def _as_2d_float(x: FloatArray, label: str) -> FloatArray:
    """Validate that ``x`` is a finite 2-D float64 array."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ShapeMismatchError(f"{label} must be 2-D (n, d), got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ShapeMismatchError(f"{label} contains non-finite values")
    return np.ascontiguousarray(arr)


def build_split(
    name: str,
    *,
    n_train: int,
    n_test: int,
    seed: int,
    n_val: int | None = None,
    **knobs: float,
) -> Split:
    """Draw an independent train / val / test triple.

    Parameters
    ----------
    name:
        Registered dataset name.
    n_train, n_test:
        Sample counts for the train and test draws.
    seed:
        Base seed. ``val`` uses ``seed + 500`` and ``test`` uses ``seed + 10_000``.
    n_val:
        Validation size; defaults to ``n_train // 5``.
    **knobs:
        Forwarded to the generator (e.g. ``noise=0.3``).

    Raises
    ------
    SeedCollisionError
        If any two split seeds coincide. This is checked, not assumed: a
        collision would silently turn the test set into a copy of the training
        set and every metric would become meaningless.
    ShapeMismatchError
        If the three draws disagree on dimensionality.
    """
    spec = get_spec(name)
    n_val = n_train // 5 if n_val is None else int(n_val)
    if n_train < 2 or n_test < 2 or n_val < 1:
        raise ShapeMismatchError(
            f"split sizes must satisfy n_train>=2, n_test>=2, n_val>=1; "
            f"got {n_train}/{n_val}/{n_test}"
        )

    train_seed = derive_seed(seed, 0)
    val_seed = derive_seed(seed, VAL_SEED_OFFSET)
    test_seed = derive_seed(seed, TEST_SEED_OFFSET)

    if len({train_seed, val_seed, test_seed}) != 3:
        raise SeedCollisionError(
            f"split seeds collided (train={train_seed}, val={val_seed}, test={test_seed}); "
            "independent sampling requires three distinct seeds"
        )

    train = _as_2d_float(spec.make(n_train, train_seed, **knobs), "train")
    val = _as_2d_float(spec.make(n_val, val_seed, **knobs), "val")
    test = _as_2d_float(spec.make(n_test, test_seed, **knobs), "test")

    dims = {train.shape[1], val.shape[1], test.shape[1]}
    if len(dims) != 1:
        raise ShapeMismatchError(
            f"split dimensionality disagrees for {name!r}: "
            f"{train.shape}, {val.shape}, {test.shape}"
        )

    return Split(
        train=train,
        val=val,
        test=test,
        dataset=name,
        train_seed=train_seed,
        val_seed=val_seed,
        test_seed=test_seed,
    )


def min_train_test_distance(split: Split, probe: int = 64) -> float:
    """Return the smallest distance from the first ``probe`` training rows to test.

    This is assertion A4. Under independent sampling with continuous
    distributions the true minimum is 0 only with probability zero, so a value of
    exactly ``0.0`` means the two sets were produced by slicing one pool. Measured
    on this machine: 5.15.
    """
    from scipy.spatial.distance import cdist  # local: keeps scipy off the import path

    head = split.train[: min(probe, split.train.shape[0])]
    return float(cdist(head, split.test).min())


__all__ = [
    "CONTROL_DATASET",
    "HEADLINE_DATASETS",
    "REGISTRY",
    "DatasetSpec",
    "Generator",
    "available_datasets",
    "build_split",
    "get_spec",
    "min_train_test_distance",
]
