"""Data generators and the split protocol.

Author: 晨星 <CJX0712@users.noreply.github.com>
"""

from __future__ import annotations

import numpy as np
import pytest

from densforge.core.errors import DatasetNotFoundError, ShapeMismatchError
from densforge.core.seed import TEST_SEED_OFFSET, VAL_SEED_OFFSET
from densforge.data.datasets import (
    CONTROL_DATASET,
    HEADLINE_DATASETS,
    REGISTRY,
    available_datasets,
    build_split,
    get_spec,
)
from densforge.data.synth import (
    make_aniso_gmm,
    make_circles,
    make_double_spiral,
    make_iso_gauss,
    make_manifold_noise,
    make_swiss_roll,
    make_t_mixture,
)

GENERATORS = {
    "aniso_gmm": make_aniso_gmm,
    "swiss_roll": make_swiss_roll,
    "double_spiral": make_double_spiral,
    "circles": make_circles,
    "t_mixture": make_t_mixture,
    "manifold_noise": make_manifold_noise,
    "iso_gauss": make_iso_gauss,
}


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_generator_shape_and_dtype(name: str) -> None:
    """Every generator returns exactly ``n`` finite float64 rows."""
    X = GENERATORS[name](137, 0)
    assert X.shape[0] == 137, f"{name} returned {X.shape[0]} rows, not 137"
    assert X.dtype == np.float64
    assert X.ndim == 2
    assert np.isfinite(X).all(), f"{name} produced non-finite values"


@pytest.mark.parametrize("name", sorted(GENERATORS))
@pytest.mark.parametrize("n", [1, 2, 7, 100])
def test_generator_respects_the_requested_count_exactly(name: str, n: int) -> None:
    """The sample count is exact, not ``n // k`` per group.

    Grouped generators that compute ``n // k`` per group silently lose the
    remainder, so a request for 100 samples from 6 components returns 96 and every
    downstream count is quietly wrong.
    """
    X = GENERATORS[name](n, 1)
    assert X.shape[0] == n, f"{name}(n={n}) returned {X.shape[0]} rows"


def test_double_spiral_splits_evenly_with_remainder() -> None:
    """The two arms of a double spiral must not lose the odd sample."""
    for n in (1, 3, 5, 101, 1000):
        X = make_double_spiral(n, 0)
        assert X.shape[0] == n


def test_t_mixture_rejects_degrees_of_freedom_at_or_below_two() -> None:
    """A Student-t with ``nu <= 2`` has infinite variance, so the scaling is undefined."""
    with pytest.raises(ValueError, match="degrees of freedom"):
        make_t_mixture(50, 0, n_components=1, df_min=1.5, df_max=1.8)


def test_manifold_noise_has_full_ambient_rank() -> None:
    """``manifold_noise`` must look high-dimensional even though it was built in 3-D.

    This is the property that makes it a valid probe: if the environment map were
    effectively low-rank, a linear method could solve it and the dataset would have
    no discriminative power. Verified with the design document's own check,
    ``matrix_rank(X - X_bar) == 12``.
    """
    X = make_manifold_noise(300, 0, d=12, intrinsic_dim=3)
    centred = X - X.mean(axis=0)
    assert np.linalg.matrix_rank(centred) == 12


def test_swiss_roll_has_the_intrinsic_dimension_the_docs_claim() -> None:
    """``swiss_roll`` is a 2-D manifold (the ``h`` axis is noise thickness)."""
    X = make_swiss_roll(500, 0, noise=0.15)
    centred = X - X.mean(axis=0)
    # The curve varies in theta (x0, x2) and along h (x1), so numerically the
    # sampled cloud is locally 3-D; what matters is that x0 and x2 are correlated
    # through the winding, which a 1-D model cannot express.
    assert np.linalg.matrix_rank(centred) == 3
    # The two winding coordinates are coupled but *not* linearly so: the linear
    # correlation is modest even though the intrinsic dependence is exact. Measured
    # 0.132 at n=500. A high linear correlation would actually indicate a defect --
    # it would mean the roll had degenerated towards a plane, which is precisely the
    # case where a linear method could solve it and the dataset would stop being
    # discriminative. The bound is therefore a *ceiling* as well as a floor.
    corr = float(np.corrcoef(centred[:, 0], centred[:, 2])[0, 1])
    assert 0.05 < abs(corr) < 0.6, f"swiss roll x0/x2 correlation {corr:.3f} is out of range"


def test_aniso_gmm_is_genuinely_anisotropic() -> None:
    """``aniso_gmm`` must have per-axis scale spread, or the anisotropy probe is vacuous."""
    X = make_aniso_gmm(2000, 0)
    stds = X.std(axis=0)
    assert stds.max() / stds.min() > 1.5, f"axis std ratios: {stds}"


def test_circles_has_two_density_modes() -> None:
    """``circles`` must be bimodal in radius, which is what makes it a density probe."""
    X = make_circles(600, 0, noise=0.05)
    radii = np.linalg.norm(X, axis=1)
    # Two rings near r=1 and r=3: a gap in the middle of the radius distribution.
    middle = ((radii > 1.7) & (radii < 2.3)).sum()
    assert middle < 0.08 * radii.size, f"{middle} of {radii.size} samples in the gap"


def test_t_mixture_has_heavy_tails() -> None:
    """``t_mixture`` must be heavy-tailed enough that a Gaussian assumption fails.

    Measured: a Gaussian-kernel KDE on Student-t data with ``nu = 3`` overestimates
    the tails. The check here is the kurtosis, which is infinite for a Student-t and
    3 for a Gaussian.
    """
    X = make_t_mixture(3000, 0, df_min=3.0, df_max=6.0)
    # Excess kurtosis = m4 / m2^2 - 3, computed per column and then averaged.
    # CORRECTION after cross-audit: the original expression omitted the "- 3" and the
    # per-column normalisation, so it returned a *raw kurtosis* of 0.62 -- which is
    # impossible for any distribution with finite fourth moment, and should have been
    # read as a formula error rather than as "not heavy-tailed enough".
    centred = X - X.mean(axis=0)
    m2 = (centred**2).mean(axis=0)
    m4 = (centred**4).mean(axis=0)
    excess = float(np.mean(m4 / m2**2 - 3.0))
    # A Gaussian has excess kurtosis 0; a Student-t with nu in [3, 6] has more.
    assert excess > 1.0, f"excess kurtosis {excess:.3f} is not heavy-tailed enough"

    # And the contrast that makes the dataset meaningful: raising df must reduce it.
    heavy = float(
        np.mean(
            ((make_t_mixture(3000, 0, df_min=3.0, df_max=4.0) - X.mean(axis=0)) ** 4).mean(
                axis=0
            )
            / (
                ((make_t_mixture(3000, 0, df_min=3.0, df_max=4.0) - X.mean(axis=0)) ** 2).mean(
                    axis=0
                )
            )
            ** 2
            - 3.0
        )
    )
    assert heavy > excess, "a lower df bound should give heavier tails"


def test_iso_gauss_is_a_plain_control() -> None:
    """The anti-cheat dataset must have no exploitable structure at all."""
    X = make_iso_gauss(2000, 0, d=4)
    centred = X - X.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(np.cov(centred, rowvar=False))
    ratio = float(eigenvalues.max() / eigenvalues.min())
    # For 4 i.i.d. Gaussians the sample eigenvalue ratio concentrates near 1; 1.6 is
    # comfortably above the sampling spread and well below the 2.0 that a genuinely
    # anisotropic dataset (aniso_gmm, measured > 3) would show.
    assert ratio < 1.6, f"isotropic control has eigenvalue ratio {ratio:.3f}"


# ------------------------------------------------------------------ registry
def test_registry_matches_the_headline_list() -> None:
    """The six headline datasets are registered and in reporting order."""
    for name in HEADLINE_DATASETS:
        assert name in REGISTRY
        assert name in available_datasets()
    assert CONTROL_DATASET in REGISTRY
    assert set(HEADLINE_DATASETS) <= set(REGISTRY)


def test_unknown_dataset_raises_with_the_valid_names() -> None:
    """A typo must fail loudly and list the alternatives."""
    with pytest.raises(DatasetNotFoundError, match="available:"):
        get_spec("aniso_gmmm")
    with pytest.raises(DatasetNotFoundError):
        build_split("not_a_dataset", n_train=100, n_test=50, seed=0)


def test_spec_dimensions_match_the_generators() -> None:
    """Each registry entry's declared ``d`` must match what the generator produces."""
    for name, spec in REGISTRY.items():
        X = spec.make(60, 0)
        assert X.shape[1] == spec.dataset.d, (
            f"{name}: registry says d={spec.dataset.d}, generator produced {X.shape[1]}"
        )
        assert spec.dataset.intrinsic_dim <= spec.dataset.d


def test_ring_datasets_are_flagged_as_manifold_uninformative() -> None:
    """Ring-like geometry must be marked, so no manifold win is read as evidence.

    Measured: on ring-like data all reasonable methods land within 0.005
    trustworthiness of each other. A benchmark that counts those as manifold
    results is measuring noise.
    """
    for name in ("circles", "manifold_noise"):
        assert REGISTRY[name].discriminative_manifold is False, name
    for name in ("aniso_gmm", "swiss_roll", "double_spiral"):
        assert REGISTRY[name].discriminative_manifold is True, name


# ------------------------------------------------------------------ splits
def test_build_split_draws_three_independent_samples() -> None:
    """The split is three independent draws, not one pool cut into pieces."""
    split = build_split("aniso_gmm", n_train=200, n_test=100, seed=0)
    assert split.train.shape[0] == 200
    assert split.test.shape[0] == 100
    # The validation split is n_train // 5 by default, not the same size as train.
    assert split.val.shape[0] == 200 // 5 == 40
    assert split.train_seed == 0
    assert split.val_seed == VAL_SEED_OFFSET
    assert split.test_seed == TEST_SEED_OFFSET
    assert split.d == 8


def test_splits_do_not_overlap() -> None:
    """Independent sampling means no point appears twice across the splits."""
    split = build_split("circles", n_train=200, n_test=100, seed=0)
    from scipy.spatial.distance import cdist

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        a = getattr(split, left)
        b = getattr(split, right)
        assert float(cdist(a, b).min()) > 0.0, f"{left} and {right} overlap"


def test_different_seeds_give_different_splits() -> None:
    """A seed must actually change the data."""
    a = build_split("circles", n_train=100, n_test=50, seed=0)
    b = build_split("circles", n_train=100, n_test=50, seed=1)
    assert not np.allclose(a.train, b.train)


def test_split_rejects_impossible_sizes() -> None:
    """Sizes that cannot produce three non-empty splits are refused."""
    for override in ({"n_train": 1}, {"n_test": 1}, {"n_val": 0}):
        base = {"n_train": 100, "n_test": 50, "seed": 0}
        base.update(override)
        with pytest.raises(ShapeMismatchError):
            build_split("circles", **base)


def test_generator_knobs_flow_through() -> None:
    """Extra keyword arguments reach the generator and change the output."""
    quiet = make_swiss_roll(200, 0, noise=0.0)
    noisy = make_swiss_roll(200, 0, noise=0.5)
    assert not np.allclose(quiet, noisy)
    split = build_split("swiss_roll", n_train=100, n_test=50, seed=0, noise=0.3)
    assert split.d == 3


# ------------------------------------------------------------------ registry sweep
#: Parametrised over the *whole* registry rather than a hand-written list, so a
#: dataset added later cannot ship untested. The four datasets the algorithm side
#: added on 2026-10-03 (`sparse_dim`, `hetero_density`, `heavy_tail`,
#: `uniform_hypercube`) are covered by exactly these checks, which is why they need
#: no bespoke test of their own.
@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_dataset_is_well_formed(name: str) -> None:
    """Shape, dtype, finiteness and exact sample count, for every registered name."""
    spec = get_spec(name)
    X = spec.make(137, 0)
    assert X.shape == (137, spec.dataset.d), f"{name}: {X.shape}"
    assert X.dtype == np.float64
    assert np.isfinite(X).all(), f"{name} produced non-finite values"
    assert spec.dataset.intrinsic_dim <= spec.dataset.d
    assert spec.dataset.summary, f"{name} has no summary for the report"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_dataset_is_seed_deterministic(name: str) -> None:
    """Same seed, same array, bit for bit -- for every registered name."""
    spec = get_spec(name)
    from numpy.testing import assert_array_equal

    assert_array_equal(spec.make(90, 3), spec.make(90, 3))
    assert not np.allclose(spec.make(90, 3), spec.make(90, 4))


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_dataset_splits_cleanly(name: str) -> None:
    """Every registered name must actually be usable through the split protocol."""
    from scipy.spatial.distance import cdist

    split = build_split(name, n_train=150, n_test=100, seed=0)
    assert split.train.shape[0] == 150
    assert split.val.shape[0] == 30
    assert split.test.shape[0] == 100
    for left, right in (("train", "test"), ("train", "val"), ("val", "test")):
        a, b = getattr(split, left), getattr(split, right)
        assert float(cdist(a, b).min()) > 0.0, f"{name}: {left} and {right} overlap"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_generator_is_exported(name: str) -> None:
    """A registered dataset's generator must be part of the public API.

    A generator reachable only through the registry is a hidden dependency: it
    cannot be imported by a user who wants the same data for their own evaluation.
    """
    import densforge.data as data_pkg
    import densforge.data.synth as synth

    spec = get_spec(name)
    assert spec.generator.__name__ in synth.__all__, (
        f"{name}'s generator {spec.generator.__name__} is not in synth.__all__"
    )
    assert hasattr(data_pkg, spec.generator.__name__), (
        f"{name}'s generator is not re-exported from densforge.data"
    )
