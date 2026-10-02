"""Shared fixtures.

Author: 晨星 <CJX0712@users.noreply.github.com>
"""

from __future__ import annotations

import numpy as np
import pytest

from densforge.core.config import Config, DensFuseConfig
from densforge.core.seed import set_all
from densforge.data.datasets import build_split
from densforge.density.flagship import DensFuse


@pytest.fixture(autouse=True)
def _deterministic_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin threading and reseed before every test.

    Two reasons. First, an invariant that only holds when a previous test happened
    to consume a particular number of random draws is not an invariant. Second,
    multi-threaded BLAS reductions sum in a nondeterministic order, which perturbs
    the last bits and would make the bit-exactness assertions flaky rather than
    informative.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.setenv(var, "1")
    set_all(0)


@pytest.fixture(scope="session")
def small_config() -> Config:
    """A config small enough that the full suite stays fast."""
    return Config(n_train=300, n_test=150, n_seeds=3, mds_subsample=200)


@pytest.fixture(scope="session")
def aniso_split():
    """``aniso_gmm`` split, session-scoped because generation is not free."""
    return build_split("aniso_gmm", n_train=300, n_test=150, seed=0)


@pytest.fixture(scope="session")
def swiss_split():
    """``swiss_roll`` split."""
    return build_split("swiss_roll", n_train=300, n_test=150, seed=0)


@pytest.fixture(scope="session")
def fitted_flagship(aniso_split):
    """A flagship fitted on ``aniso_gmm`` with a single fuse round."""
    return DensFuse(DensFuseConfig(k=12, n_fuse_rounds=1)).fit(aniso_split.train)


@pytest.fixture
def quick_config() -> DensFuseConfig:
    """A fast flagship config for tests that do not measure accuracy."""
    return DensFuseConfig(k=10, n_fuse_rounds=1, batch_size=64)


@pytest.fixture
def gaussian_cloud() -> np.ndarray:
    """A small anisotropic Gaussian cloud, ``d = 3``."""
    rs = np.random.RandomState(7)
    return rs.randn(200, 3) * np.array([3.0, 1.0, 0.5]) + 2.0
