"""Configuration, errors, and the typed record layer.

Author: 晨星 <CJX0712@users.noreply.github.com>
"""

from __future__ import annotations

import numpy as np
import pytest

from densforge.core.config import ENV_PREFIX, HPO_ORDER, Config, DensFuseConfig
from densforge.core.errors import (
    ArtifactNotFoundError,
    AtomicWriteError,
    BackendUnavailableError,
    ConfigError,
    ConfigRangeError,
    ConfigUnknownKeyError,
    ConfigValidationError,
    DataError,
    DatasetNotFoundError,
    DensForgeError,
    EvalError,
    InsufficientSeedsError,
    LeakageError,
    ModelError,
    NumericalOverflowError,
    SeedCollisionError,
    SerializationError,
    ShapeMismatchError,
)
from densforge.core.types import Dataset, Report, Result, Status, array_fingerprint
from densforge.eval.metrics import summarize


# =========================================================== error hierarchy
def test_every_error_carries_a_stable_code() -> None:
    """Each error's code is part of the public contract, so it must be stable.

    Tests assert on codes rather than message text, which lets the wording improve
    without breaking callers -- and a missing code would make that impossible.
    """
    from densforge.core import errors

    seen: dict[str, type] = {}
    for name in dir(errors):
        obj = getattr(errors, name)
        if not (isinstance(obj, type) and issubclass(obj, DensForgeError)):
            continue
        if obj is DensForgeError:
            continue
        instance = obj("message")
        assert instance.code.startswith("E"), f"{name} has no E-prefixed code"
        assert instance.code in instance.args[0], f"{name}: code missing from the message"
        if obj.default_code != "E000":
            seen[instance.code] = obj
    # Codes are unique per class in the same segment.
    assert len(seen) == len(set(seen)), "duplicate error codes"


def test_error_segments_map_to_the_documented_families() -> None:
    """E1xx config, E2xx data, E3xx model, E4xx eval, E5xx IO."""

    families = {
        "E1": (ConfigError, ConfigValidationError, ConfigRangeError, ConfigUnknownKeyError),
        "E2": (DataError, DatasetNotFoundError, ShapeMismatchError, SeedCollisionError),
        "E3": (ModelError, BackendUnavailableError, NumericalOverflowError),
        "E4": (EvalError, LeakageError),
        "E5": (ArtifactNotFoundError, SerializationError, AtomicWriteError),
    }
    for prefix, classes in families.items():
        for cls in classes:
            assert cls("m").code.startswith(prefix), f"{cls.__name__} -> {cls('m').code}"


def test_leakage_errors_are_a_subclass_of_eval_error() -> None:
    """Catching ``LeakageError`` must also catch the specific variants."""
    assert issubclass(LeakageError, EvalError)
    assert issubclass(ConfigRangeError, ConfigError)
    assert issubclass(NumericalOverflowError, ModelError)


# =========================================================== Config
def test_config_defaults_are_within_their_documented_ranges() -> None:
    """Every default must satisfy its own range check."""
    Config()  # must not raise


def test_config_floors_and_the_strict_reporting_layer() -> None:
    """The seed floor is enforced at *reporting*, not at configuration.

    Two different guards, deliberately at different layers:

    * ``Config`` accepts ``n_seeds >= 1``, because a smoke run that draws one seed
      is a legitimate thing to *execute*;
    * :func:`~densforge.eval.metrics.summarize` refuses to produce a report below 3
      seeds, because ``mean +/- std`` over fewer points is not a measurement.

    Putting the floor in ``Config`` alone would be wrong in the other direction: it
    would make a 1-seed run impossible rather than merely unreportable, and the
    error would arrive at construction instead of at the point where the statistics
    actually become meaningless. Both halves are asserted here.
    """
    from densforge.core.errors import InsufficientSeedsError
    from densforge.core.types import Result
    from densforge.eval.metrics import summarize

    ranges = Config().RANGES
    assert ranges["n_train"][0] == 100
    assert ranges["n_test"][0] == 100

    # The sample-size floors keep a "quick" run large enough for a k-NN
    # neighbourhood to mean anything at the default k = 15.
    for field in ("n_train", "n_test"):
        low = int(ranges[field][0])
        Config(**{field: low})  # the floor itself is accepted
        with pytest.raises(ConfigRangeError):
            Config(**{field: low - 1})

    # One seed is configurable...
    assert Config(n_seeds=1).n_seeds == 1
    # ...but it cannot be reported.
    rows = (Result("d", "e", Status.OK, nll=1.0),)
    for n_seeds in (1, 2):
        with pytest.raises(InsufficientSeedsError, match="minimum is 3"):
            summarize(rows, n_seeds=n_seeds)
    summarize(rows, n_seeds=3)  # three is enough


def test_config_accepts_a_single_seed_for_a_smoke_run() -> None:
    """``n_seeds = 1`` is legal at the config layer; only reporting refuses it.

    A smoke test legitimately runs at one seed, and ``summarize`` is where the
    statistical requirement belongs. Splitting it the other way -- config forbids 1,
    so the smoke test cannot be expressed at all -- would push developers toward
    faking a three-seed run.
    """
    config = Config(n_seeds=1)
    assert config.n_seeds == 1
    with pytest.raises(InsufficientSeedsError):
        summarize([], n_seeds=config.n_seeds)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("n_train", 99, ConfigRangeError),
        ("n_train", 99999, ConfigRangeError),
        ("n_test", 99, ConfigRangeError),
        # n_seeds accepts 1: the mean +/- std requirement is enforced by `summarize`.
        ("n_seeds", 0, ConfigRangeError),
        ("n_seeds", 21, ConfigRangeError),
        ("max_d", 1, ConfigRangeError),
        ("max_d", 64, ConfigRangeError),
        ("random_state", -1, ConfigRangeError),
        ("n_train", 1.5, ConfigValidationError),
        ("n_train", True, ConfigValidationError),
    ],
)
def test_config_rejects_out_of_range_values(field, value, error) -> None:
    """A bad value is refused with a typed, specific error naming the field."""
    with pytest.raises(error) as excinfo:
        Config(**{field: value})
    assert field in str(excinfo.value)


def test_config_rejects_an_empty_or_invalid_bandwidth_grid() -> None:
    """An empty grid would make tuning impossible; a non-positive entry is nonsense."""
    with pytest.raises(ConfigValidationError, match="bandwidth_grid"):
        Config(bandwidth_grid=())
    with pytest.raises(ConfigRangeError, match="positive"):
        Config(bandwidth_grid=(0.0, 1.0))
    with pytest.raises(ConfigRangeError, match="positive"):
        Config(bandwidth_grid=(1.0, -2.0))


def test_config_clamps_mds_subsample_rather_than_failing() -> None:
    """Subsampling more than the training set is harmless, so it is clamped.

    Failing would be defensible but unhelpful: the user's intent -- "use at most this
    many rows" -- is already satisfied.
    """
    config = Config(n_train=200, mds_subsample=600)
    assert config.mds_subsample == 200


def test_config_env_overrides() -> None:
    """``ENV_DENSFORGE_*`` variables override the defaults."""
    env = {
        f"{ENV_PREFIX}N_TRAIN": "1234",
        f"{ENV_PREFIX}N_SEEDS": "5",
        f"{ENV_PREFIX}RANDOM_STATE": "42",
    }
    config = Config.from_env(env)
    assert config.n_train == 1234
    assert config.n_seeds == 5
    assert config.random_state == 42


def test_config_explicit_overrides_beat_the_environment() -> None:
    """Precedence: explicit argument > environment > default."""
    env = {f"{ENV_PREFIX}N_TRAIN": "1234"}
    assert Config.from_env(env, n_train=999).n_train == 999


def test_config_rejects_an_unknown_env_key() -> None:
    """A misspelled variable must fail loudly, not silently do nothing.

    A silently-ignored deployment variable is a classic way for staging to serve
    stale numbers while everyone believes the setting took effect.
    """
    with pytest.raises(ConfigUnknownKeyError, match="not a recognised"):
        Config.from_env({f"{ENV_PREFIX}N_TRAINN": "100"})
    with pytest.raises(ConfigUnknownKeyError, match="valid keys"):
        Config.from_env({f"{ENV_PREFIX}TYPO": "1"})


def test_config_rejects_an_unparseable_env_value() -> None:
    """A non-numeric value in a numeric variable is a typed error."""
    with pytest.raises(ConfigValidationError, match="could not be parsed"):
        Config.from_env({f"{ENV_PREFIX}N_TRAIN": "many"})


def test_config_ignores_unrelated_env_variables() -> None:
    """Only ``ENV_DENSFORGE_*`` variables are considered."""
    config = Config.from_env({"PATH": "/usr/bin", "HOME": "/root"})
    assert config.n_train == Config().n_train


def test_config_seeds_and_scaled() -> None:
    """``seeds()`` is derived from the base seed; ``scaled()`` shrinks for ``--quick``."""
    config = Config(n_seeds=3, random_state=7)
    assert config.seeds() == (7, 8, 9)
    small = config.scaled(0.25)
    assert small.n_train == max(100, int(config.n_train * 0.25))
    assert small.n_seeds == config.n_seeds


# =========================================================== DensFuseConfig
def test_flagship_config_defaults() -> None:
    """The documented defaults from the design document's hyper-parameter table."""
    config = DensFuseConfig()
    assert config.k == 15
    assert config.gamma == 1.0
    assert config.beta == 1.0
    assert config.tau == 0.1
    assert config.eta is None, "eta defaults to 1/(d+4), resolved from the data"
    assert (config.c_lo, config.c_hi) == (0.25, 4.0)
    assert config.c_logp_clip == 4.0
    assert config.m_score == 384
    assert config.alpha == 0.5
    assert config.n_components == 2
    assert config.n_fuse_rounds == 2


def test_flagship_config_resolves_eta_from_dimension() -> None:
    """``eta`` defaults to ``1 / (d + 4)``, the Zelnik-Manor exponent."""
    config = DensFuseConfig()
    assert config.resolved_eta(2) == pytest.approx(1 / 6)
    assert config.resolved_eta(8) == pytest.approx(1 / 12)
    assert DensFuseConfig(eta=0.5).resolved_eta(8) == 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("k", 1),
        ("k", 2.5),
        ("n_components", 0),
        ("n_fuse_rounds", 0),
        ("m_score", 0),
        ("batch_size", 0),
        ("k_graph_div", 0),
        ("gamma", 0.0),
        ("gamma", -1.0),
        ("beta", -0.1),
        ("beta", 1.1),
        ("tau", 0.0),
        ("tau", 1.5),
        ("alpha", -0.1),
        ("alpha", 1.5),
        ("c_lo", 0.0),
        ("c_lo", 1.5),
        ("c_hi", 0.5),
        ("c_logp_clip", 0.0),
        ("eta", 0.0),
        ("eta", -1.0),
    ],
)
def test_flagship_config_rejects_invalid_values(field, value) -> None:
    """Out-of-range hyper-parameters are refused at construction.

    ``alpha > 1`` in particular: it has no theoretical interpretation, and the
    failure mode -- sparse points capturing the leading eigenvector -- is silent
    until the embedding is already wrong.
    """
    with pytest.raises(ConfigError):
        DensFuseConfig(**{field: value})


def test_flagship_config_rejects_inverted_clip_bounds() -> None:
    """``c_lo >= c_hi`` would collapse the clip window to nothing."""
    with pytest.raises(ConfigRangeError, match="c_lo"):
        DensFuseConfig(c_lo=2.0, c_hi=1.0)


def test_flagship_config_is_frozen_and_evolves() -> None:
    """Tuning produces a new config; it never mutates the old one.

    Frozen configs are what make invariant I23 ("select does not mutate the fit
    state") checkable by value comparison.
    """
    config = DensFuseConfig()
    with pytest.raises(Exception):
        config.gamma = 2.0  # type: ignore[misc]
    evolved = config.evolve(gamma=2.0, beta=0.5)
    assert evolved.gamma == 2.0 and evolved.beta == 0.5
    assert config.gamma == 1.0 and config.beta == 1.0
    assert evolved.k == config.k, "unmentioned fields must be preserved"


def test_flagship_config_rejects_unknown_fields() -> None:
    """A typo in a hyper-parameter name must not be silently ignored."""
    with pytest.raises(ConfigError, match="unknown"):
        DensFuseConfig().evolve(gammma=2.0)


def test_flagship_config_params_are_serialisable() -> None:
    """``as_params()`` goes straight into the benchmark JSON."""
    import json

    json.dumps(DensFuseConfig().as_params())


def test_hpo_order_is_the_documented_sequence() -> None:
    """The search order is part of the contract: a different order finds a different optimum.

    ``gamma`` first because it dominates the bias/variance trade-off; ``tau``,
    ``eta`` and the round count last because they are second-order.
    """
    assert HPO_ORDER == (
        "gamma",
        "beta",
        "k",
        "alpha",
        "m_score",
        "tau",
        "eta",
        "n_fuse_rounds",
    )
    assert HPO_ORDER.index("gamma") < HPO_ORDER.index("beta") < HPO_ORDER.index("k")
    assert HPO_ORDER.index("tau") > HPO_ORDER.index("m_score")


# =========================================================== records
def test_result_status_controls_aggregation() -> None:
    """Only ``ok`` rows may be aggregated; ``skipped`` and ``failed`` must not be."""
    ok = Result("d", "e", Status.OK, nll=1.0)
    skipped = Result("d", "e", Status.SKIPPED, reason="backend_unavailable")
    failed = Result("d", "e", Status.FAILED, reason="boom")
    assert ok.aggregated() and ok.is_ok
    assert not skipped.aggregated()
    assert not failed.aggregated()
    # And a skipped row carries no numbers at all -- not zeros, not nans.
    assert skipped.nll is None and skipped.trustworthiness is None


def test_report_counts_and_filters() -> None:
    """The report separates measured rows from skipped ones."""
    rows = (
        Result("d", "a", Status.OK, nll=1.0),
        Result("d", "b", Status.SKIPPED, reason="backend_unavailable"),
        Result("d", "c", Status.FAILED, reason="boom"),
    )
    report = Report(results=rows, n_seeds=3, n_skipped=1, n_failed=1)
    assert len(report.ok_results()) == 1
    assert report.skipped_datasets() == ("d",)
    payload = report.to_dict()
    assert payload["n_skipped"] == 1
    assert payload["results"][1]["nll"] is None


def test_report_serialises_to_json() -> None:
    """The report goes into ``benchmark.json`` verbatim."""
    import json

    report = Report(
        results=(Result("d", "a", Status.OK, nll=1.5, nll_std=0.1, params={"k": 3}),),
        n_seeds=3,
        n_skipped=0,
        n_failed=0,
    )
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["results"][0]["nll"] == 1.5
    assert payload["results"][0]["status"] == "ok"


def test_dataset_record_requires_a_name() -> None:
    """A nameless dataset would produce an unidentifiable report row."""
    with pytest.raises(ValueError, match="name"):
        Dataset(name="", n_train=10, n_test=5, d=2, intrinsic_dim=2, seed=0)


def test_fingerprint_is_stable_and_content_addressed() -> None:
    """Digests must be deterministic, layout-independent, and sensitive to value."""
    a = np.arange(12.0).reshape(3, 4)
    b = np.ascontiguousarray(a.T.T)  # same values, possibly different layout
    assert array_fingerprint(a) == array_fingerprint(b)
    c = a.copy()
    c[0, 0] = 1e-300  # a tiny change must change the digest
    assert array_fingerprint(a) != array_fingerprint(c)
    assert array_fingerprint(a) != array_fingerprint(a.reshape(4, 3))
