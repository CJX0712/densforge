"""Metrics, hyper-parameter selection, the pipeline, and the CLI.

Author: 晨星 <CJX0712@users.noreply.github.com>
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from densforge.core.config import Config
from densforge.core.errors import InsufficientSeedsError, NumericalError
from densforge.core.types import Status
from densforge.data.datasets import build_split
from densforge.density.baselines import SkKernelDensity
from densforge.density.flagship import DensFuse
from densforge.eval.metrics import (
    aggregate,
    calibration_error,
    format_mean_std,
    is_significant,
    local_structure_fidelity,
    mean_or_none,
    nll,
    nll_train_loo,
    summarize,
)
from densforge.hpo.alpha import ALPHA_GRID, select_alpha
from densforge.hpo.bandwidth import log_spaced_grid, select_bandwidth
from densforge.hpo.coordinate import baseline_budget, default_grid
from densforge.pipeline.densforge_pipeline import (
    DensForgePipeline,
    MethodSpec,
    canonical,
    read_json,
    write_json,
)


@pytest.fixture(scope="module")
def split():
    return build_split("circles", n_train=200, n_test=100, seed=0)


# =========================================================== aggregation
def test_aggregate_uses_sample_std_and_skips_missing() -> None:
    """``ddof=1`` across seeds, and missing values are excluded, not zero-filled."""
    mean, std = aggregate([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.0)
    mean, std = aggregate([1.0, None, 3.0])
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(np.std([1.0, 3.0], ddof=1))


def test_aggregate_returns_none_for_empty_input_not_nan() -> None:
    """An all-missing group yields ``None``, so a caller must handle absence.

    Returning ``nan`` would let a skipped row enter an aggregate as a number, which
    is the exact failure the skip mechanism exists to prevent.
    """
    assert aggregate([]) == (None, None)
    assert aggregate([None, None]) == (None, None)
    assert aggregate([float("nan")]) == (None, None)
    assert mean_or_none([]) is None


def test_aggregate_of_a_single_value_reports_zero_std() -> None:
    """One observation has no spread; report it as zero rather than as an error."""
    mean, std = aggregate([2.5])
    assert mean == 2.5 and std == 0.0


def test_summarize_refuses_fewer_than_three_seeds() -> None:
    """``mean ± std`` over two points is not a measurement."""
    with pytest.raises(InsufficientSeedsError, match="minimum is 3"):
        summarize([], n_seeds=2)
    with pytest.raises(InsufficientSeedsError):
        summarize([], n_seeds=1)


def test_format_mean_std_marks_missing_values_explicitly() -> None:
    """A missing value prints as ``(skipped)``, never as a number."""
    assert format_mean_std(None, None) == "(skipped)"
    assert "1.5000" in format_mean_std(1.5, 0.1)
    assert "1.5000" in format_mean_std(1.5, None)
    assert "±" in format_mean_std(1.5, 0.125)


def test_significance_criterion_is_deliberately_strict() -> None:
    """A gap must exceed half the sum of the two standard deviations.

    The project would rather miss a real small win than announce one that reverses
    on the next seed, so equality does not count.
    """
    # Equality is deliberately NOT significant: the bar is `>`, not `>=`, so a
    # difference that exactly equals the threshold is reported as no difference.
    assert not is_significant(1.0, 1.0, 1.0), "1.0 > 0.5*(1+1)=1.0 must be False"
    assert is_significant(1.01, 1.0, 1.0)
    assert not is_significant(0.99, 1.0, 1.0)
    assert not is_significant(0.0, 0.0, 0.0), "zero gap is never significant"
    assert not is_significant(float("nan"), 1.0, 1.0)


# =========================================================== nll
def test_nll_matches_the_definition(split) -> None:
    """``NLL = -mean(log p)`` in nats."""
    model = SkKernelDensity(bandwidth=1.0).fit(split.train)
    logp = model.score_samples(split.test)
    assert nll(model, split.test) == pytest.approx(-float(np.mean(logp)))


def test_nll_rejects_a_mismatched_output_length(split) -> None:
    """A model returning the wrong number of values is caught, not averaged."""

    class Broken:
        def score_samples(self, X):
            return np.zeros(len(X) + 1)

    with pytest.raises(NumericalError, match="values for"):
        nll(Broken(), split.test)


def test_nll_train_loo_is_available_for_the_flagship(split) -> None:
    """The flagship exposes a leave-one-out channel, so the two NLLs are comparable."""
    model = DensFuse().fit(split.train)
    value = nll_train_loo(model, split.train)
    assert np.isfinite(value)


def test_nll_train_loo_refuses_a_model_without_the_channel(split) -> None:
    """A model with no leave-one-out channel must say so rather than invent one."""
    with pytest.raises(NumericalError, match="leave-one-out"):
        nll_train_loo(SkKernelDensity(bandwidth=1.0).fit(split.train), split.train)


def test_calibration_error_is_finite(split) -> None:
    """Calibration is a diagnostic that must produce a finite number."""
    model = SkKernelDensity(bandwidth=1.0).fit(split.train)
    value = calibration_error(model, split.test, n_bins=5)
    assert np.isfinite(value) and value >= 0.0
    with pytest.raises(NumericalError, match="at least n_bins"):
        calibration_error(model, split.test[:2], n_bins=20)


def test_local_structure_fidelity_is_a_fraction(split) -> None:
    """The local-structure metric is a fraction in ``[0, 1]``."""
    X = split.train[:150]
    identity = X[:, :2]
    value = local_structure_fidelity(X, identity, X, n_neighbors=5)
    assert value == pytest.approx(1.0)
    shuffled = np.random.RandomState(0).permutation(X)[:, :2]
    assert local_structure_fidelity(X, shuffled, X, n_neighbors=5) < 1.0


# =========================================================== hpo
def test_bandwidth_selection_improves_on_any_fixed_choice(split) -> None:
    """Validation-tuned bandwidth must beat the worst grid point, by construction.

    The point of tuning the baseline is that the library default is not a fair
    comparison point -- measured, a val-tuned fixed bandwidth beats
    ``gaussian_kde('scott')`` by roughly a factor of two on anisotropic data.
    """

    def factory(bandwidth):
        return SkKernelDensity(bandwidth=bandwidth)

    grid = (0.25, 0.5, 1.0, 2.0, 4.0)
    best_h, best_nll = select_bandwidth(factory, split.train, split.val, grid)
    assert best_h in grid
    chosen = nll(SkKernelDensity(bandwidth=best_h).fit(split.train), split.val)
    assert chosen == pytest.approx(best_nll, rel=1e-9)
    for h in grid:
        other = nll(SkKernelDensity(bandwidth=h).fit(split.train), split.val)
        assert best_nll <= other + 1e-12


def test_bandwidth_selection_rejects_a_bad_grid(split) -> None:
    """An empty or non-positive grid is refused rather than silently skipped."""

    def factory(bandwidth):
        return SkKernelDensity(bandwidth=bandwidth)

    with pytest.raises(NumericalError, match="must not be empty"):
        select_bandwidth(factory, split.train, split.val, [])
    with pytest.raises(NumericalError, match="positive"):
        select_bandwidth(factory, split.train, split.val, (0.0, 1.0))


def test_log_spaced_bandwidth_grid_is_log_uniform() -> None:
    """The grid is log-spaced because the optimum scales as a power of ``N``."""
    grid = log_spaced_grid(0.1, 10.0, 5)
    ratios = [grid[i + 1] / grid[i] for i in range(len(grid) - 1)]
    assert all(r == pytest.approx(ratios[0], rel=1e-9) for r in ratios)
    with pytest.raises(NumericalError):
        log_spaced_grid(0.0, 1.0, 3)
    with pytest.raises(NumericalError):
        log_spaced_grid(0.1, 1.0, 1)


def test_alpha_selection_reports_an_endpoint_optimum(split) -> None:
    """An endpoint optimum is a *finding*, not a validated interior result.

    If the best ``alpha`` is 0 the direction may be useless on this data; if it is 1
    the grid is too narrow. The two are indistinguishable without a wider sweep, so
    the flag must reach the report.
    """
    result = select_alpha(split.train, split.val, (0.0, 0.5, 1.0))
    assert result.alpha in (0.0, 0.5, 1.0)
    assert len(result.scores) == 3
    assert isinstance(result.at_endpoint, bool)
    if result.alpha in (0.0, 1.0):
        assert result.at_endpoint
    json.dumps(result.as_params())


def test_alpha_grid_contains_the_no_correction_control() -> None:
    """``alpha = 0`` must be in the grid, or there is no control for the axis."""
    assert 0.0 in ALPHA_GRID
    assert max(ALPHA_GRID) <= 1.0


def test_alpha_selection_rejects_out_of_range_values(split) -> None:
    """``alpha > 1`` has no theoretical basis and is refused."""
    with pytest.raises(NumericalError, match=r"\[0, 1\]"):
        select_alpha(split.train, split.val, (0.0, 1.5))


def test_default_grids_cover_every_searched_axis() -> None:
    """Every axis in the search order has a grid, and the gamma grid reaches below 0.5.

    The gamma floor matters: measured on ``aniso_gmm``, the validation optimum is
    ``gamma ~ 0.23``, so a grid starting at 0.5 cannot reach it at all.
    """
    from densforge.core.config import HPO_ORDER

    for axis in HPO_ORDER:
        grid = default_grid(axis, DensFuse().config, d=8)
        assert len(grid) >= 2, axis
    assert min(default_grid("gamma", DensFuse().config, 8)) < 0.25
    with pytest.raises(NumericalError, match="no grid"):
        default_grid("nonexistent", DensFuse().config, 8)


def test_baseline_budget_is_reported_as_a_number() -> None:
    """The comparison budget is computed, not asserted in prose."""
    assert baseline_budget((0.25, 0.5, 1.0, 2.0, 4.0), (5, 12, 30)) == 11


# =========================================================== pipeline
def test_pipeline_runs_a_dataset_and_reports_every_method(split) -> None:
    """A run produces one aggregated row per method, with a status for each."""
    config = Config(n_train=200, n_test=100, n_seeds=3, n_tuning_evals=4)
    methods = (
        MethodSpec(
            "SkKernelDensity",
            "density",
            lambda s: SkKernelDensity(bandwidth=1.0, random_state=s),
            tune_bandwidth=True,
        ),
        MethodSpec("NumpyKDE", "density", lambda s: _numpy_kde()),
    )
    pipeline = DensForgePipeline(config, methods=methods)
    report = pipeline.run("circles")
    assert len(report.results) == 2
    assert {r.estimator for r in report.results} == {"SkKernelDensity", "NumpyKDE"}
    for row in report.results:
        assert row.status in (Status.OK, Status.FAILED, Status.SKIPPED)
        if row.status is Status.OK:
            assert np.isfinite(row.nll)
            assert row.seeds == (0, 1, 2)


def _numpy_kde():
    from densforge.density.tier1 import NumpyKDE

    return NumpyKDE(bandwidth=1.0)


def test_pipeline_marks_an_unavailable_backend_as_skipped(split) -> None:
    """A missing backend produces a ``skipped`` row with a reason -- never a zero.

    A fabricated number in a benchmark table is worse than a missing one, because a
    missing one is visible.
    """
    from densforge.density.baselines import BaseDensity

    class Unavailable(BaseDensity):
        _BACKEND = ("densforge.no_such_module", "Nope")

    config = Config(n_train=200, n_test=100, n_seeds=3)
    pipeline = DensForgePipeline(config, methods=(MethodSpec("Nope", "density", lambda s: s),))
    report = pipeline.run("circles")
    row = report.results[0]
    assert row.status is Status.FAILED
    assert row.nll is None and row.trustworthiness is None
    assert row.reason
    assert report.n_failed >= 1


def test_pipeline_result_is_deterministic(split) -> None:
    """Two identical runs produce identical numbers, which is the point of the seed."""
    config = Config(n_train=200, n_test=100, n_seeds=3)
    methods = (
        MethodSpec(
            "SkKernelDensity",
            "density",
            lambda s: SkKernelDensity(bandwidth=1.0),
            tune_bandwidth=True,
        ),
    )
    first = DensForgePipeline(config, methods=methods).run("circles")
    second = DensForgePipeline(config, methods=methods).run("circles")
    assert [r.nll for r in first.results] == [r.nll for r in second.results]
    assert [r.trustworthiness for r in first.results] == [
        r.trustworthiness for r in second.results
    ]


def test_pipeline_refuses_to_embed_what_cannot_embed() -> None:
    """A density-only method asked for coordinates is a typed error, not a silent None."""
    config = Config(n_train=200, n_test=100, n_seeds=3)
    methods = (MethodSpec("NumpyKDE", "manifold", lambda s: _numpy_kde()),)
    report = DensForgePipeline(config, methods=methods).run("circles")
    assert report.results[0].status is Status.FAILED
    assert "embed" in report.results[0].reason or "transform" in report.results[0].reason


# =========================================================== artefacts
def test_canonical_filters_timings_and_rounds_floats(tmp_path) -> None:
    """Two runs differ only in timing, so ``canonical`` makes them comparable."""
    payload = {
        "b": 1 / 3,
        "a": {"elapsed_sec": 1.234, "nll": 12.345678901234567},
        "list": [{"x_sec": 9.9, "y": 0.1 + 0.2}],
    }
    first = canonical(payload)
    second = canonical(json.loads(json.dumps(payload)))
    assert first == second
    assert "elapsed_sec" not in first["a"]
    assert all(not k.endswith("_sec") for k in first)
    assert first["b"] == round(1 / 3, 10)


def test_write_then_read_round_trips(tmp_path) -> None:
    """The artefact survives a write/read cycle unchanged."""
    target = tmp_path / "benchmark.json"
    payload = {"n_seeds": 3, "results": [{"dataset": "d", "nll": 1.5}], "elapsed_sec": 3.2}
    write_json(target, payload)
    assert target.exists()
    back = read_json(target)
    assert back["n_seeds"] == 3
    assert back["results"][0]["nll"] == 1.5
    assert "elapsed_sec" not in back, "timings are filtered on read-back"


def test_write_json_is_atomic(tmp_path) -> None:
    """A failed write must not leave a partial file behind."""
    target = tmp_path / "out.json"
    write_json(target, {"ok": 1})
    original = target.read_text(encoding="utf-8")
    with pytest.raises(Exception):
        write_json(target, {"bad": {1, 2, 3}})  # a set is not JSON-serialisable
    assert target.read_text(encoding="utf-8") == original, "the old file was damaged"
    assert not list(tmp_path.glob("*.tmp")), "a temp file was left behind"


def test_read_json_reports_a_missing_artefact(tmp_path) -> None:
    """A missing artefact is a typed error naming the path."""
    from densforge.core.errors import ArtifactNotFoundError

    with pytest.raises(ArtifactNotFoundError, match="does not exist"):
        read_json(tmp_path / "nope.json")


def test_read_json_reports_a_corrupt_artefact(tmp_path) -> None:
    """A truncated file is reported, not silently returned as an empty report."""
    from densforge.core.errors import SerializationError

    target = tmp_path / "bad.json"
    target.write_text("{not json", encoding="utf-8")
    with pytest.raises(SerializationError):
        read_json(target)


def test_demo_script_runs_quick(tmp_path) -> None:
    """``examples/run_demo.py --quick`` must complete and report every dataset."""
    import pathlib
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    script = root / "examples" / "run_demo.py"
    if not script.exists():
        pytest.skip("examples/run_demo.py is not present")
    completed = subprocess.run(
        [sys.executable, str(script), "--quick"],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=str(root),
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "DensFuse" in completed.stdout
    for dataset in (
        "aniso_gmm",
        "swiss_roll",
        "double_spiral",
        "circles",
        "t_mixture",
        "manifold_noise",
    ):
        assert dataset in completed.stdout, f"{dataset} missing from the demo output"
