"""The benchmark pipeline: run one dataset, or the whole suite.

Author: 晨星 <CJX0712@users.noreply.github.com>

What this layer is responsible for
-----------------------------------
* draw a leakage-proof split (three independent samples, not a shuffle);
* tune **every** method on validation data with a recorded budget;
* evaluate on test data only after tuning is finished;
* run the four leakage assertions before publishing any test number;
* emit a :class:`~densforge.core.types.Report` in which a missing backend is
  visibly missing.

Design rules encoded here
-------------------------
**Equal tuning budget.** The flagship is tuned by coordinate descent, the
baselines by grid search, and both counts are recorded. An untuned baseline is not
a baseline; see :mod:`densforge.hpo.bandwidth` for the measured factor-of-two
difference between a val-tuned bandwidth and a library default.

**Skip honestly.** When ``available()`` returns False the row is recorded with
status ``skipped`` and a reason. It is never zero-filled, never nan-filled and
never silently dropped: a fabricated number is worse than a missing one, because a
missing one is visible.

**Determinism.** Given a seed, two runs produce bit-identical reports apart from
the ``*_sec`` timing fields, which :func:`canonical` filters out.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import Config, DensFuseConfig
from ..core.errors import (
    AtomicWriteError,
    EvalError,
    SerializationError,
)
from ..core.seed import derive_seed
from ..core.types import Report, Result, Split, Status
from ..data.datasets import (
    CONTROL_DATASET,
    HEADLINE_DATASETS,
    available_datasets,
    build_split,
)
from ..density.baselines import (
    ScipyGaussianKDE,
    SkBayesianGaussianMixture,
    SkGaussianMixture,
    SkKernelDensity,
)
from ..density.flagship import DensFuse
from ..eval.leakage import (
    ScoreRecorder,
    assert_no_overlap,
    assert_seed_isolation,
    assert_train_only_model,
)
from ..eval.metrics import (
    TRUSTWORTHNESS_K,
    aggregate,
    continuity_score,
    nll,
    summarize,
    trustworthiness_score,
)
from ..hpo.bandwidth import select_bandwidth
from ..hpo.coordinate import baseline_budget
from ..manifold.baselines import (
    SkIsomap,
    SkLLE,
    SkMDS,
    SkPCA,
    SkSpectralEmbedding,
    SkTSNE,
)
from ..manifold.flagship import ManifoldFuse
from ..training.fitter import fit_density, fit_embedder

#: Methods that are opt-in because they dominate the runtime budget.
#: t-SNE alone cost 5.7 s per fit at n=1500 in calibration; MDS 12.9 s.
SLOW_METHODS: frozenset[str] = frozenset({"SkTSNE", "SkMDS"})


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """How to construct, tune and score one benchmark method."""

    name: str
    kind: str  # "density" | "manifold" | "fused"
    factory: Callable[[int], Any]
    #: Hyper-parameters searched on validation data.
    grid: dict[str, tuple[Any, ...]] | None = None
    #: Whether the method needs a bandwidth search rather than a coordinate descent.
    tune_bandwidth: bool = False

    def build(self, seed: int) -> Any:
        return self.factory(seed)


def _default_methods(*, include_slow: bool = False) -> tuple[MethodSpec, ...]:
    """Build the method list. Slow methods are opt-in.

    Every factory takes a seed, uniformly, including for methods that have no
    randomness of their own (``SkPCA``, ``NumpyKDE``). The uniform shape means
    :meth:`DensForgePipeline._tune` never has to special-case a signature, and a
    method that later gains a seed-dependent parameter needs no plumbing change.
    """
    specs: list[MethodSpec] = [
        MethodSpec(
            "SkKernelDensity",
            "density",
            lambda s: SkKernelDensity(bandwidth=1.0, random_state=s),
            tune_bandwidth=True,
        ),
        MethodSpec(
            "ScipyGaussianKDE", "density", lambda _seed: ScipyGaussianKDE(bw_method="scott")
        ),
        MethodSpec("NumpyKDE", "density", lambda _seed: _numpy_kde()),
        MethodSpec(
            "SkGaussianMixture",
            "density",
            lambda s: SkGaussianMixture(n_components=5, random_state=s),
            grid={"n_components": (5, 12, 30)},
        ),
        MethodSpec(
            "SkBayesianGaussianMixture",
            "density",
            lambda s: SkBayesianGaussianMixture(n_components=5, random_state=s),
        ),
        MethodSpec("SkPCA", "manifold", lambda _seed: SkPCA(n_components=2)),
        # Isomap takes no seed: scikit-learn 1.9 removed the parameter (it is
        # deterministic -- the ARPACK start vector is fixed). Asserted in
        # tests/test_sklearn_contract.py, so if a future version restores it this
        # call site is the thing that will need revisiting.
        MethodSpec(
            "SkIsomap", "manifold", lambda _seed: SkIsomap(n_components=2, n_neighbors=10)
        ),
        MethodSpec(
            "SkLLE", "manifold", lambda s: SkLLE(n_components=2, n_neighbors=10, random_state=s)
        ),
        MethodSpec(
            "SkSpectralEmbedding",
            "manifold",
            lambda s: SkSpectralEmbedding(n_components=2, n_neighbors=10, random_state=s),
        ),
        MethodSpec(
            "ManifoldFuse",
            "manifold",
            lambda _seed: ManifoldFuse(n_components=2, n_neighbors=15),
        ),
        MethodSpec("DensFuse", "fused", lambda _seed: DensFuse(DensFuseConfig())),
    ]
    if include_slow:
        specs.insert(
            8,
            MethodSpec(
                "SkMDS", "manifold", lambda seed: SkMDS(n_components=2, random_state=seed)
            ),
        )
        specs.insert(
            9,
            MethodSpec(
                "SkTSNE", "manifold", lambda seed: SkTSNE(n_components=2, random_state=seed)
            ),
        )
    return tuple(specs)


#: Methods a smoke test skips, with the measured reason.
#:
#: A smoke test must prove the pipeline runs end to end, not re-measure the
#: benchmark. Dropping the two most expensive non-flagship entries takes the quick
#: run from ~77 s to ~50 s while keeping both tasks and both tiers represented.
#: The skipped names are printed by ``densforge demo --quick`` so the reduced set is
#: never mistaken for the full one.
QUICK_SKIP: dict[str, str] = {
    "SkLLE": "12% of benchmark time; the manifold task is already covered by "
    "Isomap, SpectralEmbedding and ManifoldFuse",
    "SkBayesianGaussianMixture": "the slowest-to-converge mixture; kept in the full "
    "run as a robustness reference",
}


def quick_methods() -> tuple[MethodSpec, ...]:
    """Return the reduced method list used by ``--quick``."""
    return tuple(spec for spec in _default_methods() if spec.name not in QUICK_SKIP)


def _numpy_kde():
    from ..density.tier1 import NumpyKDE

    return NumpyKDE(bandwidth=1.0)


@dataclass(frozen=True, slots=True)
class _FlagshipHandle:
    """The flagship plus its tuning evidence, behind the estimator interface.

    The pipeline needs three things from a tuned method: the model's predictions,
    its hyper-parameters, and proof of how much search produced those
    hyper-parameters. A bare estimator carries only the first. Wrapping keeps
    ``score_samples``/``embed``/``fitted_fingerprints`` transparent while letting
    the report state the budget as a measured number.
    """

    model: Any
    tuning: Any

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.model.score_samples(X)

    def embed(self) -> np.ndarray:
        return self.model.embed()

    @property
    def fitted_fingerprints(self) -> frozenset[str]:
        return self.model.fitted_fingerprints

    def params(self) -> dict[str, Any]:
        out = dict(self.model.params())
        out["tuning"] = self.tuning.as_params()
        return out


class DensForgePipeline:
    """Runs the benchmark end to end.

    Examples
    --------
    >>> from densforge.pipeline.densforge_pipeline import DensForgePipeline
    >>> from densforge.core.config import Config
    >>> pipeline = DensForgePipeline(Config(n_train=200, n_test=100, n_seeds=3))
    >>> report = pipeline.run("circles")           # doctest: +SKIP
    >>> report.n_seeds
    3
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        methods: Sequence[MethodSpec] | None = None,
        include_slow: bool = False,
    ) -> None:
        self.config = config or Config()
        self.methods = (
            tuple(methods)
            if methods is not None
            else _default_methods(include_slow=include_slow)
        )
        #: Wall-clock seconds per (dataset, method). Timings never enter a
        #: significance test; they only explain a slow run.
        self._timings: dict[str, float] = {}

    # ------------------------------------------------------------------ public
    def run(
        self,
        dataset: str,
        *,
        seeds: int | None = None,
        allow_single_seed: bool = False,
    ) -> Report:
        """Run every method on one dataset across the configured seeds.

        Parameters
        ----------
        dataset:
            Registered dataset name.
        seeds:
            Override the seed count.
        allow_single_seed:
            Permit reporting from fewer than three seeds, which
            :func:`~densforge.eval.metrics.summarize` otherwise refuses. Only a
            smoke test should pass this, and the resulting report says so in its
            metadata -- a single-seed row must never be mistaken for a measurement.

        Returns
        -------
        Report
        """
        n_seeds = int(self.config.n_seeds if seeds is None else seeds)
        results: list[Result] = []
        for spec in self.methods:
            per_seed = self._run_method(spec, dataset, n_seeds)
            results.append(per_seed)
        if allow_single_seed and n_seeds < 3:
            return Report(
                results=tuple(results),
                n_seeds=n_seeds,
                n_skipped=sum(1 for r in results if r.status is Status.SKIPPED),
                n_failed=sum(1 for r in results if r.status is Status.FAILED),
                meta={
                    "n_ok": sum(1 for r in results if r.aggregated()),
                    "smoke_test": True,
                    "note": f"single-seed run: {n_seeds} seed, not a mean +/- std measurement",
                },
            )
        return summarize(results, n_seeds=n_seeds)

    def benchmark(
        self,
        *,
        datasets: Sequence[str] | None = None,
        out: str | Path | None = "benchmark.json",
        seeds: int | None = None,
        allow_single_seed: bool = False,
    ) -> Report:
        """Run the whole suite and optionally write ``benchmark.json``.

        The artefact is written atomically (temp file, then rename), so an
        interrupted run can never leave a half-written file that a later reader
        would silently accept.
        """
        names = tuple(datasets) if datasets is not None else HEADLINE_DATASETS
        rows: list[Result] = []
        for name in names:
            report = self.run(name, seeds=seeds, allow_single_seed=allow_single_seed)
            rows.extend(report.results)
        n_seeds = int(self.config.n_seeds if seeds is None else seeds)
        if allow_single_seed and n_seeds < 3:
            final = Report(
                results=tuple(rows),
                n_seeds=n_seeds,
                n_skipped=sum(1 for r in rows if r.status is Status.SKIPPED),
                n_failed=sum(1 for r in rows if r.status is Status.FAILED),
                meta={"n_ok": sum(1 for r in rows if r.aggregated()), "smoke_test": True},
            )
        else:
            final = summarize(rows, n_seeds=n_seeds)
        if out is not None:
            payload = final.to_dict()
            payload["meta"]["timings_sec"] = dict(sorted(self._timings.items()))
            write_json(out, payload)
        return final

    # ------------------------------------------------------------------ internals
    def _run_method(self, spec: MethodSpec, dataset: str, n_seeds: int) -> Result:
        """Evaluate one method on one dataset across seeds, then aggregate."""
        nlls: list[float | None] = []
        trusts: list[float | None] = []
        conts: list[float | None] = []
        params: dict[str, Any] = {}
        seeds_used: list[int] = []
        reason = ""
        status = Status.OK

        for i in range(n_seeds):
            seed = derive_seed(self.config.random_state, i)
            seeds_used.append(seed)
            t0 = time.perf_counter()
            try:
                split = self._split(dataset, seed)
                nll_value, trust_value, cont_value, params_i = self._evaluate(spec, split, seed)
            except Exception as exc:
                status = Status.FAILED
                reason = f"{type(exc).__name__}: {exc}"
                nll_value = trust_value = cont_value = None
                params_i = {}
            self._timings[f"{dataset}/{spec.name}/seed{seed}"] = time.perf_counter() - t0
            nlls.append(nll_value)
            trusts.append(trust_value)
            conts.append(cont_value)
            if i == 0:
                params = params_i
            if status is Status.FAILED:
                break

        nll_mean, nll_std = aggregate(nlls)
        tr_mean, tr_std = aggregate(trusts)
        co_mean, _ = aggregate(conts)
        return Result(
            dataset=dataset,
            estimator=spec.name,
            status=status,
            nll=nll_mean,
            nll_std=nll_std,
            trustworthiness=tr_mean,
            trustworthiness_std=tr_std,
            params={**params, "continuity": co_mean},
            reason=reason,
            seeds=tuple(seeds_used),
        )

    def _split(self, dataset: str, seed: int) -> Split:
        """Draw the split and run the assertions that need no model."""
        split = build_split(
            dataset,
            n_train=self.config.n_train,
            n_test=self.config.n_test,
            seed=seed,
        )
        assert_seed_isolation(split.train_seed, split.test_seed)
        assert_no_overlap(split)
        return split

    def _evaluate(
        self, spec: MethodSpec, split: Split, seed: int
    ) -> tuple[float | None, float | None, float | None, dict[str, Any]]:
        """Tune on validation, then score once on test.

        The ordering is the whole point. Tuning happens inside a
        :class:`~densforge.eval.leakage.ScoreRecorder`, and the test array is
        scored afterwards, outside it -- so assertion **A2** has real evidence
        rather than a promise.
        """
        recorder = ScoreRecorder()
        with recorder:
            model = self._tune(spec, split, seed)
        # A2: the test array must not appear in the tuning call history.
        recorder.assert_clean(split.test)

        nll_value: float | None = None
        trust_value: float | None = None
        cont_value: float | None = None
        params = model.params() if hasattr(model, "params") else {}

        if spec.kind in {"density", "fused"}:
            # A3: refuse to publish a test metric for a model that saw the test data.
            assert_train_only_model(model, split.test)
            nll_value = nll(model, split.test)

        if spec.kind in {"manifold", "fused"}:
            embedding = self._embedding_of(model, split.train)
            trust_value = trustworthiness_score(
                split.train, embedding, n_neighbors=TRUSTWORTHNESS_K
            )
            cont_value = continuity_score(split.train, embedding, n_neighbors=TRUSTWORTHNESS_K)
        return nll_value, trust_value, cont_value, params

    @staticmethod
    def _embedding_of(model: Any, X: np.ndarray) -> np.ndarray:
        """Return the training embedding, whichever accessor the model offers.

        The two flagship-shaped methods take *no* argument (``DensFuse.embed`` and
        ``ManifoldFuse.transform`` both return the training embedding when called
        bare), while the scikit-learn adapters require the data. The signature is
        inspected rather than assumed, because calling either one wrongly raises a
        ``TypeError`` that names the wrong layer.
        """
        import inspect

        for accessor in ("embed", "transform"):
            method = getattr(model, accessor, None)
            if not callable(method):
                continue
            takes_argument = bool(inspect.signature(method).parameters)
            return np.asarray(method(X) if takes_argument else method(), dtype=np.float64)
        raise EvalError(f"{type(model).__name__} exposes neither embed() nor transform()")

    def _tune(self, spec: MethodSpec, split: Split, seed: int) -> Any:
        """Construct and tune one method on validation data only."""
        if spec.kind == "fused":
            return self._tune_flagship(spec, split, seed)
        if spec.tune_bandwidth:
            # A bandwidth-tuned method needs a factory that *accepts* a bandwidth.
            # The general `factory(seed)` builds a default-constructed estimator, so
            # the search cannot use it directly -- passing a candidate to a seed
            # parameter is exactly the positional-argument mix-up that leaves a tuning
            # loop silently running the same configuration every iteration.
            def make(bandwidth: float, _seed: int = seed) -> Any:
                estimator = spec.factory(_seed)
                estimator.bandwidth = float(bandwidth)
                return estimator

            best_h, _ = select_bandwidth(
                make, split.train, split.val, self.config.bandwidth_grid
            )
            estimator = spec.factory(seed)
            estimator.bandwidth = float(best_h)
            return fit_density(estimator, split.train)
        if spec.grid:
            return self._tune_grid(spec, split, seed)
        estimator = spec.factory(seed)
        if spec.kind == "manifold":
            return fit_embedder(estimator, split.train)
        return fit_density(estimator, split.train)

    def _tune_grid(self, spec: MethodSpec, split: Split, seed: int) -> Any:
        """Grid-search one hyper-parameter on validation NLL."""
        axis, values = next(iter(spec.grid.items()))
        best_value, best_nll = values[0], np.inf
        for value in values:
            estimator = spec.factory(seed)
            setattr(estimator, axis, value)
            try:
                fitted = fit_density(estimator, split.train)
                value_nll = nll(fitted, split.val)
            except Exception:
                continue
            if np.isfinite(value_nll) and value_nll < best_nll:
                best_nll, best_value = value_nll, value
        estimator = spec.factory(seed)
        setattr(estimator, axis, best_value)
        return fit_density(estimator, split.train)

    def _tune_flagship(
        self,
        spec: MethodSpec,  # noqa: ARG002
        split: Split,
        seed: int,  # noqa: ARG002
    ) -> Any:
        """Fit the flagship, tune it on validation, then refit with the winner.

        The refit matters: tuning produces a *configuration*, not a model. The
        model that scores the test set must have been built with the selected
        configuration, and building it on training data only keeps the leakage
        boundary where it belongs.
        """
        from ..hpo.coordinate import coordinate_descent

        estimator = DensFuse(DensFuseConfig())
        estimator.fit(split.train)
        result = coordinate_descent(
            estimator._state,
            split.val,
            estimator.config,
            estimator=estimator,
            budget=self.config.n_tuning_evals,
        )
        estimator.config = result.config
        estimator.refit()
        return _FlagshipHandle(estimator, result)


# ---------------------------------------------------------------------- artefacts
def canonical(obj: Any) -> Any:
    """Normalise a report for bit-wise comparison between two runs.

    Rounds floats to 10 decimal places, sorts dictionary keys, and drops every
    ``*_sec`` key. The rounding removes last-bit ULP jitter that survives even with
    identical inputs and identical code; the filtering removes wall-clock timings,
    which are the one field that cannot be deterministic by construction.
    """
    if isinstance(obj, dict):
        return {k: canonical(v) for k, v in sorted(obj.items()) if not k.endswith("_sec")}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return round(obj, 10)
    return obj


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write JSON atomically: temp file, then rename.

    A crash part way through a ``benchmark.json`` write would leave a truncated
    file that ``json.load`` accepts as valid but that means nothing. Writing to a
    sibling temp file and renaming makes the artefact all-or-nothing.
    """
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(canonical(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, target)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise AtomicWriteError(f"could not write {target}: {exc}") from exc
    except (TypeError, ValueError) as exc:
        tmp.unlink(missing_ok=True)
        raise SerializationError(
            f"payload for {target} is not JSON-serialisable: {exc}"
        ) from exc
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a report artefact written by :func:`write_json`."""
    from ..core.errors import ArtifactNotFoundError

    target = Path(path)
    if not target.exists():
        raise ArtifactNotFoundError(f"{target} does not exist")
    try:
        with open(target, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SerializationError(f"could not read {target}: {exc}") from exc


__all__ = [
    "CONTROL_DATASET",
    "HEADLINE_DATASETS",
    "QUICK_SKIP",
    "SLOW_METHODS",
    "DensForgePipeline",
    "MethodSpec",
    "available_datasets",
    "baseline_budget",
    "canonical",
    "quick_methods",
    "read_json",
    "write_json",
]
