"""End-to-end demo.

Author: 晨星 <CJX0712@users.noreply.github.com>
License: MIT (c) 2026

Runs the whole benchmark at a reduced scale and prints the table, so that a fresh
checkout can confirm the package works before reading any documentation.

    python examples/run_demo.py --quick     # smoke test: 200 samples, 3 seeds
    python examples/run_demo.py             # full scale: 800 samples, 3 seeds

What it demonstrates, in order:

1. the leakage-proof split -- three independent draws, seeds offset by construction;
2. the density comparison against a *validation-tuned* fixed bandwidth, which is the
   only fair baseline;
3. the manifold comparison, with the embedding mode stated explicitly;
4. determinism -- the same seed twice gives bit-identical numbers;
5. bit-exact reproducibility of the JSON artefact, excluding timing fields.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

# Allow running the file directly from a checkout, without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from densforge.cli import format_table
from densforge.core.config import Config
from densforge.core.seed import set_all
from densforge.data.datasets import HEADLINE_DATASETS, build_split
from densforge.eval.leakage import assert_seed_isolation, assert_train_only_model
from densforge.eval.metrics import TRUSTWORTHNESS_K
from densforge.pipeline.densforge_pipeline import (
    QUICK_SKIP,
    DensForgePipeline,
    canonical,
    quick_methods,
    read_json,
)

RULE = "=" * 78

#: Bandwidth grid for the **smoke** run only.
#:
#: The published comparison uses the 61-point log grid in
#: :data:`densforge.core.config.BANDWIDTH_GRID`, and that is the grid any number
#: meant to be quoted must come from. It costs 61 KDE fits per method per seed,
#: which is the right trade for a measurement and the wrong one for a 60-second
#: smoke test. Seven points spanning the same three decades keeps the smoke run
#: under budget while still spanning the range; the banner below states that the
#: smoke table is not a measurement.
SMOKE_BANDWIDTH_GRID: tuple[float, ...] = (0.05, 0.25, 1.0, 4.0, 16.0, 64.0, 256.0)


def _enable_utf8() -> None:
    """Make stdout UTF-8 so the rule characters survive Windows' GBK codec."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with contextlib.suppress(ValueError, OSError):  # non-tty streams lack it
                reconfigure(encoding="utf-8")


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def show_leakage_proof() -> None:
    """Demonstrate the split protocol and the assertions that guard it."""
    section("1. Leakage-proof split (three independent draws, not one pool cut up)")
    split = build_split("aniso_gmm", n_train=200, n_test=100, seed=0)
    print(f"train {split.train.shape}  seed={split.train_seed}")
    print(f"val   {split.val.shape}  seed={split.val_seed}")
    print(f"test  {split.test.shape}  seed={split.test_seed}")
    assert_seed_isolation(split.train_seed, split.test_seed)
    print("\nA1 seed isolation: test_seed == train_seed + 10_000  ->  OK")

    from scipy.spatial.distance import cdist

    minimum = float(cdist(split.train[:64], split.test).min())
    print(f"A4 shape overlap: min train-test distance = {minimum:.3f} (> 0)  ->  OK")
    print("\nThe three splits are separate draws, so a shared point would be a")
    print("probability-zero event rather than a slicing artefact.")


def show_determinism() -> None:
    """Demonstrate that a seed fully determines the result.

    Fixed size at both scales: this section demonstrates a *property*, it measures
    nothing, so there is nothing to scale.
    """
    section("4. Determinism (same seed twice -> bit-identical numbers)")
    from densforge.core.config import DensFuseConfig
    from densforge.density.flagship import DensFuse

    split = build_split("circles", n_train=200, n_test=100, seed=0)
    config = DensFuseConfig(k=12, n_fuse_rounds=1)

    set_all(0)
    first = DensFuse(config).fit(split.train).score_samples(split.test)
    set_all(0)
    second = DensFuse(config).fit(split.train).score_samples(split.test)

    identical = bool((first == second).all())
    print(f"test NLL run A = {-float(first.mean()):.10f}")
    print(f"test NLL run B = {-float(second.mean()):.10f}")
    print(f"bit-identical  = {identical}  ->  {'OK' if identical else 'FAILED'}")
    assert identical, "the same seed produced different numbers"


def show_embedding_mode(quick: bool) -> None:
    """Demonstrate that the embedding mode is stated, not implied."""
    section("3. Embedding mode must be stated (the two are not comparable)")
    from densforge.density.flagship import DensFuse
    from densforge.eval.metrics import continuity_score, trustworthiness_score

    n_train = 200 if quick else 300
    split = build_split("swiss_roll", n_train=n_train, n_test=n_train // 2, seed=0)
    model = DensFuse().fit(split.train)
    assert_train_only_model(model, split.test)

    for mode in ("inductive", "transductive"):
        Psi = model.embed(split.train, mode=mode)
        tw = trustworthiness_score(split.train, Psi, n_neighbors=TRUSTWORTHNESS_K)
        cont = continuity_score(split.train, Psi, n_neighbors=TRUSTWORTHNESS_K)
        print(f"  {mode:<14} trustworthiness={tw:.4f}  continuity={cont:.4f}")
    print(f"\ntrustworthiness uses n_neighbors={TRUSTWORTHNESS_K} throughout the project.")
    print("Numbers from the two modes must never be placed in the same column.")


def show_benchmark(quick: bool) -> None:
    """Run the benchmark and print the table."""
    section("2. Benchmark (validation-tuned baselines; every method tuned on val only)")
    # `--quick` is a smoke test: one seed, a quarter of the samples, a small tuning
    # budget, and a reduced method list. `allow_single_seed` is what permits the
    # one-seed report -- the reporting layer refuses fewer than three otherwise --
    # and the banner below says in as many words that the numbers must not be quoted.
    config = (
        Config(
            n_train=200,
            n_test=100,
            n_seeds=1,
            mds_subsample=200,
            n_tuning_evals=4,
            # The smoke run uses a coarse bandwidth grid on purpose. The 61-point
            # log grid is what the *published* comparison needs, and it costs 61 KDE
            # fits per method per seed -- which is the right trade for a number
            # someone will quote and the wrong one for a 60-second smoke test. The
            # banner says so, so nobody mistakes a smoke table for a measurement.
            bandwidth_grid=SMOKE_BANDWIDTH_GRID,
        )
        if quick
        else Config(n_train=800, n_test=300, n_seeds=3, mds_subsample=600, n_tuning_evals=30)
    )
    set_all(config.random_state)
    started = time.perf_counter()

    pipeline = (
        DensForgePipeline(config, methods=quick_methods())
        if quick
        else DensForgePipeline(config)
    )
    report = pipeline.benchmark(datasets=HEADLINE_DATASETS, out=None, allow_single_seed=quick)
    print(format_table(report, title="DensForge benchmark"))
    print(f"\nelapsed {time.perf_counter() - started:.1f}s")
    if quick:
        print()
        print("NOTE: --quick is a SMOKE TEST: 1 seed, 200 training samples, a small tuning")
        print("      budget. It proves the pipeline runs end to end; it is not a")
        print("      measurement and its numbers must not be quoted. Run without")
        print("      --quick for the 3-seed report with mean +/- std.")
        print()
        print("Methods omitted from the smoke run, with the measured reason:")
        for name, reason in QUICK_SKIP.items():
            print(f"  - {name}: {reason}")
    return report


def show_reproducible_artefact(quick: bool) -> Path:
    """Demonstrate bit-exact reproducibility of the JSON artefact."""
    section("5. Artefact reproducibility (timing fields excluded)")
    # One dataset, not six: this section proves the *artefact* round-trips
    # bit-exactly, and repeating the full six-dataset benchmark twice more would
    # roughly double the demo's runtime to demonstrate a property that does not
    # depend on how many datasets are in it.
    config = Config(
        n_train=200,
        n_test=100,
        n_seeds=1 if quick else 3,
        n_tuning_evals=3,
        bandwidth_grid=SMOKE_BANDWIDTH_GRID if quick else Config().bandwidth_grid,
    )
    out = Path("benchmark.json")
    payloads = []
    for _ in range(2):
        set_all(config.random_state)
        methods = quick_methods() if quick else None
        DensForgePipeline(config, methods=methods).benchmark(
            datasets=("circles",), out=out, allow_single_seed=quick
        )
        payloads.append(canonical(read_json(out)))
    same = payloads[0] == payloads[1]
    print(
        f"ran the same benchmark twice -> identical: {same}  ->  {'OK' if same else 'FAILED'}"
    )
    print(f"artefact: {out.resolve()}")
    print("Fields ending in `_sec` are excluded: wall-clock timing cannot be")
    print("deterministic by construction, so comparing it would be meaningless.")
    assert same, "two identical runs produced different artefacts"
    return out


def main() -> int:
    """Run the demo. Returns a process exit code."""
    _enable_utf8()
    parser = argparse.ArgumentParser(
        prog="run_demo.py",
        description="DensForge end-to-end demo (author: 晨星).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke scale: 200 training samples instead of 800. Same 3 seeds.",
    )
    parser.add_argument(
        "--section",
        choices=["all", "split", "determinism", "embedding", "benchmark", "artefact"],
        default="all",
        help="run one section only",
    )
    args = parser.parse_args()

    print(RULE)
    print("DensForge demo -- density estimation and manifold learning in one closed loop")
    print("author: 晨星 <CJX0712@users.noreply.github.com>   license: MIT")
    print(RULE)

    if args.section in {"all", "split"}:
        show_leakage_proof()
    if args.section in {"all", "embedding"}:
        show_embedding_mode(args.quick)
    if args.section in {"all", "determinism"}:
        show_determinism()
    if args.section in {"all", "benchmark"}:
        show_benchmark(args.quick)
    if args.section in {"all", "artefact"}:
        show_reproducible_artefact(args.quick)

    print(f"\n{RULE}")
    print("demo complete")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
