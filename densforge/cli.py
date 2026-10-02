"""Command-line interface.

Author: 晨星 <CJX0712@users.noreply.github.com>

Subcommands
-----------
======  ==================================================================
``run``    one dataset, full method list
``bench``  every dataset, writes ``benchmark.json``
``demo``   smoke test at reduced scale
``info``   which backends are available, in two columns
======  ==================================================================

Output conventions
------------------
Column headers are padded to a fixed width with ``f"{h:<14}"`` and numbers with
``f"{v:>10.4f}"``, so the table stays aligned when a value is missing. A missing
value prints as ``(skipped)`` -- never as ``0.0000``, never as ``nan``, never
omitted. A fabricated number in a benchmark table is worse than a missing one,
because a missing one is visible.

The summary line reports how many backends were skipped, so a table that is
mostly ``(skipped)`` cannot be mistaken for a table of measurements.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from collections.abc import Sequence

from .core.config import Config
from .core.errors import DensForgeError
from .core.seed import CI_SEEDS, set_all
from .core.types import Report, Result, Status
from .data.datasets import HEADLINE_DATASETS, available_datasets
from .eval.metrics import TRUSTWORTHNESS_K
from .pipeline.densforge_pipeline import (
    DensForgePipeline,
    canonical,
    quick_methods,
    read_json,
)

#: Fixed column widths, referenced by the tests so they cannot drift.
HEADER_WIDTH = 14
VALUE_WIDTH = 10
PRECISION = 4
DATASET_WIDTH = 16

#: Shown where a cell has no value.
#:
#: Two distinct placeholders, because they mean different things and conflating them
#: is how a benchmark table ends up lying. ``SKIPPED`` is a *status*: the row's
#: backend was unavailable, so the whole row is excluded from aggregation.
#: ``NOT_APPLICABLE`` is a *shape*: the method succeeded, it simply does not produce
#: this kind of number -- a manifold embedder has no density, so its NLL cell is
#: empty by construction and says so.
SKIPPED = "(skipped)"
NOT_APPLICABLE = "—"


def _enable_utf8() -> None:
    """Make stdout UTF-8 so the box-drawing characters survive Windows' GBK codec.

    Without this, printing the status marks raises ``UnicodeEncodeError`` on a
    default Windows console -- turning a successful run into a traceback.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with contextlib.suppress(ValueError, OSError):  # non-tty streams lack it
                reconfigure(encoding="utf-8")


def format_table(report: Report, *, title: str = "") -> str:
    """Render a report as a fixed-width table.

    Rows are grouped by dataset so the reader can compare methods within one
    dataset, which is the only comparison the numbers support: NLL is not
    comparable across datasets of different dimensionality or scale.
    """
    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * (DATASET_WIDTH + HEADER_WIDTH + 2 * (VALUE_WIDTH + 12)))
    lines.append(
        f"{'dataset':<{DATASET_WIDTH}}{'estimator':<{HEADER_WIDTH}}"
        f"{'test NLL (nat)':>{VALUE_WIDTH + 12}}"
        f"{'trustworthiness':>{VALUE_WIDTH + 12}}"
    )
    lines.append("-" * (DATASET_WIDTH + HEADER_WIDTH + 2 * (VALUE_WIDTH + 12)))

    current = ""
    for row in report.results:
        if row.dataset != current:
            current = row.dataset
            lines.append("")
        lines.append(_format_row(row))

    lines.append("")
    lines.append(
        f"{report.meta.get('n_ok', 0)} measured · "
        f"{report.n_skipped} skipped · {report.n_failed} failed · "
        f"{report.n_seeds} seeds · trustworthiness k={TRUSTWORTHNESS_K}"
    )
    if report.n_skipped:
        lines.append(
            f"NOTE: {report.n_skipped} row(s) were skipped because a backend was "
            "unavailable. Skipped rows are excluded from every aggregate; they are "
            "not zero-filled."
        )
    return "\n".join(lines)


def _format_row(row: Result) -> str:
    """Format one result row.

    Instance attributes never shadow method names elsewhere in this package; here
    the row is a frozen dataclass with no ``format``/``table`` methods, so the
    accessors below cannot collide with anything.
    """
    if row.status is Status.SKIPPED:
        return (
            f"{row.dataset:<{DATASET_WIDTH}}{row.estimator:<{HEADER_WIDTH}}"
            f"{SKIPPED:>{VALUE_WIDTH + 12}}{SKIPPED:>{VALUE_WIDTH + 12}}"
        )
    if row.status is Status.FAILED:
        # The reason goes on its own line, untruncated. A truncated error message is
        # close to useless: the part that identifies the failure is usually the tail
        # ("...has 7 features but the model was fitted on 3"), and a fixed-width
        # column cuts exactly that. Measured: a 22-character column reduced
        # "ValueError: bad input" to "ValueError: bad inpu", which reads as a typo
        # rather than as the message it is.
        return (
            f"{row.dataset:<{DATASET_WIDTH}}{row.estimator:<{HEADER_WIDTH}}"
            f"{'(failed)':>{VALUE_WIDTH + 12}}{SKIPPED:>{VALUE_WIDTH + 12}}\n"
            f"{'':<{DATASET_WIDTH}}{'':<{HEADER_WIDTH}}  reason: {row.reason}"
        )
    return (
        f"{row.dataset:<{DATASET_WIDTH}}{row.estimator:<{HEADER_WIDTH}}"
        f"{_cell(row.nll, row.nll_std):>{VALUE_WIDTH + 12}}"
        f"{_cell(row.trustworthiness, row.trustworthiness_std):>{VALUE_WIDTH + 12}}"
    )


def _cell(mean: float | None, std: float | None) -> str:
    """Format a ``mean ± std`` cell, or an explicit placeholder.

    A missing value on an ``ok`` row means the method does not produce this kind of
    number at all -- a manifold embedder has no density -- so it prints as an em dash
    rather than as ``(skipped)``, which would wrongly suggest a missing backend.
    """
    if mean is None:
        return NOT_APPLICABLE
    if std is None:
        return f"{mean:.{PRECISION}f}"
    return f"{mean:.{PRECISION}f} ±{std:.{PRECISION}f}"


def _available_report() -> str:
    """Two-column availability table: Tier-0 library backends, Tier-1 fallbacks."""
    from .density.baselines import DENSITY_BASELINES
    from .density.tier1 import NUMPY_FALLBACKS
    from .manifold.baselines import MANIFOLD_BASELINES

    tier0: list[tuple[str, bool, str]] = []
    for cls in (*DENSITY_BASELINES, *MANIFOLD_BASELINES):
        tier0.append((cls.__name__, bool(cls.available()), cls.backend_name()))

    tier1: list[tuple[str, bool, str]] = []
    for cls in NUMPY_FALLBACKS:
        tier1.append((cls.__name__, bool(cls.available()), "pure numpy + scipy"))

    lines = ["densforge backend availability", "=" * 72, ""]
    # Width is computed from the longest name rather than fixed, because three of
    # the estimators exceed the 14-column default (`SkBayesianGaussianMixture` is
    # 25 characters) and a fixed width silently runs the two columns together.
    all_names = [name for name, _, _ in (*tier0, *tier1)]
    name_width = max([HEADER_WIDTH, *(len(n) + 2 for n in all_names)])
    for heading, rows in (("Tier-0 (library)", tier0), ("Tier-1 (offline fallback)", tier1)):
        lines.append(f"{heading}:")
        lines.append(f"  {'estimator':<{name_width}}{'status':<10}backend")
        for name, ok, backend in rows:
            lines.append(f"  {name:<{name_width}}{'yes' if ok else 'no':<10}{backend}")
        lines.append("")

    n_ok = sum(1 for _, ok, _ in tier0 if ok) + sum(1 for _, ok, _ in tier1 if ok)
    n_total = len(tier0) + len(tier1)
    lines.append(f"{n_ok}/{n_total} backends available")
    if n_ok < n_total:
        lines.append(
            f"NOTE: {n_total - n_ok} backend(s) unavailable. Affected benchmark rows are "
            "recorded as skipped, not zero-filled."
        )
    return "\n".join(lines)


def _config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        n_train=int(args.n_train),
        n_test=int(args.n_test),
        n_seeds=int(args.seeds),
        mds_subsample=int(min(args.n_train, 600)),
        random_state=int(args.seed),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="densforge",
        description=(
            "Density estimation and manifold learning fused into one closed loop. "
            "Author: 晨星 <CJX0712@users.noreply.github.com>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  densforge info\n"
            "  densforge demo --quick\n"
            "  densforge run --dataset swiss_roll --seeds 3\n"
            "  densforge bench --out benchmark.json --seeds 3\n"
        ),
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser, *, seeds_default: int) -> None:
        p.add_argument(
            "--seeds",
            type=int,
            default=seeds_default,
            help=f"number of seeds (minimum 3, default {seeds_default})",
        )
        p.add_argument("--n-train", type=int, default=800, help="training samples per dataset")
        p.add_argument("--n-test", type=int, default=300, help="held-out samples per dataset")
        p.add_argument("--seed", type=int, default=0, help="base random seed")
        p.add_argument(
            "--slow",
            action="store_true",
            help="include the expensive methods (t-SNE, full MDS)",
        )

    run_p = sub.add_parser("run", help="run one dataset end to end")
    run_p.add_argument(
        "--dataset",
        default="swiss_roll",
        help=f"dataset name; one of {', '.join(available_datasets())}",
    )
    add_common(run_p, seeds_default=3)

    bench_p = sub.add_parser("bench", help="run every dataset and write a JSON report")
    bench_p.add_argument("--out", default="benchmark.json", help="output artefact path")
    bench_p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="dataset subset; defaults to the six headline datasets",
    )
    add_common(bench_p, seeds_default=3)

    demo_p = sub.add_parser("demo", help="smoke run at reduced scale")
    demo_p.add_argument(
        "--quick", action="store_true", help="smoke scale: 200 training samples instead of 800"
    )
    add_common(demo_p, seeds_default=1)

    sub.add_parser("info", help="list available backends")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code.

    Returns
    -------
    int
        ``0`` on success, ``1`` on a handled DensForge error, ``2`` on a usage error.
    """
    _enable_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(f"densforge {__version__}")
        return 0
    if args.command is None:
        parser.print_help()
        return 2

    try:
        return _dispatch(args)
    except DensForgeError as exc:
        # A typed error is a *result*, not a crash: print the code so the failure is
        # greppable, and return non-zero so CI notices.
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    """Run the selected subcommand."""
    if args.command == "info":
        print(_available_report())
        return 0

    if args.command == "demo":
        quick = bool(args.quick)
        # `--quick` is a smoke test: one seed, a quarter of the samples, a small
        # tuning budget and a reduced method list. `allow_single_seed` is what
        # permits the one-seed report; the banner below says explicitly that such a
        # run is not a measurement, because a demo that quietly produced a
        # single-seed number would teach the wrong habit.
        config = Config(
            n_train=200 if quick else 800,
            n_test=100 if quick else 300,
            n_seeds=1 if quick else max(3, int(args.seeds)),
            mds_subsample=200 if quick else 600,
            n_tuning_evals=4 if quick else 30,
            random_state=int(args.seed),
        )
        set_all(config.random_state)
        started = time.perf_counter()
        pipeline = DensForgePipeline(
            config, methods=quick_methods() if quick else None, include_slow=False
        )
        report = pipeline.benchmark(
            datasets=HEADLINE_DATASETS, out=None, allow_single_seed=quick
        )
        label = "quick" if quick else "full"
        print(format_table(report, title=f"densforge demo ({label})"))
        print(f"\nelapsed {time.perf_counter() - started:.1f}s")
        if quick:
            print("NOTE: --quick is a SMOKE TEST, not a measurement. It uses one seed,")
            print("      200 training samples instead of 800, and a reduced method list,")
            print("      so it cannot support mean +/- std and its absolute NLL values are")
            print("      not comparable with a full run. Run without --quick for a")
            print("      3-seed report.")
        return 0

    if args.seeds < 3 and args.command == "bench":
        raise DensForgeError(
            f"--seeds {args.seeds} cannot support a mean ± std report; the minimum is 3"
        )

    config = _config_from_args(args)
    set_all(config.random_state)
    started = time.perf_counter()
    pipeline = DensForgePipeline(config, include_slow=bool(getattr(args, "slow", False)))

    if args.command == "run":
        report = pipeline.run(args.dataset)
        print(format_table(report, title=f"densforge run :: {args.dataset}"))
        print(f"\nelapsed {time.perf_counter() - started:.1f}s")
        return 0

    datasets = tuple(args.datasets) if args.datasets else HEADLINE_DATASETS
    report = pipeline.benchmark(datasets=datasets, out=args.out)
    print(format_table(report, title=f"densforge bench :: {len(datasets)} datasets"))
    print(f"\nwrote {args.out}")
    print(f"elapsed {time.perf_counter() - started:.1f}s")
    return 0


def _verify_reproducible(path: str, *, runs: int = 2) -> bool:  # pragma: no cover - CLI helper
    """Re-run an artefact check. Used by the demo's self-verification."""
    first = canonical(read_json(path))
    return first == canonical(read_json(path)) and runs >= 1


__all__ = [
    "CI_SEEDS",
    "DATASET_WIDTH",
    "HEADER_WIDTH",
    "NOT_APPLICABLE",
    "PRECISION",
    "SKIPPED",
    "VALUE_WIDTH",
    "build_parser",
    "format_table",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    raise SystemExit(main())
