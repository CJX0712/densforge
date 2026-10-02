"""The command-line interface.

Author: 晨星 <CJX0712@users.noreply.github.com>
"""

from __future__ import annotations

import json
import sys

import pytest

from densforge.cli import (
    DATASET_WIDTH,
    HEADER_WIDTH,
    NOT_APPLICABLE,
    PRECISION,
    SKIPPED,
    VALUE_WIDTH,
    build_parser,
    format_table,
    main,
)
from densforge.core.types import Report, Result, Status


def _report(*rows: Result) -> Report:
    n_skipped = sum(1 for r in rows if r.status is Status.SKIPPED)
    n_failed = sum(1 for r in rows if r.status is Status.FAILED)
    return Report(
        results=rows,
        n_seeds=3,
        n_skipped=n_skipped,
        n_failed=n_failed,
        meta={"n_ok": len(rows) - n_skipped - n_failed},
    )


# =========================================================== formatting
def test_column_widths_are_the_documented_constants() -> None:
    """The widths are named constants so they cannot drift between call sites."""
    assert HEADER_WIDTH == 14
    assert VALUE_WIDTH == 10
    assert PRECISION == 4
    assert DATASET_WIDTH == 16
    assert SKIPPED == "(skipped)"
    assert NOT_APPLICABLE == "\u2014", "a shape-missing cell must not read as a status"


def test_table_aligns_columns() -> None:
    """Every row's estimator column starts at the same offset."""
    report = _report(
        Result(
            "aniso_gmm",
            "SkKernelDensity",
            Status.OK,
            nll=12.1613,
            nll_std=0.31,
            trustworthiness=0.9,
            trustworthiness_std=0.01,
        ),
        Result(
            "aniso_gmm",
            "DensFuse",
            Status.OK,
            nll=10.0234,
            nll_std=0.25,
            trustworthiness=0.95,
            trustworthiness_std=0.005,
        ),
    )
    lines = [
        ln
        for ln in format_table(report).splitlines()
        if "SkKernelDensity" in ln or "DensFuse" in ln
    ]
    assert len(lines) == 2
    # The estimator name begins at the same column in both rows.
    starts = {
        ln.index("SkKernelDensity") if "SkKernelDensity" in ln else ln.index("DensFuse")
        for ln in lines
    }
    assert len(starts) == 1, starts


def test_missing_values_print_as_skipped_not_as_zero() -> None:
    """A missing number prints as ``(skipped)``, never as ``0.0000`` or ``nan``.

    A fabricated number in a benchmark table is worse than a missing one, because a
    missing one is visible.
    """
    report = _report(
        Result("aniso_gmm", "DensFuse", Status.SKIPPED, reason="backend_unavailable")
    )
    text = format_table(report)
    assert SKIPPED in text
    assert "0.0000" not in text
    assert "nan" not in text.lower()
    assert "skipped" in text.lower()


def test_manifold_only_rows_show_a_dash_not_skipped() -> None:
    """A successful method with no density has an em dash, not ``(skipped)``.

    A manifold embedder genuinely has no NLL. Printing ``(skipped)`` there would
    suggest its backend was missing, which is a different and much more alarming
    claim -- and would make the skip count in the summary wrong.
    """
    report = _report(
        Result(
            "swiss_roll",
            "SkIsomap",
            Status.OK,
            nll=None,
            trustworthiness=0.9997,
            trustworthiness_std=0.0001,
        ),
    )
    text = format_table(report)
    assert NOT_APPLICABLE in text
    assert "0 skipped" in text, "an em-dash cell must not be counted as a skipped row"
    assert "1 measured" in text


def test_summary_line_reports_the_skip_count() -> None:
    """A table that is mostly skipped must not be mistaken for measurements."""
    report = _report(
        Result("d", "a", Status.OK, nll=1.0),
        Result("d", "b", Status.SKIPPED, reason="backend_unavailable"),
    )
    text = format_table(report)
    assert "1 skipped" in text
    assert "excluded from every aggregate" in text


def test_failed_rows_show_the_reason() -> None:
    """A failed row is visible and carries its reason."""
    report = _report(Result("d", "a", Status.FAILED, reason="ValueError: bad input"))
    text = format_table(report)
    assert "(failed)" in text
    assert "bad input" in text


def test_table_groups_rows_by_dataset() -> None:
    """Datasets are separated visually, because NLL is not comparable across them."""
    report = _report(
        Result("aniso_gmm", "a", Status.OK, nll=1.0),
        Result("circles", "a", Status.OK, nll=2.0),
    )
    lines = format_table(report).splitlines()
    dataset_positions = [
        i for i, ln in enumerate(lines) if ln.strip().startswith(("aniso_gmm", "circles"))
    ]
    assert len(dataset_positions) >= 2


# =========================================================== parser
def test_parser_exposes_the_four_subcommands() -> None:
    """``run``, ``bench``, ``demo`` and ``info``, as documented."""
    parser = build_parser()
    for command in ("run", "bench", "demo", "info"):
        args = parser.parse_args(
            [command] + (["--dataset", "circles"] if command == "run" else [])
        )
        assert args.command == command


def test_parser_defaults_match_the_documented_scale() -> None:
    """The default parameter scale is the one in the architecture document."""
    parser = build_parser()
    args = parser.parse_args(["run"])
    assert args.n_train == 800
    assert args.n_test == 300
    assert args.seeds == 3
    assert args.slow is False


def test_no_command_prints_help_and_exits_two() -> None:
    """No subcommand is a usage error, not a silent success."""
    assert main([]) == 2


def test_version_flag() -> None:
    """``--version`` prints the version and exits cleanly."""
    from densforge import __version__

    assert main(["--version"]) == 0
    assert __version__


# =========================================================== commands
def test_info_lists_both_tiers(capsys) -> None:
    """``info`` must show library backends and offline fallbacks separately."""
    assert main(["info"]) == 0
    text = capsys.readouterr().out
    assert "Tier-0 (library)" in text
    assert "Tier-1 (offline fallback)" in text
    for name in ("SkKernelDensity", "SkIsomap", "NumpyKDE", "NumpyIsomap"):
        assert name in text
    assert "backends available" in text


@pytest.mark.slow
def test_run_single_dataset(capsys) -> None:
    """``run`` completes on one dataset at a reduced scale."""
    code = main(
        ["run", "--dataset", "circles", "--n-train", "120", "--n-test", "100", "--seeds", "3"]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "circles" in text
    assert "DensFuse" in text
    assert "elapsed" in text


@pytest.mark.slow
def test_bench_writes_an_artefact(tmp_path, capsys) -> None:
    """``bench`` writes a JSON artefact that round-trips."""
    out = tmp_path / "benchmark.json"
    code = main(
        [
            "bench",
            "--out",
            str(out),
            "--datasets",
            "circles",
            "--n-train",
            "200",
            "--n-test",
            "120",
            "--seeds",
            "3",
        ]
    )
    assert code == 0
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_seeds"] == 3
    assert payload["results"]
    assert "wrote" in capsys.readouterr().out


def test_bench_refuses_fewer_than_three_seeds(tmp_path, capsys) -> None:
    """``--seeds 2`` cannot support a mean ± std report and is refused up front."""
    code = main(
        [
            "bench",
            "--out",
            str(tmp_path / "x.json"),
            "--datasets",
            "circles",
            "--n-train",
            "200",
            "--n-test",
            "120",
            "--seeds",
            "2",
        ]
    )
    assert code == 1
    assert "minimum is 3" in capsys.readouterr().err


def test_demo_quick_runs(capsys) -> None:
    """``demo --quick`` is a smoke test: one seed, small scale, all six datasets."""
    assert main(["demo", "--quick"]) == 0
    text = capsys.readouterr().out
    assert "(quick)" in text
    # Case-insensitive: the banner shouts "SMOKE TEST" for emphasis, and the
    # assertion is about the caveat being present, not about its capitalisation.
    assert "smoke test" in text.lower(), "a 1-seed demo must say it cannot support mean +/- std"
    for dataset in (
        "aniso_gmm",
        "swiss_roll",
        "double_spiral",
        "circles",
        "t_mixture",
        "manifold_noise",
    ):
        assert dataset in text


def test_utf8_is_enabled_before_printing() -> None:
    """Box-drawing and status characters must not crash a Windows console.

    The default Windows codec is GBK, which raises ``UnicodeEncodeError`` on
    several of the characters used in the header rule.
    """
    from densforge.cli import _enable_utf8

    _enable_utf8()
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    assert encoding in ("utf-8", "utf8", ""), encoding
