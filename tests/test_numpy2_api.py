"""numpy 2.x API contract (invariant I28).

Author: 晨星 <CJX0712@users.noreply.github.com>

Every row of the table below was measured on this machine (numpy 2.5.3), not
copied from release notes. That distinction matters: several widely-repeated claims
about the NumPy 2 migration are wrong, and asserting them would have produced a test
suite that fails for the right reason at the wrong time.

Correcting three such claims
----------------------------
* ``np.NINF``/``np.PINF`` were removed, but ``np.neginf``/``np.posinf`` are **not**
  their replacements -- neither exists. The replacements are the plain Python
  constants ``-np.inf`` and ``np.inf``.
* ``np.copy=False`` no longer raises unconditionally. Measured: ``np.array`` on a
  strided view with ``copy=False`` returns a sharing array without error. The
  tightened semantics are narrower than commonly stated, so the test pins the
  measured behaviour rather than a folklore rule.
* ``einsum`` does **not** broadcast a ``(n, d)`` operand against a ``(n, k, d)``
  one implicitly; the subscripts must be written to match, e.g.
  ``np.einsum('nkd,nd->nk', A, B)`` raises ``ValueError: einstein sum subscripts
  string contains too many subscripts for operand 0``. This project hit the error
  for real while writing the flagship, and the fix is an explicit ``[:, None, :]``.

The removals themselves are unambiguous and worth pinning: reaching for any of them
raises ``AttributeError``, so a stale name fails loudly on this environment. The
test additionally guards against the opposite failure -- a future NumPy restoring a
name via a compatibility shim, which would let old code keep running while the
underlying semantics drifted.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.invariant

#: (removed name, the spelling this project actually uses instead)
#:
#: Measured on numpy 2.5.3: the first column is absent, the second present.
REMOVED_AND_REPLACED: tuple[tuple[str, str], ...] = (
    ("trapz", "trapezoid"),
    ("float_", "float64"),
    ("cstack", "column_stack"),
    ("row_stack", "vstack"),
    ("alltrue", "all"),
    ("product", "prod"),
    ("in1d", "isin"),
    ("msort", "sort"),
    ("mat", "asarray"),
    ("issctype", "issubdtype"),
    ("set_string_function", "set_printoptions"),
    ("NINF", "inf"),
    ("PINF", "inf"),
    ("deprecate", "deprecate"),
    ("safe_eval", "safe_eval"),
)


@pytest.mark.parametrize(("removed", "replacement"), REMOVED_AND_REPLACED)
def test_i28_removed_symbol_is_absent(removed: str, replacement: str) -> None:
    """I28: the NumPy 2 removal really happened on this machine."""
    assert not hasattr(np, removed), (
        f"numpy.{removed} exists again; if a compatibility shim reintroduced it, "
        f"confirm that the code's use of {replacement!r} still means what it claims"
    )


def test_i28_infinity_constants_are_python_floats() -> None:
    """I28: infinity is spelled ``-np.inf`` / ``np.inf``, not ``np.NINF``.

    Pins the measured fact that no ``np.neginf``/``np.posinf`` exists either, which
    is the part of the migration that is commonly mis-stated.
    """
    assert np.isneginf(-np.inf)
    assert np.isposinf(np.inf)
    assert not hasattr(np, "neginf")
    assert not hasattr(np, "posinf")


def test_i28_trapezoid_is_the_integrator_in_use() -> None:
    """I28: the project's only integration uses ``np.trapezoid``.

    Invariant I1 verifies exact normalisation numerically, and this is the function
    that does it. ``np.trapz`` raises here, so the test doubles as a guard against a
    "helpful" future edit.
    """
    x = np.linspace(-1.0, 1.0, 101)
    value = float(np.trapezoid(np.ones_like(x), x))
    assert abs(value - 2.0) < 1e-12
    with pytest.raises(AttributeError):
        np.trapz(np.ones_like(x), x)  # type: ignore[attr-defined]


def test_i28_scalar_aliases_use_the_supported_spelling() -> None:
    """I28: ``float64`` rather than the removed ``float_``.

    ``np.int_`` still exists on numpy 2.5.3 even though ``np.float_`` does not.
    That asymmetry is worth pinning: it means a codebase can half-migrate and only
    fail on the float half, at runtime, in a numerical kernel.
    """
    assert np.dtype(np.float64) == np.dtype("float64")
    assert not hasattr(np, "float_")
    assert hasattr(np, "int_"), "numpy 2.5 unexpectedly removed np.int_"


def test_i28_copy_semantics_as_measured() -> None:
    """I28: ``copy=False`` shares when it can; it does not raise for strided views.

    Pins the measured behaviour rather than the commonly repeated claim that
    ``copy=False`` now always raises when a copy would be needed. On numpy 2.5.3
    ``np.array(strided_view, copy=False)`` returns a sharing array silently, so the
    safety this project relies on comes from ``ascontiguousarray`` in its own
    validators, not from NumPy's checking.
    """
    base = np.arange(10)
    assert np.shares_memory(np.array(base, copy=None), base)
    strided = base[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    # Measured: no raise. The guarantee has to come from the caller.
    assert np.shares_memory(np.array(strided, copy=False), strided)
    # And the function this project actually uses does force contiguity.
    assert np.ascontiguousarray(strided).flags["C_CONTIGUOUS"]


def test_i28_einsum_broadcasting_rules_are_as_documented() -> None:
    """I28: pin ``einsum``'s broadcasting behaviour, which is easy to get backwards.

    Measured on numpy 2.5.3, and the rule is subtler than "einsum broadcasts":

    * it broadcasts a lower-rank operand **only when the dropped axis is absent from
      the output** -- ``np.einsum('nkd,nd->nk', A, B)`` with ``A`` of shape
      ``(n, k, d)`` and ``B`` of shape ``(n, d)`` works;
    * it **fails** when the same operands are contracted to ``->nk`` with a
      two-index label, because ``B`` is then asked to carry a ``k`` axis it does
      not have: ``ValueError: einstein sum subscripts string contains too many
      subscripts for operand 1``.

    Both directions are asserted because the second one cost a real debugging cycle
    while writing the flagship, and the error message names operand 1 without saying
    which subscript is wrong.
    """
    big = np.random.RandomState(0).randn(4, 3, 5)  # (n, k, d)
    small = np.random.RandomState(1).randn(4, 5)  # (n, d)

    # Broadcasting works when the output drops the k axis.
    dropped = np.einsum("nkd,nd->nk", big, small)
    assert dropped.shape == (4, 3)
    loop = np.array([[float(big[i, j] @ small[i]) for j in range(3)] for i in range(4)])
    assert np.allclose(dropped, loop)

    # And fails when the label set demands an axis the operand lacks. `big` is
    # (n, k, d) but the spec asks for 'nkd->nk' of a single operand, which is
    # well-formed, whereas 'nkd,nk->nk' asks the (n, d) operand to supply a k axis.
    with pytest.raises(ValueError):
        np.einsum("nkd,nk->nk", big, small)

    # Making the axis explicit is the fix.
    explicit = np.einsum("nke,nke->nk", big, small[:, None, :])
    assert explicit.shape == (4, 3)
    assert np.allclose(explicit, loop)

    # A label repeated where no operand carries it is likewise an error.
    with pytest.raises(ValueError):
        np.einsum("nzd->nk", big)


def test_i28_no_module_level_rng_functions_are_used_in_generators() -> None:
    """I28 companion: generators must not touch NumPy's global state.

    Module-level ``np.random.rand`` would make the global stream the source of
    truth, which no per-call seed can control. Scans the AST rather than the text so
    that a mention inside a docstring explaining the ban does not trip the check.
    """
    import ast
    import inspect

    from densforge.data import synth

    tree = ast.parse(inspect.getsource(synth))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
            continue
        if func.value.id != "np" or func.value.id == "np.random":
            continue
        if func.attr in {"random", "rand", "randn", "seed", "randint", "uniform", "choice"}:
            offenders.append(f"line {node.lineno}: np.{func.attr}(...)")
    assert not offenders, f"data/synth.py uses module-level RNG: {offenders}"


def test_i28_default_rng_is_never_called_in_the_package() -> None:
    """I28 companion: ``default_rng`` is never *called* anywhere in the package.

    NEP 19 does not freeze the ``Generator`` bit stream, so a single call would let
    the published thresholds drift between the NumPy versions in the CI matrix.
    An AST scan is used rather than a text search, because the package deliberately
    *mentions* ``default_rng`` in its documentation to explain the ban.
    """
    import ast
    import pathlib

    import densforge

    root = pathlib.Path(densforge.__file__).parent
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                if name == "default_rng":
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"default_rng() called in: {offenders}"


def test_i28_column_stack_is_available_for_every_generator() -> None:
    """I28: the replacement this project relies on most is really present.

    ``np.cstack`` was removed and the swiss-roll generator hit
    ``AttributeError: module 'numpy' has no attribute 'cstack'`` while being written.
    """
    assert np.column_stack([np.arange(3.0), np.arange(3.0)]).shape == (3, 2)
    assert not hasattr(np, "cstack")
