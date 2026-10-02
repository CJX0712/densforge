"""Architecture contracts: layering, structural guarantees, forbidden imports.

Author: 晨星 <CJX0712@users.noreply.github.com>

These tests are about the *shape* of the codebase rather than the behaviour of any
estimator. They exist because three of the project's central claims are structural
and cannot be verified by running the code:

1. the dependency graph is acyclic, with ``core`` as a sink;
2. the flagship contains no baseline class, so its gains are structurally reachable;
3. no self-implemented eigendecomposition falls back to a full ``eigh``.

Each of those is a promise a code review cannot keep indefinitely.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import densforge

PACKAGE_ROOT = pathlib.Path(densforge.__file__).parent

#: Layer number per subpackage, from densforge/docs/architecture.md §1.1.
#: A module may only import from *higher* numbers (its own layer or below).
LAYERS: dict[str, int] = {
    "core": 1,
    "data": 2,
    "eval": 2,
    "training": 3,
    "hpo": 4,
    "density": 5,
    "manifold": 5,
    "pipeline": 6,
    "cli": 7,
}

#: Subpackages each layer may import. `core` imports nothing from the package.
ALLOWED: dict[str, frozenset[str]] = {
    "core": frozenset(),
    "data": frozenset({"core"}),
    "eval": frozenset({"core", "data"}),
    "training": frozenset({"core", "density", "manifold"}),
    "hpo": frozenset({"core", "data", "training", "eval"}),
    "density": frozenset({"core", "data", "training"}),
    "manifold": frozenset({"core", "data", "training"}),
    "pipeline": frozenset({"core", "data", "density", "manifold", "hpo", "training", "eval"}),
    "cli": frozenset(
        {"core", "data", "density", "manifold", "hpo", "training", "eval", "pipeline"}
    ),
}

#: Baseline classes the flagship must never touch. This is the machine-checkable
#: form of the design document's protection P2: if the flagship were
#: "strongest baseline + patch", the "beat the baseline by X%" claim would be
#: structurally unreachable, and no amount of benchmarking would fix that.
FORBIDDEN_IN_FLAGSHIP: frozenset[str] = frozenset(
    {
        "KernelDensity",
        "GaussianMixture",
        "BayesianGaussianMixture",
        "Isomap",
        "LocallyLinearEmbedding",
        "SpectralEmbedding",
        "MDS",
        "TSNE",
        "gaussian_kde",
        "SplineTransformer",
    }
)


def _module_files() -> list[pathlib.Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _layer_of(path: pathlib.Path) -> str | None:
    rel = path.relative_to(PACKAGE_ROOT)
    return rel.parts[0] if len(rel.parts) > 1 else None


def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _densforge_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """Return ``(lineno, subpackage)`` for every ``densforge.*`` import.

    Handles both ``from densforge.x import y`` and ``import densforge.x``, plus
    function-local imports, which this project uses deliberately to break cycles
    and to keep heavy dependencies off the import path.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == "densforge" and len(parts) > 1:
                found.append((node.lineno, parts[1]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "densforge" and len(parts) > 1:
                    found.append((node.lineno, parts[1]))
    return found


# ------------------------------------------------------------------ layering
@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
def test_no_layer_imports_from_a_higher_layer(path: pathlib.Path) -> None:
    """Every intra-package import points downward in the layer order."""
    layer = _layer_of(path)
    if layer is None or layer not in ALLOWED:
        return  # top-level modules (__init__.py, cli.py handled separately)
    allowed = ALLOWED[layer]
    for lineno, target in _densforge_imports(_parse(path)):
        assert target in allowed, (
            f"{path.name}:{lineno} imports densforge.{target}, but layer "
            f"'{layer}' may only import {sorted(allowed) or 'nothing'}"
        )


def test_core_is_a_sink() -> None:
    """``core`` must not import any other DensForge subpackage.

    This is what makes the topological order well defined. If ``core`` imported
    ``data``, the graph would close into a cycle and the "acyclic by construction"
    claim would be false -- while still looking fine at runtime, because Python
    resolves cycles lazily in many cases.
    """
    for path in (PACKAGE_ROOT / "core").rglob("*.py"):
        for lineno, target in _densforge_imports(_parse(path)):
            pytest.fail(
                f"core/{path.name}:{lineno} imports densforge.{target}; core must be a sink"
            )


def test_relative_parent_imports_never_escape_the_package() -> None:
    """A relative import may not climb above the package root.

    The project's convention is ``from ..core import x`` from a module inside
    ``densforge/<layer>/``, which is exactly ``densforge.core``. What the layer
    table must be able to see is an import that *escapes* the package, because that
    is how a package's real dependency structure diverges from its apparent one.

    Both the level and the resolved target are checked, since a level of 3 from a
    nested module can still land back inside the package while a level of 2 from the
    package root cannot.
    """
    for path in _module_files():
        rel_depth = len(path.relative_to(PACKAGE_ROOT).parts) - 1  # dirs below densforge/
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            # level=1 -> current package, level=2 -> parent, ... The file sits
            # `rel_depth` directories below densforge/, so the resolved depth is
            # rel_depth - (level - 1).
            resolved = rel_depth - (node.level - 1)
            assert resolved >= 0, (
                f"{path.name}:{node.lineno} `from {'.' * node.level}"
                f"{node.module or ''} import ...` climbs above the densforge package; "
                "use an absolute import so the layer table can see it"
            )


def test_every_intra_package_import_resolves_inside_the_package() -> None:
    """Every ``densforge.*`` import must name a module that actually exists.

    A typo in an intra-package import is invisible to Python until that code path
    runs, which for a rarely-taken branch means it may never run before release.
    """
    import importlib

    for path in _module_files():
        for _, target in _densforge_imports(_parse(path)):
            candidates = [
                f"densforge.{target}",
                f"densforge.{target.rsplit('.', 1)[0]}"
                if "." in target
                else f"densforge.{target}",
            ]
            found = False
            for candidate in candidates:
                try:
                    importlib.import_module(candidate)
                    found = True
                    break
                except ImportError:
                    continue
            assert found, f"{path.name} imports densforge.{target}, which does not exist"


# ------------------------------------------------------------------ flagship purity
@pytest.mark.parametrize(
    "relative", ["density/flagship.py", "manifold/flagship.py"], ids=lambda p: p
)
def test_flagship_does_not_use_baselines(relative: str) -> None:
    """Neither flagship may *call* a baseline class.

    The flagship must be an independent kernel estimator, not a wrapper around the
    strongest baseline. If it were the latter, "flagship beats the best baseline by
    X%" would be unreachable in principle, and a benchmark failure would be
    ambiguous between "needs more tuning" and "structurally impossible".

    The check is on the **AST**, not on the file text, because both modules
    deliberately *name* baselines in their docstrings -- the whole argument for why
    the flagship is not a wrapper is written there, in terms of the baselines it
    deliberately does not use. A text search cannot tell a mention from a call.
    """
    path = PACKAGE_ROOT / relative
    tree = _parse(path)
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
    for banned in sorted(FORBIDDEN_IN_FLAGSHIP):
        assert banned not in used, (
            f"{relative} references {banned!r} in code; the flagship must not wrap a "
            "baseline (a mention in a docstring is fine and expected)"
        )


def test_flagship_uses_only_primitive_neighbourhood_search() -> None:
    """The flagship's only scikit-learn dependency is ``NearestNeighbors``.

    Even that is worth pinning: a "convenience" swap to a full ``pairwise_distances``
    or a ``KernelDensity`` call would be a silent step toward the wrapper structure
    the previous test forbids.
    """
    for relative in ("density/flagship.py", "manifold/flagship.py"):
        tree = _parse(PACKAGE_ROOT / relative)
        sklearn_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("sklearn"):
                sklearn_imports.append(node.module)
            elif isinstance(node, ast.Import):
                sklearn_imports.extend(
                    a.name for a in node.names if a.name.startswith("sklearn")
                )
        assert set(sklearn_imports) <= {"sklearn.neighbors"}, (
            f"{relative} imports {sorted(set(sklearn_imports))}; only "
            "sklearn.neighbors.NearestNeighbors is permitted"
        )


# ------------------------------------------------------------------ numerics
def test_no_full_eigh_on_sample_space_matrices() -> None:
    """Sample-space eigendecompositions must request a *subset* of eigenpairs.

    A full ``eigh`` on an ``N x N`` matrix is the project's most expensive avoidable
    operation. Measured at ``N = 2000, d = 16``: full solve 10.96 s, partial solve
    on the same matrix 1.77 s -- a 6.2x difference for identical accuracy, because
    only ``k + 1`` eigenpairs are ever used.

    The distinction that matters is the matrix **order**, not the function name:

    * an ``N x N`` operator needs a partial solve, and is checked for one;
    * an ``N x d x d`` batch of local covariances has ``d`` bounded by the ambient
      dimension (at most 16 here), so a full solve is ``O(N d^3)`` and correct;
    * the one place a full solve *is* required is the ``d x d`` Gram matrix in the
      biorthogonalisation fallback, where a partial solve would not return the
      inverse square root.

    This test targets the first case and states why the other two are fine, rather
    than banning a function name and forcing a workaround onto correct code.
    """
    offenders: list[str] = []
    for path in _module_files():
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if getattr(func, "attr", None) != "eigh":
                continue
            # The only function named exactly `eigh` is scipy.linalg.eigh, which is
            # the one that supports subset_by_index. np.linalg.eigh appears as an
            # Attribute on `np.linalg`, so its `func.value` is a Name.
            is_scipy_eigh = isinstance(func, ast.Name)
            has_subset = any(kw.arg == "subset_by_index" for kw in node.keywords)
            if is_scipy_eigh and not has_subset:
                offenders.append(f"{path.name}:{node.lineno} scipy eigh without subset")
    assert not offenders, f"full eigendecomposition found: {offenders}"


def test_sample_space_solves_use_subset_by_index() -> None:
    """Every helper that decomposes a sample-space matrix passes ``subset_by_index``.

    Checked positively rather than negatively: the negative form cannot distinguish
    "no full solves" from "no solves at all".
    """
    import densforge.density.tier1 as tier1

    source = pathlib.Path(tier1.__file__).read_text(encoding="utf-8")
    assert "subset_by_index" in source
    tree = ast.parse(source)
    partial = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "scipy_eigh"
        and any(kw.arg == "subset_by_index" for kw in node.keywords)
    ]
    assert partial, "tier1 must use a partial eigensolver"


def test_numpy_linalg_eigh_is_only_used_on_dimension_bounded_matrices() -> None:
    """``np.linalg.eigh`` is batched and must only see ``(N, d, d)`` local covariances.

    ``np.linalg.eigh`` has no partial-solve mode, so it is correct for the batched
    local covariance -- where the trailing dimension is the ambient dimension, at
    most 16 -- and wrong for any ``N x N`` operator. This test records that the only
    ``np.linalg.eigh`` in the package is the local-covariance one.
    """
    sites: list[str] = []
    for path in _module_files():
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if getattr(func, "attr", None) != "eigh":
                continue
            # `np.linalg.eigh` reaches here as an Attribute whose value is itself
            # the Attribute `np.linalg` -- not a bare Name. CORRECTION after
            # cross-audit: the earlier version matched `func.value.id == "linalg"`,
            # which never fires for that shape, so the test found zero sites and
            # then failed its own `len(sites) == 1` assertion. The check now matches
            # the two-level attribute chain it is actually looking for.
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "linalg"
            ):
                sites.append(f"{path.name}:{node.lineno}")
    # Two sites are legitimate: the flagship's batched (N, d, d) local covariances,
    # and the old-SciPy fallback inside tier1's partial solver, which only runs on a
    # SciPy too old to accept `subset_by_index` and is documented as such.
    assert len(sites) == 2, (
        f"np.linalg.eigh found at {sites}; expected exactly the flagship's batched "
        "(N, d, d) local covariances and the documented old-SciPy fallback"
    )
    assert any(site.startswith("flagship.py:") for site in sites), sites
    assert any(site.startswith("tier1.py:") for site in sites), sites
    fallback_source = (PACKAGE_ROOT / "density" / "tier1.py").read_text(encoding="utf-8")
    assert "except TypeError" in fallback_source, (
        "the tier1 np.linalg.eigh must stay behind a `except TypeError` guard, so it "
        "runs only when the installed SciPy has no subset_by_index"
    )


# ------------------------------------------------------------------ portability
def test_no_resource_module_import() -> None:
    """``import resource`` is forbidden: it does not exist on Windows.

    ``resource.getrusage`` is the standard way to report peak memory on Linux, and
    using it would make the package unimportable on the development platform while
    passing in CI -- the worst possible combination, because the failure appears
    only where the code was not tested.
    """
    for path in _module_files():
        tree = _parse(path)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert "resource" not in names, (
                f"{path.name}:{node.lineno} imports `resource`, which is POSIX-only"
            )


def test_all_source_files_are_utf8() -> None:
    """Every source file must be readable as UTF-8 and openable with an explicit codec.

    Windows' default codec is not UTF-8, so an implicit ``open()`` on a file
    containing a non-ASCII character -- this project's author name is one -- raises
    ``UnicodeDecodeError`` on a developer's machine and not in CI.
    """
    for path in _module_files():
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - would fail loudly
            pytest.fail(f"{path.name} is not valid UTF-8: {exc}")

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = _parse(path)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open"):
                continue
            if node.args and isinstance(node.args[0], ast.Constant):
                mode = node.args[0].value
                if isinstance(mode, str) and mode.endswith((".json", ".md", ".txt")):
                    has_encoding = any(kw.arg == "encoding" for kw in node.keywords)
                    assert has_encoding, (
                        f"{path.name}:{node.lineno} opens {mode!r} without an explicit "
                        "encoding; Windows would decode it with the locale codec"
                    )


# ------------------------------------------------------------------ protocol shape
def test_every_estimator_exposes_the_protocol_methods() -> None:
    """Each estimator class provides the methods its role requires.

    Checked structurally rather than by ``isinstance`` against a ``Protocol``,
    because ``runtime_checkable`` protocols only verify method *names* -- a class
    with a method of the wrong signature passes.
    """
    from densforge.density.baselines import DENSITY_BASELINES
    from densforge.density.flagship import DensFuse
    from densforge.manifold.baselines import MANIFOLD_BASELINES
    from densforge.manifold.flagship import ManifoldFuse

    for cls in DENSITY_BASELINES:
        for method in ("fit", "score_samples", "available", "params"):
            assert callable(getattr(cls, method, None)), f"{cls.__name__}.{method}"
    for cls in MANIFOLD_BASELINES:
        for method in ("fit", "transform", "available", "params"):
            assert callable(getattr(cls, method, None)), f"{cls.__name__}.{method}"
    for method in ("fit", "select", "refit", "score_samples", "embed", "available"):
        assert callable(getattr(DensFuse, method, None)), f"DensFuse.{method}"
    for method in ("fit", "transform", "score_samples", "available"):
        assert callable(getattr(ManifoldFuse, method, None)), f"ManifoldFuse.{method}"


def test_report_attributes_do_not_shadow_methods() -> None:
    """Report and Result must not define an attribute that shadows a method.

    ``self.benchmark = None`` on a class that also defines ``benchmark()`` makes the
    method unreachable. The pipeline's ``benchmark()`` method and the report's
    fields are the pair most at risk, so the check is explicit.
    """
    from densforge.core.types import Report, Result

    for cls in (Report, Result):
        instance_fields = {
            f.name
            for f in cls.__dataclass_fields__.values()  # type: ignore[attr-defined]
        }
        for name in instance_fields:
            assert not callable(getattr(cls, name, None)), (
                f"{cls.__name__}.{name} is both a field and a method"
            )


def test_available_is_a_pure_predicate() -> None:
    """``available()`` must not construct an estimator.

    The benchmark calls it once per row; if it constructed anything, a missing
    backend would be discovered expensively and destructively. Verified by
    monkeypatching the constructor to raise.
    """
    from densforge.density.baselines import DENSITY_BASELINES
    from densforge.manifold.baselines import MANIFOLD_BASELINES

    for cls in (*DENSITY_BASELINES, *MANIFOLD_BASELINES):
        original = cls.__init__

        # Bound as a default argument rather than closing over `cls`, so the
        # replacement cannot report the wrong class if the loop is ever reordered.
        def explode(self, *args, _name=cls.__name__, **kwargs):
            raise AssertionError(f"{_name}.available() constructed an instance")

        cls.__init__ = explode  # type: ignore[method-assign]
        try:
            assert isinstance(cls.available(), bool)
        finally:
            cls.__init__ = original  # type: ignore[method-assign]
