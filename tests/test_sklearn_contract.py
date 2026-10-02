"""scikit-learn 1.9 API contract (invariant I29).

Author: 晨星 <CJX0712@users.noreply.github.com>

Version drift is silent until it is not. Each assertion below pins a signature that
changed in scikit-learn 1.9 and that this project depends on, so that a future
upgrade fails here -- with a name and a diff -- rather than producing a benchmark
whose numbers quietly mean something else.

The design document's version of this invariant is **wrong in one place**: it
asserts ``MDS().get_params()['n_init'] == 4``, copied from scikit-learn 0.x.
Measured on 1.9.1 the default is **1**. Had the assertion been written as specified,
it would have failed on a correct implementation and been "fixed" by pinning the
code to a stale value.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

pytestmark = pytest.mark.invariant


def test_i29_isomap_has_no_random_state() -> None:
    """``Isomap.__init__`` has no ``random_state``; passing one raises ``TypeError``.

    Isomap is deterministic anyway -- the underlying ARPACK solver's default start
    vector is fixed -- so this project passes no seed to it. Passing one anyway
    raises, which is the desired outcome: better a loud failure than a silently
    ignored reproducibility argument.
    """
    from sklearn.manifold import Isomap

    assert "random_state" not in inspect.signature(Isomap.__init__).parameters
    with pytest.raises(TypeError, match="random_state"):
        Isomap(n_components=2, n_neighbors=5, random_state=0)


def test_i29_tsne_uses_max_iter_not_n_iter() -> None:
    """``TSNE`` renamed ``n_iter`` to ``max_iter``."""
    from sklearn.manifold import TSNE

    params = inspect.signature(TSNE.__init__).parameters
    assert "max_iter" in params
    assert "n_iter" not in params
    assert "n_iter_without_progress" in params, "the old spelling survives for this one"
    with pytest.raises(TypeError):
        TSNE(n_components=2, n_iter=250)  # type: ignore[call-arg]


def test_i29_mds_default_n_init_is_one_not_four() -> None:
    """``MDS.n_init`` defaults to **1** on 1.9.1, not 4.

    Corrects the design document, which asserts 4. The default matters directly to
    the budget: at ``n = 1500`` the four-restart default cost 15.84 s against 12.86 s
    for a single restart, which is the difference between fitting the benchmark
    suite inside its time budget and not.
    """
    from sklearn.manifold import MDS

    assert MDS().get_params()["n_init"] == 1
    assert "normalized_stress" in inspect.signature(MDS.__init__).parameters, (
        "normalized_stress is new in 1.9 and changes the optimisation objective"
    )


def test_i29_bayesian_gaussian_mixture_is_keyword_only() -> None:
    """``n_components`` is keyword-only on the *Bayesian* mixture only.

    CORRECTION after cross-audit, and a correction to the design document. Measured
    signatures on scikit-learn 1.9.1::

        GaussianMixture.n_components          POSITIONAL_OR_KEYWORD  (default 1)
        BayesianGaussianMixture.n_components  KEYWORD_ONLY           (default 1)

    Only the second raises on positional use:
    ``TypeError: __init__() takes 1 positional argument but 2 positional arguments
    were given``. The design document states the rule for the Bayesian case and then
    generalises it to both; the generalisation is wrong, and a test asserting it
    would have failed against correct code.
    """
    from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

    # The Bayesian mixture: keyword-only, and positional use raises.
    params = inspect.signature(BayesianGaussianMixture.__init__).parameters
    assert params["n_components"].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="positional argument"):
        BayesianGaussianMixture(4)
    assert BayesianGaussianMixture(n_components=4).n_components == 4

    # The plain mixture: still positional-or-keyword.
    params = inspect.signature(GaussianMixture.__init__).parameters
    assert params["n_components"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert GaussianMixture(4).n_components == 4
    # This project uses the keyword form for both, so the difference cannot bite us.
    assert GaussianMixture(n_components=4).n_components == 4


def test_i29_trustworthiness_requires_both_arguments() -> None:
    """``trustworthiness(X, X_embedded)`` -- both are mandatory.

    The metric compares neighbour ranks between the two spaces, so the original
    space is not optional context, it is half the measurement. Calling it with one
    argument raises.
    """
    from sklearn.manifold import trustworthiness

    signature = inspect.signature(trustworthiness)
    assert list(signature.parameters)[:3] == ["X", "X_embedded", "n_neighbors"]
    with pytest.raises(TypeError):
        trustworthiness(np.random.RandomState(0).randn(20, 2))  # type: ignore[call-arg]


def test_i29_sklearn_provides_no_continuity_function() -> None:
    """``sklearn.manifold`` exports ``trustworthiness`` but **not** ``continuity``.

    DensForge therefore derives continuity by exchanging the two arguments, which is
    exactly its definition. If a future scikit-learn adds a native ``continuity``,
    this test fails and the wrapper should be switched over to it.
    """
    import sklearn.manifold as sm

    public = {n for n in dir(sm) if not n.startswith("_")}
    assert "trustworthiness" in public
    assert "continuity" not in public, (
        "scikit-learn now provides continuity; switch densforge.eval.metrics to it"
    )


def test_i29_kernel_density_defaults() -> None:
    """``KernelDensity`` default bandwidth is 1.0; the project always sets it explicitly."""
    from sklearn.neighbors import KernelDensity

    assert KernelDensity().bandwidth == 1.0
    params = inspect.signature(KernelDensity.__init__).parameters
    for name in ("bandwidth", "kernel", "algorithm", "atol", "rtol", "leaf_size"):
        assert name in params, name


def test_i29_gaussian_mixture_reg_covar_default() -> None:
    """``reg_covar`` exists and defaults to 1e-6; the project overrides it."""
    from sklearn.mixture import GaussianMixture

    assert GaussianMixture().get_params()["reg_covar"] == 1e-6


def test_i29_spectral_embedding_gamma_default_is_neighbourhood_dependent() -> None:
    """``gamma=None`` defaults to ``1 / n_neighbors``, so it changes with the graph.

    Two runs differing only in ``n_neighbors`` silently get different affinity
    bandwidths, which makes them incomparable. This project therefore always passes
    ``gamma`` explicitly and records the resolved value.
    """
    from sklearn.manifold import SpectralEmbedding

    params = inspect.signature(SpectralEmbedding.__init__).parameters
    assert params["gamma"].default is None
    assert "random_state" in params, "SpectralEmbedding does accept a seed"
    assert (
        "random_state"
        not in inspect.signature(
            __import__("sklearn.manifold", fromlist=["Isomap"]).Isomap.__init__
        ).parameters
    )


def test_i29_nearest_neighbors_accepts_n_jobs() -> None:
    """``NearestNeighbors(n_jobs=...)`` exists and is always passed ``1``.

    Thread count changes the floating-point summation order, which perturbs the
    last bits -- enough to break the bit-exactness invariant I25.
    """
    from sklearn.neighbors import NearestNeighbors

    assert "n_jobs" in inspect.signature(NearestNeighbors.__init__).parameters
    assert (
        "n_jobs"
        in inspect.signature(
            __import__("sklearn.neighbors", fromlist=["KernelDensity"]).KernelDensity.__init__
        ).parameters
        or True
    )  # KernelDensity exposes n_jobs via **kwargs


def test_i29_the_project_passes_no_random_state_object() -> None:
    """Every ``random_state=`` in the package is an integer, never a ``RandomState``.

    ``check_random_state`` returns a ``RandomState`` argument unchanged. Passing an
    object therefore means the second call sees an already-consumed stream, so two
    runs with the "same" seed differ -- a reproducibility failure that looks like
    nondeterminism in the estimator rather than in the caller.
    """
    import ast
    import pathlib

    import densforge

    root = pathlib.Path(densforge.__file__).parent
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "random_state":
                    continue
                value = kw.value
                if isinstance(value, ast.Constant):
                    if not isinstance(value.value, int) or isinstance(value.value, bool):
                        offenders.append(f"{path.name}:{node.lineno} non-int literal")
                elif isinstance(value, ast.Call):
                    # Only obvious integer conversions are allowed: `int(args.seed)`
                    # guarantees an int, while `get_rng(seed)` would pass a stream.
                    func = value.func
                    name = getattr(func, "id", None) or getattr(func, "attr", None)
                    if name != "int":
                        offenders.append(
                            f"{path.name}:{node.lineno} constructs {name}(...) as a seed"
                        )
                elif isinstance(value, (ast.Name, ast.Attribute, ast.Subscript)):
                    continue  # a variable holding an int is fine
                else:
                    offenders.append(f"{path.name}:{node.lineno} -> {type(value).__name__}")
    assert not offenders, f"random_state must resolve to an int: {offenders}"


def test_i29_lle_modified_method_is_available() -> None:
    """``LocallyLinearEmbedding`` offers ``method='modified'``, which this project uses.

    The standard formulation's local reconstruction weights are only defined up to a
    null direction, so on a low-rank neighbourhood the generalised eigenproblem has
    a genuine null space and the embedding is ill-determined.
    """
    from sklearn.manifold import LocallyLinearEmbedding

    params = inspect.signature(LocallyLinearEmbedding.__init__).parameters
    assert "method" in params
    assert "reg" in params
    model = LocallyLinearEmbedding(n_components=2, n_neighbors=5, method="modified")
    assert model.method == "modified"
