# DensForge architecture

Author: 晨星 &lt;CJX0712@users.noreply.github.com&gt; · License: MIT

DensForge is a numerical-computing and contract-engineering project, not a
deep-learning one. The architecture is therefore organised around four questions:
what does each interface *mean*, what makes a result *reproducible*, how does the
package *degrade* when a dependency is missing, and how is the evaluation kept
*honest*. Each of those is an engineering problem, and the package structure
follows from them.

---

## 1. Layering

Strictly acyclic, and the ordering is a topological sort of the dependency graph.

```
 L7   cli.py · examples/run_demo.py
 L6   pipeline/            orchestration, artefact I/O
 L5   density/  manifold/ the domain layer
 L4   hpo/                hyper-parameter selection
 L3   training/           fit bookkeeping and provenance
 L2   eval/  data/        metrics, leakage firewall, generators
 L1   core/               types, errors, config, protocols, seed   <- sink
```

`core` imports nothing from the package. That is what makes the ordering
well-defined: if `core` reached back into `data`, the graph would close into a
cycle, and Python would resolve many such cycles lazily -- so the code would run
while the "acyclic by construction" claim was quietly false.

| layer | modules | may import |
|---|---|---|
| L1 | `types` `errors` `config` `interfaces` `seed` | stdlib + NumPy types only |
| L2 | `data/synth` `data/datasets` `eval/metrics` `eval/leakage` | `core`, `data` |
| L3 | `training/fitter` | `core`, `density`, `manifold` (via Protocol only) |
| L4 | `hpo/bandwidth` `hpo/alpha` `hpo/coordinate` | `core`, `data`, `training`, `eval` |
| L5 | `density/*` `manifold/*` | `core`, `data`, `training` |
| L6 | `pipeline/densforge_pipeline` | L1–L5 |
| L7 | `cli.py`, `examples/run_demo.py` | L1–L6 |

**Enforced, not documented.** `tests/test_architecture.py` walks the AST of every
module and fails on any upward import, on any relative import that escapes the
package, and on any intra-package import that does not resolve.

### 1.1 The one apparent cycle, and how it is broken

`training/fitter.py` needs to name concrete density and manifold estimators, and
those estimators need the metrics. The resolution:

* `fitter` depends on `core/interfaces.py` — a `Protocol` — for its type
  annotations, never on a concrete class;
* a concrete class is resolved at call time through the `available()` factory;
* `density/flagship.py` imports `manifold/diffusionmaps.py` for the shared
  operator, and `manifold/flagship.py` imports no density module at all.

So the static graph is acyclic even though the runtime collaboration is
bidirectional. `ruff`'s `TID` rules plus the AST test keep it that way.

---

## 2. The contracts

### 2.1 Three protocols, one convention

`core/interfaces.py` declares `DensityEstimator`, `ManifoldEmbedder` and
`FusedEstimator`. The one invariant that makes cross-method comparison possible:

> **`score_samples` returns log-density, and larger means more likely.**

Every density number in the project is `NLL = -mean(score_samples(X_test))` in
nats. Returning a probability instead is not a style choice. Measured on 500×2
Gaussian samples, shrinking the bandwidth to `1e-3` drives `log p` to `-1.73e6`;
`np.exp` of that overflows to `inf`, the NLL becomes `nan`, and **nothing raises**.
The whole benchmark is silently destroyed.

`FusedEstimator` is deliberately **not** a subclass of the other two. It is their
structural intersection, declared independently, so `isinstance` cannot report
`True` for an object missing one of the methods.

### 2.2 Tiered availability

`available()` is a class-level lazy factory. Two rules, both learned the hard way:

**Do not read the attribute off the base class.** A method that does
`return cls._estimator_cls is not None` finds the *base's* `None` through the MRO,
so every subclass reports "unavailable" and the benchmark skips everything while
appearing to run. `densforge/density/baselines.py` resolves each subclass's
backend in `__init_subclass__` and writes the result into that subclass's own
namespace.

**Never construct anything.** `available()` is a boolean predicate, called once
per benchmark row. `tests/test_architecture.py` monkey-patches every estimator's
`__init__` to raise, then calls `available()`, so a future construction inside the
probe fails the build.

### 2.3 Tier-1 offline fallbacks

`densforge/density/tier1.py` re-implements the same mathematics with no
scikit-learn dependency: `NumpyKDE`, `NumpyGMM`, `NumpyIsomap`, `NumpySpectral`,
`NumpyDiffusionMaps`. A missing backend becomes a **measured** row rather than a
skipped one.

Every self-implemented decomposition uses a **partial** solve:

```python
scipy.linalg.eigh(matrix, subset_by_index=[0, k - 1])
```

Measured at `N = 2000, d = 16`: a full `eigh` takes **10.96 s**, the partial solve
on the same matrix **1.77 s** — **6.2×** for identical accuracy, because only
`k + 1` eigenpairs are ever needed. `tests/test_architecture.py` fails the build on
a full `eigh` of a sample-space matrix.

The offline path is tested by *simulating* the dependency's absence: the
scikit-learn modules are replaced with `None` in `sys.modules` and blocked in
`builtins.__import__`, so an accidental dependency inside the fallback raises
instead of quietly succeeding on a developer machine that happens to have the
library.

---

## 3. Evaluation and the leakage firewall

### 3.1 Split protocol

```
make(n_train,      seed)           -> train    the only data fit() may see
make(n_train // 5, seed + 500)     -> val      hyper-parameter selection only
make(n_test,       seed + 10_000)  -> test     final scoring only
```

Three **independent draws**, never a random shuffle of one pool. Slicing one pool
makes the halves share sampling noise, which biases test NLL optimistically and
makes the headline number unfalsifiable. Because the test seed is the train seed
plus a fixed offset, the independence is *provable* rather than merely likely.

### 3.2 Four assertions

| # | assertion | mechanism | error |
|---|---|---|---|
| **A1** | seed isolation | `test_seed == train_seed + 10_000` | `LeakageError` (E401) |
| **A2** | metric isolation | every array scored during HPO is recorded; the test array's `id` must not appear | `LeakageTestSetTouchedError` (E402) |
| **A3** | model isolation | the model records a **content digest** of everything fitted on | `LeakageModelNotTrainOnlyError` (E403) |
| **A4** | shape overlap | `min(cdist(train[:64], test)) > 0` | `LeakageError` (E401) |

A3 uses a SHA256 **digest** rather than `id()`, because a *copy* of the test array
has a different identity and the same content — and is just as much of a leak.

A2 is the one that catches the subtlest failure: a tuning loop that accidentally
evaluates on the test set produces entirely plausible numbers and a
slightly-too-good result, with nothing else wrong anywhere.

### 3.3 Honest reporting

A `skipped` row carries no numbers. Not zeros, not `nan`, not omitted — because a
fabricated number in a benchmark table is worse than a missing one, which at least
is visible. `Result.aggregated()` is the single gate, and a value that is missing
prints as `(skipped)` for an unavailable backend and as an em dash for a method
that legitimately does not produce that quantity (a manifold embedder has no
density). Conflating those two would report a *shape* as a *failure*.

### 3.4 Significance

`delta > 0.5 * (sigma_a + sigma_b)`, strictly. Requiring the gap to exceed half
the sum of the two standard deviations means a difference that could plausibly be
noise is reported as no difference. The project would rather miss a real small win
than announce one that reverses on the next seed.

---

## 4. Determinism

| risk | mitigation |
|---|---|
| `default_rng` stream is not frozen across NumPy versions (NEP 19) | `core/seed.get_rng` returns a legacy `RandomState`; an AST test fails on any `default_rng()` call |
| module-level `np.random.rand` pollutes global state | generators take an explicit seed; a test snapshots `np.random.get_state()` and asserts it is unchanged |
| multi-threaded reductions reorder floating-point sums | `set_all` pins `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB` to 1 thread; CI sets them too |
| `eigh` eigenvector sign is arbitrary under degeneracy | geometric metrics are invariant to it; element-wise tests align signs |
| sklearn `random_state` given a `RandomState` object | always an **integer**; an AST test enforces it |
| `KernelDensity` tree pruning (`atol`/`rtol`) introduces jitter | pinned to `0.0` |

The golden values in `tests/test_determinism.py` are exact full-precision reprs of
`np.random.RandomState(12345)`, so a NumPy upgrade that changed the legacy stream
fails loudly instead of letting every published threshold drift between CI jobs.

---

## 5. Where the flagship's assumptions hold, and where they do not

This section reports measurements, not claims. Full detail in
[`math_verification.md`](math_verification.md).

**Three axes, each orthogonal to what a baseline can express.** The flagship
(`density/flagship.py`) imports no baseline class — not `KernelDensity`, not
`GaussianMixture`, not `SpectralEmbedding`, not `Isomap`. That is the structural
guarantee that its gains are reachable: if it were "strongest baseline + patch",
"beats the best baseline by X%" would be impossible in principle, and a failure
would be ambiguous between "needs tuning" and "structurally impossible". An AST
test enforces the ban.

**Measured, at `n = 400`, one seed, validation-tuned on both sides:**

| dataset | `rho/median(rho)` range | flagship vs `fixedkde(val-bw)` |
|---|---|---|
| `double_spiral` | 11.0× | **+69.0%** |
| `circles` | 8.0× | +5.3% |
| `t_mixture` | 3.5× | +0.19% |
| `aniso_gmm` | 3.5× | +0.13% |
| `swiss_roll` | 2.8× | −0.12% |
| `manifold_noise` | 1.8× | −2.57% |

The self-tuning axis pays exactly where the local scale varies. `double_spiral`
is a 1-D curve in 3-D, so the Euclidean scale varies 11-fold along it, and the
flagship improves the NLL by 69%. The mixture datasets keep all components at
similar density, so `rho` barely varies and there is nothing for the exponent to
exploit. This is the flagship working as designed, on the data class where its
central assumption holds.

**The anisotropy axis does not currently pay.** `beta = 0` is selected by
validation on 6 of 6 datasets. The cause is identified rather than merely
observed: the local covariance is `d × d` and estimated from `k` neighbours, and
at `d = 6…12` with `k ∈ {15, 30}` that estimate is under-determined. The loss
narrows monotonically with `k` (−3.24% at `k = 8` to −1.57% at `k = 60`), which
identifies estimation noise rather than absent anisotropy — `aniso_gmm` has
per-axis σ spread over `[0.25, 2.0]`, a factor of 8.

The design document's own protection **P3** says that if validation selects the
degenerate corner, "a threshold failure is the *correct* conclusion rather than
something to tune away". The three options are (a) raise `k` until the local
covariance is estimable, (b) shrink `C_i` toward its isotropic average so partial
whitening is available at small `k`, or (c) report the gate as unmet. **This
package does not choose** — it records the measurement, and the flag is visible in
every report row.

---

## 6. Numerical contracts worth knowing

| contract | why |
|---|---|
| `np.trapezoid`, not `np.trapz` | removed in NumPy 2 |
| `cdist` with the diagonal set to `inf` | a point must never be its own neighbour |
| `subset_by_index` for every `N × N` solve | 6.2× |
| `logsumexp` everywhere a sum of exponentials appears | no overflow, no underflow |
| symmetrise the **directed** weights exactly once | symmetrising twice halves one-directional edges — measured 25% error |
| truncate neighbours by **kernel weight**, not Euclidean distance | an anisotropic kernel's nearest points in Mahalanobis terms may be far in Euclidean terms; measured L1 error 0.665 → 0.241 |
| divide the local covariance by `rho²` | makes the kernel family scale-covariant, restoring invariant I9 to machine precision |
| normalise local eigenvalues to unit geometric mean | makes `beta` a pure shape parameter rather than a hidden bandwidth multiplier |

---

## 7. Error codes

| segment | class | scenario |
|---|---|---|
| E1xx | `ConfigError` `ConfigValidationError` `ConfigRangeError` `ConfigUnknownKeyError` | configuration |
| E2xx | `DataError` `DatasetNotFoundError` `ShapeMismatchError` `InsufficientSamplesError` `SeedCollisionError` | data |
| E3xx | `ModelError` `BackendUnavailableError` `FitFailedError` `NumericalError` `NumericalOverflowError` | model and numerics |
| E4xx | `EvalError` `LeakageError` `LeakageTestSetTouchedError` `LeakageModelNotTrainOnlyError` `InsufficientSeedsError` | evaluation |
| E5xx | `IOErrorBase` `ArtifactNotFoundError` `SerializationError` `AtomicWriteError` | artefacts |

Codes are asserted by tests, not by message text, so the wording can improve
without breaking callers. An unknown `ENV_DENSFORGE_*` key **raises** rather than
being ignored: a silently-ignored deployment variable is a classic way for staging
to serve stale numbers while everyone believes the setting took effect.

---

## 8. Artefacts

`benchmark.json` is written atomically — temp file, then `os.replace` — so an
interrupted run cannot leave a truncated file that `json.load` accepts and that
means nothing.

Reproducibility comparison goes through `canonical()`, which rounds floats to 10
decimal places, sorts keys, and drops every `*_sec` field. Rounding removes
last-bit ULP jitter that survives even with identical inputs and identical code;
the filtering removes wall-clock timing, which is the one field that cannot be
deterministic by construction.
