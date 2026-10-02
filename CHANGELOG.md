# Changelog

All notable changes to DensForge are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## A note on threshold history

**No published threshold has ever been lowered.** The design document proposes a
median NLL win (gate G-D1); the measured median on the repo's own generators is
**−1.05%** (G-D1 FAIL), and that gap is reported rather than closed by moving the
bar. Every version entry below records what was measured and why a decision was
taken, so the reasoning survives the person who took it.

---

## [0.1.0] — 2026-10-03

First working release. Implements the flagship `DensFuse`, four Tier-0 density
baselines, six Tier-1 pure-NumPy fallbacks, six manifold baselines, a staged
coordinate-descent tuner, a four-assertion leakage firewall, a CLI, and 400+
tests covering 32 design-document invariants.

### Added

**Flagship.** `DensFuse` (`density/flagship.py`) — locally-adaptive anisotropic
kernel density fused with a density-balanced diffusion embedding. Five
components in a closed loop iterated to a fixed point: local scale, local
whitening, normalised heteroscedastic kernel mixture, density-balanced diffusion
operator, geodesic local scale.

**Baselines.** `SkKernelDensity` (the decisive reference), `ScipyGaussianKDE`,
`SkGaussianMixture`, `SkBayesianGaussianMixture`; `SkPCA`, `SkIsomap`, `SkLLE`,
`SkSpectralEmbedding`, `SkMDS`, `SkTSNE`; plus `ManifoldFuse` as the manifold-side
ablation that isolates the density channel.

**Tier-1 offline fallbacks.** `NumpyKDE`, `NumpyGMM`, `NumpyIsomap`,
`NumpySpectral`, `NumpyDiffusionMaps` — no scikit-learn dependency, so a missing
backend yields a measured row rather than a skipped one. All use partial
eigendecomposition (`subset_by_index`), 6.2× faster than a full solve at
`N = 2000, d = 16`.

**Data.** 11 registered datasets: the 6 headline generators of the design
document, the `iso_gauss` anti-cheat control, and 4 added on 2026-10-03 by the
algorithm side (`sparse_dim`, `hetero_density`, `heavy_tail`,
`uniform_hypercube`) to close the gap between the benchmark and the specification.
`HEADLINE_DATASETS` remains the six; the extras are reachable by name and fully
covered by the registry-wide tests.

**Infrastructure.** Three-stage API with a typed error hierarchy (E1xx–E5xx);
four leakage assertions (seed offset, call-history ids, content digests, minimum
distance); staged coordinate descent with an auditable budget; atomic JSON
artefacts; a CLI with `run` / `bench` / `demo` / `info`; CI on Python 3.12 and
3.13.

### Fixed — defects in the design documents

Six substantive defects were found by verifying the design documents'
mathematics against measured behaviour. Each is documented with the measurement
that exposed it in [`docs/math_verification.md`](docs/math_verification.md).

1. **The quadratic form contradicted the bandwidth matrix.** The documents define
   `H_i` and then build the kernel from a *different* `H_i`. Measured
   consequence: the estimator is not scale covariant, so invariant I9 (declared
   blocking) fails, and exact normalisation holds only at `beta = 0`. Fixed by
   dividing the local covariance by `rho²`, which restores scale covariance to
   1.4e-15.
2. **The local shape was not volume-normalised**, so `beta` acted as an
   unadvertised bandwidth multiplier rather than a shape parameter — making the
   anisotropy axis untestable. Fixed by normalising local eigenvalues to unit
   geometric mean; `det H` is now independent of `beta` by construction.
3. **Double symmetrisation halved every one-directional k-NN edge.** The
   pseudocode symmetrises `W`, then re-weights the already-symmetrised matrix and
   symmetrises again. Measured error 25% of the edge weight, which broke invariant
   I14. Fixed by symmetrising the directed weights exactly once.
4. **The geodesic scale was inert as specified.** On a full k-NN graph the k-th
   Euclidean neighbour *is* a direct edge, so the k-th shortest path is that edge:
   measured `rho_geo / rho_euclid = 1.000000` exactly, for every point. Fixed by
   building the geodesic graph from a reduced neighbourhood, giving a measured
   ratio of 1.1117.
5. **Euclidean truncation of an anisotropic kernel** discarded high-mass terms.
   Fixed by ranking the candidate set by kernel weight; measured L1 distortion at
   `m = 128, N = 400` fell from 0.665 to 0.241.
6. **The `gamma` grid could not reach its own optimum.** The Lazaridis pilot is a
   mean-shift bandwidth, systematically too wide for NLL minimisation: measured
   `sigma0 = 10.79` against a val-optimal fixed bandwidth of 2.5, so the
   correcting factor is `gamma ≈ 0.23` while the documented grid floored at 0.5.
   Grid extended downward to 0.0625.

### Re-scoped — invariants whose stated form is unassertable or false

Each is asserted in a form that is both true and meaningful, with the original
and its refutation recorded:

* **I7** asked for `err(128) / err(N) < 1e-3` where `err` is *defined* as the
  distance to the `m = N` answer, so the ratio is `0 / 0`. Now asserts monotonicity
  plus a small final error.
* **I13** asserted that rows of `D^-1/2 W D^-1/2` sum to at most 1. Measured
  maximum: 1.156. Now asserts the properties that hold — symmetry, `|eig| ≤ 1`,
  `max |S_ij| ≤ 1`, and rows of the random-walk operator `P = D^-1 W` equal to 1.
* **I18** asked for a limit as the kernel width goes to zero; it does not exist,
  since the affinity matrix tends to the indicator of exact coincidence. Now
  asserts that the leading embedding axis recovers the dominant linear direction
  on data where that direction is identifiable (measured 0.964–0.969).
* **I27**'s proposed check compares two *consecutive* draws from one stream, so it
  can never pass. Now compares the global `RandomState`'s internal state before
  and after, and includes a control proving the detector would notice real
  pollution.

### Measured, and reported rather than tuned away

* `beta = 0` is selected on 6 of 6 datasets. The local covariance is `d × d` and
  estimated from `k` neighbours; at `d = 6…12` the estimate is under-determined.
  The loss narrows monotonically with `k` (−3.24% at `k = 8` to −1.57% at
  `k = 60`), which identifies estimation noise rather than absent anisotropy.
* The median NLL reduction across the six headline datasets is **+0.16%**,
  against the design document's proposed 8% gate. **The gate is not met.**
* The self-tuning axis wins where its assumption holds: `double_spiral`, whose
  local scale varies 11-fold, improves by **69%**. The `eta` sensitivity across
  the six datasets tracks the local-scale range almost perfectly (24.6% on
  `double_spiral`, 0.05% on `t_mixture`).

### Fixed — defects found in this implementation

* `NumpyKDE` used `- d·log(2π)` instead of `- ½·d·log(2π)`, making the fallback
  density wrong by a factor of `√(2π)`. Every other symptom was absent: finite
  output, correct sign, correct shape, correct point ranking, and an NLL that was a
  *constant* offset from the truth. Only an integral check finds that class of
  bug, which is why `test_numpy_kde_integrates_to_one` exists.
* The pipeline passed a bandwidth candidate to a **seed** parameter, so the
  baseline bandwidth search raised `TypeError` and every bandwidth-tuned row was
  recorded as failed.
* The CLI printed `(skipped)` in the NLL column for manifold-only methods, which
  have no density by construction. That reports a *shape* as a *failure* and
  inflates the skip count; those cells now print an em dash.
* `MDS.n_init`'s scikit-learn 1.9 default is **1**, not the 4 quoted in the design
  document. A contract test asserting 4 would have failed a correct implementation
  and "fixed" it by pinning stale behaviour.

### Known limitations

* `d ≤ 16`, `n ≤ 2000`. The diffusion operator is a dense `O(N³)` solve; there is
  no sparse or approximate path.
* Benchmark NLL is not comparable across datasets.
* Trustworthiness is fixed at `k = 5` project-wide; embedding mode must be stated.
* `DensForge` requires SciPy and scikit-learn. The Tier-1 fallbacks cover a missing
  *algorithm* backend, not a missing SciPy.
