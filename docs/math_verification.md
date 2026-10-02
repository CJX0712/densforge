# Math verification log — DensForge flagship

Author: 晨星 <CJX0712@users.noreply.github.com>

Every number here was measured on this machine, not estimated:
Python 3.13.14 · numpy 2.5.3 · scipy 1.18.1 · scikit-learn 1.9.1 · Windows 11 · single thread.

The design documents (01-algorithm-design.md, 02-system-architecture.md) were used as
the specification. Where their mathematics was internally inconsistent or
self-contradictory, the mathematics was corrected and the conflict is recorded below.
Nothing was "fixed" by loosening a tolerance.

---

## A. Defects found in the design documents

### D1 — The quadratic form contradicts the bandwidth matrix (BLOCKING)

01 §3.2 states the bandwidth matrix as

    H_i = gamma^2 sigma_i^2 V diag(lam^beta) V^T

and then states the quadratic form as

    (x-x_i)^T H_i^-1 (x-x_i) = (1/gamma^2 sigma_i^2) sum_m lam^beta (v_m.u)^2

The second line is the quadratic form of `H_i = gamma^2 sigma^2 V diag(lam^-beta) V^T`.
So the document defines `H` and then uses a *different* `H` when forming the kernel. The
§4.2 code inherits the same split: `R2 = V diag(lam^beta) V^T` is used as the
**precision** matrix while `logdetH = 2 d log(sigma) - beta sum log(lam)` is the log
determinant of the **inverse** convention.

Consequence, measured: with the two conventions mixed, `H_i(cX) != c^2 H_i(X)` for any
`beta != 0`, so the estimator is not scale covariant and invariant **I9** (declared 🔴
blocking) fails. Exact-normalisation invariant **I1** (also 🔴) holds only at `beta = 0`.

**Resolution.** One convention, chosen so that `beta = 1` means "full local whitening",
which is what §3.2 says the exponent *means*:

    Chat_i   = C_i / rho_i^2                      (dimensionless local shape)
    H_i      = gamma^2 sigma_i^2 V diag(lam^beta)^-1 V^T
    quad_i   = (1/gamma^2 sigma_i^2) sum_m lam^beta (v_m.u)^2
    logdet H = 2 d log(gamma sigma_i) - beta sum_m log lam

Dividing `C_i` by `rho_i^2` is what makes the family scale covariant: `Chat(cX) = Chat(X)`,
so `H(cX) = c^2 H(X)` exactly. Measured scale-equivariance deviation after the fix:
**8.0e-16** (beta=0), **1.4e-15** (beta=0.5), **1.4e-15** (beta=1) — machine epsilon.

### D2 — The local shape must be volume-normalised, or `beta` is not a shape parameter

With `Chat = C / rho^2` alone, increasing `beta` also inflates the kernel volume
(`det H` shrinks as `lam^-beta`), so `beta = 1` acts as a bandwidth increase and
*loses* to `beta = 0` on anisotropic data. Measured on anisotropic 4-D, val NLL:

| gamma | beta=0 | beta=1 | change |
|---|---|---|---|
| 0.25 | 173.87 | 49.59 | **-71%** |
| 0.50 | 17.10 | 9.67 | **-43%** |
| 1.00 | 9.16 | 9.03 | -1.3% |
| 2.00 | 13.54 | 13.54 | +0.0% |
| 4.00 | 19.00 | 19.00 | +0.0% |

The gain is entirely a bandwidth effect at small `gamma` and vanishes at large `gamma`.

**Resolution.** Normalise `lam` to unit geometric mean:

    lam_f <- lam_f / exp(mean(log lam_f))

Then `det H_i` is **independent of beta** and `beta` is a pure shape parameter. After
the fix, `beta = 1` wins at every `gamma` up to the point where a very wide kernel
washes the shape out, and invariant **I1** is exact at `beta = 1` as well
(measured integral `1.000000000000` for beta in {0, 0.5, 1}).

### D3 — Double symmetrisation silently halves asymmetric edges

§4.2 step (5) symmetrises `W`, then step (7) reads `W[rows, cols]` off the
*already symmetrised* matrix and symmetrises again. For a k-NN edge that exists in
only one direction, the value has already been halved by `(W + W.T)/2`; halving it a
second time gives `0.25 v` where `0.5 v` is correct. Measured max deviation of
`Wb - W` at `alpha = 0`: **2.5e-01** — a 25% error on real edges, which breaks
invariant **I14** (declared 🔴, "`alpha = 0` must reduce to `W`").

**Resolution.** Keep the directed weight matrix and symmetrise exactly once per output.
Measured after the fix: `max|Wb - W| = 0.0` at `alpha = 0`, bit for bit.

### D4 — The geodesic scale is a mathematical no-op as specified

§3.4 / §4.2 build the k-NN graph and call `dijkstra(..., limit=k)`, then take the
k-th smallest finite distance. But the k-th Euclidean nearest neighbour **is a direct
edge of the graph**, so the k-th shortest path is that single edge and
`rho_geo == rho_euclid` exactly. Measured ratio: **1.000000** for every point. The
component that the design calls "the third edge of the closed loop" was inert.

**Resolution.** Build the geodesic graph from a reduced neighbourhood
(`k_graph = max(3, k // 3)`) so the k-th shortest path is genuinely multi-hop and
measures accumulated path length. Measured on a noisy swiss roll: mean ratio
`rho_geo / rho_euclid = 1.1117`, i.e. the geodesic scale is now strictly larger, as
invariant **I31** requires.

### D5 — "Rows of `S = D^-1/2 W D^-1/2` sum to at most 1" is false

§8 I13 asserts `S.sum(1) <= 1`. That is not a property of the symmetric normalisation.
Measured on a real affinity matrix: `max row sum = 1.1557`. The bound that *does* hold
is on the entries, not the rows: `S_ij = W_ij / sqrt(d_i d_j) <= 1` because
`W_ij <= d_i` and `W_ij <= d_j`.

**Resolution.** Assert the properties that are actually true and are actually
load-bearing:

| property | measured |
|---|---|
| `S == S.T` exactly | holds |
| `max abs(S_ii) <= 1` | holds |
| `max S_ij <= 1` | holds |
| `eig(S)` within `[-1, 1]` | measured `[-0.2809, 1.000000]` |
| rows of `P = D^-1 W` sum to 1 | measured max deviation `3.3e-16` |

`P` is the actual Markov chain; `S` is its symmetrically normalised, spectrally
conjugate form. Row sums belong to `P`, eigenvalues belong to `S`.

### D6 — Euclidean truncation of an anisotropic kernel (engineering, not a doc defect)

§3.3 truncates the kernel sum to the `m` nearest training points **by Euclidean
distance**, while the kernel weight is an anisotropic Mahalanobis distance. A point
that is far in Euclidean terms can be near in kernel terms, so truncation discards
high-mass terms. Measured L1 distortion against the full sum at `m = 128`, `N = 400`:
**0.665** nats with Euclidean selection.

**Resolution.** Two-stage selection: take a generous Euclidean candidate set
(`min(N, 4m)`) and keep the `m` largest **kernel log-weights**. Measured distortion
falls to **0.241**, and to **6.8e-07** at `m = 384`.

---

## B. Invariants whose *formulation* is unassertable

### I7 — "assert err(128)/err(N) < 1e-3" is `0/0`

§8 I7 defines `err(m) = || p^(m) - p^(N) ||` and then asks for
`err(128) / err(N) < 1e-3`. By construction `err(N) = 0`, so the ratio is undefined.
No implementation can satisfy it.

**Resolution.** Assert the two claims that carry the intent and are measurable:

1. `err(m)` is non-increasing in `m` (measured: `2.4, 1.8, 1.2, 0.66, 0.24, 0.021, 6.8e-07`
   for `m = 8 … 384` — strictly monotone, all configurations);
2. the relative distortion at the largest usable `m` is small
   (`1.7e-07` at `N = 400, m = 384`).

### I8 — Monotone convergence in `N` needs a tolerance band and a wider `gamma` grid

§8 I8 asks for `np.all(np.diff(err) <= 0)` (with a 1% allowance) across
`N in {200, 500, 1000, 2000, 4000}`. With the `gamma` grid `{1, 4, 16, 64, 256, 1024}`
the estimator selects `gamma = 4` at every `N`, and the measured L1 error sequence
`0.2401, 0.1012, 0.0813, 0.0584, 0.0603` **increases** on the last step: Monte-Carlo
noise in the L1 estimate, not a property of the estimator. The measured val NLL
sequence `1.4937, 1.4461, 1.4436, 1.4352, 1.4390` is likewise flat to within 0.004
after `N = 1000`, which is expected — a 1-D Gaussian's optimal bandwidth shrinks as
`N^-1/5`, so the NLL improvement per doubling is `O(1/N)`.

**Resolution.** Nested prefixes of **one** sample (removes resampling noise) and
**one** `gamma` chosen once at the largest `N` and held fixed (removes selection
noise), so the only varying quantity is `N`:

| `N` | 200 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|
| L1 error to `N(0,1)` | 0.1517 | 0.1402 | 0.1184 | 0.0810 | 0.0800 |
| ratio to previous | — | 0.925 | 0.844 | 0.684 | 0.988 |

Monotone under a 1% band, final error 0.0800 < 0.1.

### I4 — "`beta = 1` beats `beta = 0`" is a performance claim, not an invariant

§8 I4 asserts `nll(beta=1) < nll(beta=0)` on anisotropic data, with the escape
hatch "若不成立则记录但不失败" (record but do not fail).

Measured on the real generators at `n = 400`:

| dataset | best `gamma` | gain of `beta=1` | grid points won |
|---|---|---|---|
| `aniso_gmm` (d=8) | 0.25 | **-2.6%** | 1/5 |
| `t_mixture` (d=6) | 0.25 | **-2.0%** | 1/5 |
| `aniso_gmm`, wider scales | 0.25 | **-2.7%** | 1/5 |

`beta = 1` loses everywhere, and the loss **narrows monotonically with `k`**:

| `k` | 8 | 12 | 15 | 20 | 30 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| gain of `beta=1` | -3.24% | -2.77% | -2.64% | -2.14% | -1.90% | -1.61% | -1.57% |

That trend identifies the cause: the local covariance is `d x d` and estimated from
`k` neighbours. At `d = 8`, `k = 15` it is under-determined, so whitening amplifies
the directions with the least evidence. It is **not** that the data lacks
anisotropy — `aniso_gmm` has per-axis `sigma` spread over `[0.25, 2.0]`, a factor
of 8. It is that the anisotropy is not *resolvable* from 15 neighbours in 8
dimensions. The trend also predicts the effect reverses at larger `k`, which is a
falsifiable prediction rather than a rationalisation.

**Resolution.** I4 splits into two claims, and only the one that is a property of
the estimator is asserted:

* **mechanism (asserted)** — `beta` changes the kernel's *shape* and not its
  *volume*. Verified exactly: `det H_i` is independent of `beta` by construction
  after the unit-geometric-mean normalisation (finding D2), and `beta = 0` collapses
  the quadratic form to the pure Euclidean one to 5.4e-16 relative error (I3).
* **performance (recorded, not asserted)** — whether whitening helps is a
  dataset-and-`k`-dependent empirical question. The test records the sign and the
  `k`-trend, and fails only if `beta` has *no effect whatsoever*, which would mean
  the hyper-parameter is not reaching the estimator.

The design document already prescribes exactly this split for I4.

### I18 — "as the kernel width goes to zero the embedding becomes the top-m PCs" has no limit

§8 I18 asks for the limit `sigma -> 0`. It does not exist: as `sigma -> 0` the affinity
matrix `W_ij = exp(-d_ij^2 / 2 sigma^2)` tends to the indicator of exact coincidence,
which for continuous data is the zero matrix. There is nothing left to decompose.

The natural surrogate — "the embedding spans the data's linear subspace" — is also
unassertable as stated. Measured canonical correlations between the embedding and the
data's principal subspace:

| data | correlations | verdict |
|---|---|---|
| isotropic Gaussian, d=3, m=2 | `0.937, 0.917` | passes |
| isotropic Gaussian, d=6, m=2 | `0.892, 0.531` | fails |
| isotropic Gaussian, d=8, m=3 | `0.893, 0.873, 0.052` | fails |
| isotropic Gaussian, d=12, m=3 | `0.871, 0.731, 0.507` | fails |
| 2-D subspace in 6-D, thickness 1e-4 | `0.832, 0.002` | fails |

Two independent reasons, both structural:

1. For data that is isotropic **within** its own subspace, every orthonormal basis of
   that subspace is equally valid, so correlating against one *chosen* basis measures
   an arbitrary rotation, not a quality difference. The comparison must be against the
   subspace, and even then the higher-order directions of a near-degenerate local
   covariance are dominated by the spectral floor.
2. In `d >= 6` the k-NN local covariance of an isotropic cloud is itself poorly
   conditioned, so `beta = 1` whitening amplifies estimation noise in exactly the
   directions with the least data.

**Resolution.** I18 is asserted on data with an **identifiable dominant
direction**, where the target is well defined. Measured canonical correlation
between the leading embedding axis and the top principal axis:

| data | correlation | verdict |
|---|---|---|
| isotropic Gaussian d=3 | 0.933 | pass |
| isotropic Gaussian d=8 | 0.866 | pass |
| isotropic Gaussian d=6 | 0.274 | fail |
| isotropic Gaussian d=12 | 0.150 | fail |
| **anisotropic d=4** | **0.964** | **pass** |
| **anisotropic d=6** | **0.969** | **pass** |
| **anisotropic d=8** | **0.967** | **pass** |

The isotropic failures follow directly from point 1 above: with no preferred
direction in the data, "the top principal axis" is not a well-defined target and
the diffusion operator may pick any direction in the degenerate subspace.

This captures the design's intent — the diffusion operator recovers dominant
linear structure when the data *is* linearly low-dimensional — without asserting a
limit that does not exist or a rotation-invariance the metric does not have. See
`tests/test_invariants.py::test_i18_dominant_axis_alignment`.

---

## B3. A defect found in this implementation, not in the design documents

### The flagship's default bandwidth is miscalibrated by roughly 4x

The Lazaridis pilot is a **mean-shift** bandwidth, not a density-estimation
bandwidth. Its `N / (d + 2)` factor exists to make the kernel wide enough for the
density gradient to have a well-defined mode, which makes it systematically too
wide for NLL minimisation.

Measured on `aniso_gmm` (d=8, n=400): `sigma0 = 10.79`, while the
validation-optimal *fixed* bandwidth is `h = 2.5`. The compensating factor is
`gamma ~ 0.23` — **below the design document's grid floor of 0.5**, so the
documented grid cannot reach the optimum at all.

| `gamma` | 0.0625 | 0.125 | **0.25** | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|
| val NLL (k=15, beta=0) | 44.73 | 23.31 | **20.26** | 22.44 | 26.84 | 32.06 |

**Resolution.** The `gamma` grid is extended downward to
`{0.0625, 0.125, 0.25, 0.5, 1, 2, 3, 4, 6}`, and the pilot's documented role is
"a scale-setting prior to be corrected by `gamma`" rather than "a good default".
The default `gamma = 1.0` remains a starting point, not a recommendation, and the
benchmark always reports the validation-selected value.

### The self-tuning exponent is nearly inert on the headline datasets

`eta` scales the bandwidth by `(rho_i / median(rho))^eta`. On `aniso_gmm`,
`rho / median(rho)` spans only `[0.589, 2.047]`, so even `eta = 2` produces less
than a 4x spread. Measured val NLL across `eta in {1/(d+4), 1/d, 1/2, 1, 2}`:
`20.6952, 20.6956, 20.7083, 20.7464, 20.8694` — a 0.8% total range.

This is a property of the **data**, not a bug: `aniso_gmm` draws six components
with equal weight and similar per-axis spread, so the local scale barely varies and
there is nothing for `eta` to exploit. A dataset with genuinely varying density is
where this axis pays. Recorded here so the near-flat `eta` sensitivity is not
later mistaken for a broken code path — `test_hyperparameters_reach_the_estimator`
checks that different `eta` values produce different outputs, which is the actual
defect class this guards against.

---

## C. Confirmed-correct design decisions

| item | measurement |
|---|---|
| `kneighbors()` with `X=None` excludes self | `dist[:, 0] > 0` for all points |
| `rho = dist[:, k-1]` after self-exclusion | exact |
| Lazaridis pilot `log sigma0^2 = mean(log rho1^2) + log N - log(d+2)` | sane magnitudes; the sign-flipped variant gives `sigma0 ~ 1e-3` |
| `I1` exact normalisation, `d = 1`, `scipy.integrate.quad` | `1.000000000000` for beta in {0, 0.5, 1} |
| `I9` scale equivariance | `1.4e-15` max deviation over `c in {0.5, 2, 10}` |
| `I10` translation equivariance | `7.1e-15` over `a in {0, 5, -100}` |
| `I11` permutation equivariance | `3.6e-15` |
| `I19` rotation invariance | `3.6e-15` |
| `I12` graph has zero diagonal, symmetric, non-negative | holds for `W` and `Wb` |
| `I15` leave-one-out log density clipped to `[-4, 4]` | holds |
| `I16` alpha = 1 does not explode the embedding | `std ratio = 1.005` (threshold 100) |
| `I17` `Psi^T D Psi = I` | `7.4e-12` max deviation |
| `I5` NLL is U-shaped in `gamma`, argmin interior | argmin at `gamma = 1.0` of `{0.125 … 32}` |
| `I30` closed loop reaches a fixed point | relative change `1.5e-03` after round 1, then exactly `0` |
| I1 at `beta = 0` via `np.trapezoid` (not `np.trapz`) | numpy 2 contract holds |
| sklearn 1.9 `Isomap` has no `random_state` | confirmed by `inspect.signature` |
| sklearn 1.9 `BayesianGaussianMixture(n_components=...)` is keyword-only | `TypeError` when positional |

## D. Environment facts that differ from the design document

| claim in design | measured on this machine |
|---|---|
| 01 §6.7: "`MDS.n_init` default is 4" | **1** |
| 01 §2.2: "TSNE / MDS budget 12.9 s at n=1500" | not re-measured; budgets re-derived from scratch (see `docs/architecture.md`) |
| 01 §8 I29: "`MDS().get_params()['n_init'] == 4`" | **1** — the invariant is corrected accordingly |

---

## D. Where the flagship actually wins, and where it does not

Full val-based search per dataset (`k x gamma x beta x eta`, n=400, seed 0), test
NLL against `fixedkde(val-bw)`:

| dataset | d | `rho/median(rho)` range | fixedkde | flagship | delta | selected |
|---|---|---|---|---|---|---|
| `aniso_gmm` | 8 | [0.59, 2.05] | 20.713 | 20.687 | **+0.13%** | k=30, g=0.25, b=0 |
| `swiss_roll` | 3 | [0.56, 1.55] | 8.709 | 8.719 | -0.12% | k=30, g=0.125, b=0 |
| `double_spiral` | 3 | [0.30, 3.29] | 2.289 | 0.709 | **+69.0%** | k=30, g=0.125, b=0 |
| `circles` | 2 | [0.32, 2.55] | 2.884 | 2.732 | +5.3% | k=15, g=0.125, b=0 |
| `t_mixture` | 6 | [0.62, 2.17] | 14.253 | 14.226 | +0.19% | k=30, g=0.25, b=0 |
| `manifold_noise` | 12 | [0.77, 1.42] | 24.795 | 25.433 | -2.57% | k=15, g=0.5, b=0 |

### Finding 1: the self-tuning axis pays exactly where local scale varies

`eta` sensitivity (validation NLL spread across `eta in {1/(d+4), 1/4, 1}`) tracks
the `rho/median(rho)` range almost perfectly:

| dataset | `rho/med` range | `eta` spread |
|---|---|---|
| `double_spiral` | 11.0x | **24.6%** |
| `circles` | 8.0x | 1.4% |
| `manifold_noise` | 1.8x | 0.5% |
| `swiss_roll` | 2.8x | 0.24% |
| `aniso_gmm` | 3.5x | 0.23% |
| `t_mixture` | 3.5x | 0.05% |

`double_spiral` is a 1-D curve in 3-D, so the Euclidean scale varies by 11x along
it, and the flagship improves the NLL by **69%**. The mixture datasets keep all
components at similar density, so `rho` barely varies and there is nothing for the
self-tuning exponent to exploit. This is the flagship working as designed, on the
data class where its central assumption holds.

### Finding 2 (material): `beta = 0` is selected on all six datasets

The design document's protection **P3** ("degeneracy corner audit") says: if
validation picks `beta = 0` and `alpha = 0` on most datasets, the flagship is not
using the new hypothesis directions, and a threshold failure is the *correct*
conclusion rather than something to tune away.

Measured: **`beta = 0` is selected on 6 of 6 datasets.** With the `k`-trend from
section B1, the cause is identified rather than merely observed: full local
whitening needs a `d x d` covariance estimated from `k` neighbours, and at
`d = 6..12` with `k in {15, 30}` that estimate is under-determined. The whitening
amplifies estimation noise faster than it removes bias.

Consequences for the threshold discussion (Phase 3, not decided here):

* the anisotropy axis contributes nothing measurable at the design's `k` values on
  these datasets;
* the median relative reduction across the six datasets is **+0.16%**, against the
  design's G-D1 gate of 8%;
* the flagship still wins decisively on `double_spiral` (+69%) and moderately on
  `circles` (+5.3%), so the *median* gate is the binding constraint, not a
  universal failure.

The honest options are (a) raise `k` substantially so the local covariance becomes
estimable, (b) introduce a shrinkage estimator between `C_i` and its isotropic
average so partial whitening is possible at small `k`, or (c) report the gate as
not met. **This report does not choose** -- it records the measurement. Note that
option (b) is a real algorithmic contribution rather than a threshold adjustment,
and is the direction the design document's own §3.2 "partial whitening" row
(`0 < beta < 1`) points at but never implements.
