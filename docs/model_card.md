# Model card — DensForge

Author: 晨星 &lt;CJX0712@users.noreply.github.com&gt; · License: MIT · Version 0.1.0

---

## Summary

DensForge estimates a probability density and a low-dimensional embedding from the
same unlabelled sample, in one closed loop. The flagship `DensFuse` couples a
locally-adaptive anisotropic kernel density with a density-balanced diffusion
embedding: the local scale `rho` sets each kernel's shape, the resulting density
corrects the diffusion operator's sampling-density bias, and the embedding
refines `rho` through the geodesic structure. The loop is iterated to a fixed
point.

| | |
|---|---|
| **Task** | unsupervised density estimation + manifold learning |
| **Input** | unlabelled `(n, d)` float array, `d ≤ 16`, `n ≤ 2000` |
| **Output** | normalised log-density at arbitrary points; `m`-dimensional embedding |
| **Primary metrics** | held-out test NLL (nats, ↓) · trustworthiness (↑) |
| **Secondary** | continuity, calibration error, local-structure fidelity |
| **Compute** | CPU only; `O(N·k·d)` per fit plus one `O(N³)` eigendecomposition |
| **Determinism** | bit-exact given a seed and pinned thread count |
| **Training data** | none — this is not a learned-weights model |

---

## Intended use

**In scope.** Exploring the shape and density of an unlabelled sample whose size
is known to be in the low thousands and whose dimensionality is at most 16;
establishing a reproducible baseline against which a new density or manifold
method can be compared; teaching the numerical subtleties of adaptive kernel
estimation.

**Out of scope, deliberately.**

* **Data beyond ~10⁶ points.** The diffusion operator is a dense `N × N`
  eigendecomposition, `O(N³)`. At `N = 2000` that is 32 MB and a few seconds; at
  `N = 10000` it is 800 MB and minutes. The Tier-1 fallbacks use the same dense
  algebra. There is no sparse or approximate path, and a result at that scale
  obtained by a subsample is not a result about the full data.
* **`d > 16`.** `Config` refuses it with `E102`. The local covariance is `d × d`
  per point, and past 16 the estimation noise in that covariance dominates the
  anisotropy it is meant to resolve — the anisotropy axis stops paying (see
  *Limitations*).
* **Streaming or online data.** The density is a batch estimator over the whole
  sample.
* **Anything supervised.** No labels, no downstream task, no fine-tuning.

---

## Evaluation protocol

Splits are **three independent draws**, never a shuffle of one pool:

```
train = make(n_train,        seed)          # the only data fit() may see
val   = make(n_train // 5,   seed + 500)    # hyper-parameter selection only
test  = make(n_test,         seed + 10_000) # final scoring only
```

Slicing one pool would make the halves share sampling noise, which biases test NLL
optimistically. Because the test seed is the train seed plus a fixed offset, the
independence is *provable* rather than merely likely.

Four assertions guard the boundary, and all four must pass before a test number is
published:

| # | assertion | mechanism |
|---|---|---|
| A1 | seed isolation | `test_seed == train_seed + 10_000` |
| A2 | metric-source isolation | every array scored during tuning is recorded; the test array's id must not appear |
| A3 | model isolation | the model records a **content digest** of everything fitted on — a *copy* of the test array has a different `id()` and the same content, and is caught |
| A4 | shape overlap | `min(cdist(train[:64], test)) > 0` |

**Every baseline is tuned on validation data too.** Thresholds are measured
against `fixedkde(val-bw)`, a fixed-bandwidth KDE whose bandwidth was chosen on
validation data — not the library default. Measured on a d=6 anisotropic mixture,
a val-tuned bandwidth scores 16.69 against 33.92 for `gaussian_kde('scott')`: a
factor of two. Tuning only the flagship would have manufactured the result.

**Reporting.** `mean ± std` over ≥ 3 seeds, `ddof = 1`. A difference is called
significant only when it exceeds `0.5 · (σ₁ + σ₂)` — a deliberately strict bar, so
a gap that could be noise is reported as no difference. A skipped backend
produces a `skipped` row with a reason; it is never zero-filled or nan-filled.

---

## Results, and what they do not show

Measured at `n = 400`, one seed, both sides validation-tuned
(`docs/math_verification.md` §D has the full sweep):

| dataset | local-scale range | flagship vs `fixedkde(val-bw)` |
|---|---|---|
| `double_spiral` | 11.0× | **+69.0%** |
| `circles` | 8.0× | +5.3% |
| `t_mixture` | 3.5× | +0.19% |
| `aniso_gmm` | 3.5× | +0.13% |
| `swiss_roll` | 2.8× | −0.12% |
| `manifold_noise` | 1.8× | −2.57% |

**Read this table as one positive result and one negative one.**

*The positive result.* The self-tuning axis pays exactly where the local scale
varies. `double_spiral` is a 1-D curve in 3-D, so the Euclidean scale varies
11-fold along it, and the flagship improves the NLL by 69%. The mixture datasets
hold all components at similar density, so `rho` barely varies and there is nothing
for the exponent to exploit. That is the mechanism working where its assumption
holds, not a fluke.

*The negative result.* **`beta = 0` is selected on 6 of 6 datasets.** Full local
whitening needs a `d × d` covariance estimated from `k` neighbours; at
`d = 6…12` with `k ∈ {15, 30}` that estimate is under-determined, so whitening
amplifies noise faster than it removes bias. The loss narrows monotonically with
`k` (−3.24% at `k = 8` to −1.57% at `k = 60`), which identifies the cause rather
than merely observing it.

**The median reduction across the six datasets is +0.16%, against the design
document's proposed 8% gate. That gate is not met.** The report says so rather than
lowering the bar. Fixing it means one of: raising `k` until the local covariance is
estimable, adding shrinkage between `C_i` and its isotropic average so partial
whitening is available at small `k`, or reporting the gate as unmet. Choosing among
those is a research decision, not an engineering one, and the selected `beta` is
visible in every report row so the gap cannot be quietly forgotten.

---

## Limitations

* **Synthetic evaluation data does not transfer.** Every number above is measured
  on generators this project wrote. Real data has structure none of the six
  datasets contains, and a benchmark score on synthetic data is evidence about the
  method, not about anyone's dataset.
* **Benchmark NLL is not comparable across datasets.** Different dimensionality,
  different scale, different intrinsic geometry. Compare methods *within* a
  dataset, never across the table.
* **Trustworthiness depends on `n_neighbors`** and can be "improved" by tuning it.
  It is fixed at `k = 5` project-wide and every number must state it.
* **Embedding mode must be stated.** `inductive` (query points never enter the
  graph) and `transductive` (they do) give different numbers. Both are reported;
  mixing them in one column is not a comparison.
* **Kernel and normalisation must match before comparing NLL across papers.** A
  different kernel, or a density normalised differently, shifts every number by a
  constant that has nothing to do with model quality.
* **The anomaly scores are not calibrated probabilities.** A low density means
  "unlike the training sample", not "anomalous" and certainly not "malicious".
* **Round-off in the normaliser.** Truncating the kernel sum to `m` neighbours
  makes the estimate slightly biased upward; measured relative distortion
  1.7e-07 at `m = 384, N = 400`, and it is always reported alongside `m`.

---

## Ethical risks

### 1. Density estimation enables membership inference — the primary risk

**A density model memorises its training points, and that is a privacy liability.**

An adversary holding a candidate record `x` can compare its score under a model
trained *with* `x` against one trained *without* it. A model that memorises
assigns `x` a higher score in the first case, and the difference is membership
evidence. This is not a hypothetical: it is the standard attack against
generative models of tabular data, and the mitigations are known and imperfect.

**DensForge is a particularly good target, and that is worth stating plainly.**
Its per-point bandwidths mean a rare point receives a *distinctive* kernel —
precisely the signal the attack looks for. The `hetero_density` dataset in the
registry exists partly to make this failure mode reproducible.

Mitigations, and what each costs:

| mitigation | privacy gain | cost |
|---|---|---|
| add noise to inputs during fitting (DP-SGD style) | formal `(ε, δ)`-DP | accuracy loss on the tails, which is where the density is most informative |
| cap `m_score` and increase the bandwidth | lower peak memorisation | worse fit everywhere; the bias is not concentrated where the attack is |
| report only the leave-one-out density | removes the self-contribution term | still a membership signal; it raises the bar, does not close the door |
| do not expose `score_samples` for per-record queries | removes the oracle | usually not an option for a library |

**Mitigating factors.** The evaluation data is entirely synthetic, so running the
benchmark transfers no risk. The risk arises the moment a fitted model is applied
to real records — which is the intended use, and therefore the risk belongs here.

### 2. Outlier detection is a double-edged tool

The same machinery flags rare points as low-density. Deployed as an audit or
fraud filter, that is a mechanism for automated exclusion of legitimate minorities:
a group whose data is systematically under-represented *in the training sample*
scores low regardless of behaviour. If the training sample is not representative,
the model encodes the sampling process, not the world.

### 3. Benchmark results can be over-read

A favourable number here is evidence about synthetic generators of a known shape.
Reporting it as a general capability claim would misrepresent it. This is why the
table above is presented with its negative half intact.

### 4. Compute and energy

The `O(N³)` eigendecomposition is genuinely expensive at scale. Repeated
hyper-parameter search multiplies it. The default configuration is bounded and
single-threaded, but a user sweeping a large grid on a shared machine should
account for it.

---

## Maintenance

| concern | status |
|---|---|
| scikit-learn 1.9 API drift | pinned in `requirements.lock.txt`; signature contract asserted in `tests/test_sklearn_contract.py` |
| NumPy 2 API drift | asserted in `tests/test_numpy2_api.py`, including the removals that are *absent* |
| RandomState stream drift | exact golden values asserted in `tests/test_determinism.py` |
| threshold drift | **no threshold has been lowered.** Every version change is recorded in `CHANGELOG.md` with its reason |

The test suite is the specification. If a change makes an invariant fail, the
question is whether the invariant or the code is wrong — and the answer goes in
`docs/math_verification.md` either way. Three invariants were re-scoped during
development because their stated form was unassertable; the reasoning and the
measurements that exposed them are in that file, not in a commit message.

## Citation

Zelnik–Manor & Perona (2004) *Self-Tuning Laplacian Density Estimation*;
Lazaridis (2006) *Mean Shift Density Estimation for Space Denoising*;
Coifman & Lafon (2006) *Diffusion maps*; Szlam et al. (2004) on biorthogonal
coordinates; He & Niyogi (2009) on density balancing; Niyogi & Smolensky (2008) on
local scaling.
