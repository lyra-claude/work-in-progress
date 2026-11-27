# Rolling / Causal Ranks -- Shared-Time-Trend Confound Fix

**Setup:** N=4000 items, K=3 judges, seed=20261123, trailing windows [50, 100, 200].
Same seeding convention as `pilot.py` (`np.random.default_rng(20261123)`, per-scenario
spawned sub-streams).

## The confound being fixed

The global (whole-stream) rank-PIT in `pilot.py` absorbs *independent* per-judge
marginal drift cleanly. But when **all judges share a common slow time-trend**
(scenario D: one shared mean trajectory mu(t) with std ~0.5 marginal-sd units,
plus independent per-judge noise, so dependence given time is ZERO), the global rank ties each
judge's rank to the time index. This produces a **spurious** cross-judge rank
correlation -- drift masquerading as co-failure.

## Results

| Scenario | Rank method | KS-to-Unif | pairwise \|corr\| | n_usable |
|----------|-------------|-----------:|------------------:|---------:|
| A dependence | global | 0.0002 | 0.3956 | 4000 |
| A dependence | rolling-50 | 0.0166 | 0.3931 | 3980 |
| A dependence | rolling-100 | 0.0117 | 0.3964 | 3980 |
| A dependence | rolling-200 | 0.0097 | 0.3996 | 3960 |
| B drift-null | global | 0.0002 | 0.1120 | 4000 |
| B drift-null | rolling-50 | 0.0195 | 0.0129 | 3980 |
| B drift-null | rolling-100 | 0.0172 | 0.0144 | 3980 |
| B drift-null | rolling-200 | 0.0218 | 0.0175 | 3960 |
| D shared-trend | global | 0.0002 | 0.2133 | 4000 |
| D shared-trend | rolling-50 | 0.0178 | 0.0174 | 3980 |
| D shared-trend | rolling-100 | 0.0140 | 0.0177 | 3980 |
| D shared-trend | rolling-200 | 0.0165 | 0.0207 | 3960 |

- **KS-to-Unif**: mean KS of per-judge PIT to Uniform[0,1] (lower = marginals absorbed).
- **\|corr\|**: mean absolute pairwise Pearson correlation of the PITs (the co-failure signal).
- **n_usable**: rows with all-K non-NaN PITs. Rolling ranks have a causal warm-up
  (first ~window/5 obs per judge lack enough past points), hence fewer usable rows.

## Success criterion

- (1) **Kill** the spurious \|corr\| in D shared-trend: global = **0.2133**, rolling should -> ~0 (<0.07).
- (2) **Preserve** genuine \|corr\| in A dependence: global = **0.3956**, rolling should stay elevated (>0.20).

| Window | D shared-trend \|corr\| | spurious killed? | A dependence \|corr\| | genuine preserved? | Verdict |
|-------:|------------------------:|:----------------:|----------------------:|:------------------:|:-------:|
| 50 | 0.0174 | YES | 0.3931 | YES | **MET** |
| 100 | 0.0177 | YES | 0.3964 | YES | **MET** |
| 200 | 0.0207 | YES | 0.3996 | YES | **MET** |


**Overall: criterion MET for at least one window.**

## Trend-strength robustness sweep (shared-trend scenario only)

The spurious global |corr| grows monotonically with the shared-trend strength;
the rolling fix suppresses it across the whole regime (trend_std=0.5 brackets
the prior run's ~0.2-0.25 spurious correlation).

| trend_std (sd units) | GLOBAL \|corr\| | roll-50 | roll-100 | roll-200 |
|---------------------:|---------------:|--------:|---------:|---------:|
| 0.30 | 0.0840 | 0.0183 | 0.0235 | 0.0341 |
| 0.50 | 0.2060 | 0.0245 | 0.0408 | 0.0731 |
| 1.00 | 0.5239 | 0.0195 | 0.0265 | 0.0521 |
| 2.00 | 0.8149 | 0.0344 | 0.0692 | 0.1071 |


At trend_std=2.0 (a very strong trend) the global |corr| reaches ~0.81 and the
widest window (200) leaks a little (~0.11) because the trend is no longer
approximately constant across 200 points; shorter windows still kill it.

## Interpretation and trade-offs

- **Global ranks confirm the confound.** D shared-trend GLOBAL \|corr\| = 0.2133
  (vs drift-null B global \|corr\| = 0.1120 and genuine A = 0.3956).
  The shared trend alone fabricates a co-failure-sized correlation under global ranks.
- **Rolling/causal ranks remove it.** By ranking only against a short trailing window,
  the slow shared trend is ~constant within the window and is differenced away, driving
  D's \|corr\| toward the drift-null level.
- **Genuine dependence survives** because it is present *within every window*, not an
  artefact of the time ordering.
- **Trade-off (power vs window size):** smaller windows kill the trend more aggressively
  but add sampling noise to each local rank (coarser PIT, inflated KS, a small floor on
  \|corr\| from finite-window noise) and discard a warm-up region. Larger windows retain
  more of the shared trend and so leak a little more spurious correlation. The window is
  the bias/variance knob between trend-robustness and local-rank precision.

---
*Empirical motivation only. Causal ranking is online-computable (each obs ranked only
against strictly past obs) but this is NOT yet an anytime-valid e-process.*

---

# APPENDED: Tail Co-Failure Diagnostic

**Setup:** N=4000 items, K=3 judges, rolling window=100, 10 master seeds (base seed=20261123, seeds [20261123, 20261124, 20261125, 20261126, 20261127, 20261128, 20261129, 20261130, 20261131, 20261132]).
Same scenarios/seeding/transforms as `rolling_ranks.py` (imported directly).

## What this adds

The pairwise `|corr|` statistic measures *average* co-movement and is (near-)blind to **tail** co-failure. This diagnostic targets the tail directly: the **joint upper-tail exceedance rate** = fraction of items where **all K judges' PITs exceed 1-alpha simultaneously**. Under independent uniform margins this equals `alpha^K` exactly, so we report

- **excess** = observed_rate - `alpha^K`  (0 under independence; >0 = tail co-failure)
- **lift**   = observed_rate / `alpha^K`  (1 under independence)

Lower tail is the mirror (`all judges < alpha`), same null. **Batch / empirical diagnostic only -- NOT an anytime-valid e-process.**

## Upper-tail results (mean over 10 seeds)

| Scenario | Rank method | alpha | null alpha^K | joint-tail rate | excess (mean+/-sd) | lift | ~n_events | n_usable |
|----------|-------------|------:|------------:|---------------:|------------------:|-----:|----------:|---------:|
| A (genuine dependence) | global | 0.10 | 0.00100 | 0.01378 | +0.01277+/-0.00188 | 13.77 | 55.1 | 4000 |
| A (genuine dependence) | global | 0.05 | 0.00013 | 0.00480 | +0.00468+/-0.00101 | 38.40 | 19.2 | 4000 |
| A (genuine dependence) | rolling | 0.10 | 0.00100 | 0.01384 | +0.01284+/-0.00128 | 13.84 | 55.1 | 3980 |
| A (genuine dependence) | rolling | 0.05 | 0.00013 | 0.00477 | +0.00465+/-0.00079 | 38.19 | 19.0 | 3980 |
| B (drift-only null (independent)) | global | 0.10 | 0.00100 | 0.00130 | +0.00030+/-0.00115 | 1.30 | 5.2 | 4000 |
| B (drift-only null (independent)) | global | 0.05 | 0.00013 | 0.00027 | +0.00015+/-0.00032 | 2.20 | 1.1 | 4000 |
| B (drift-only null (independent)) | rolling | 0.10 | 0.00100 | 0.00103 | +0.00003+/-0.00055 | 1.03 | 4.1 | 3980 |
| B (drift-only null (independent)) | rolling | 0.05 | 0.00013 | 0.00015 | +0.00003+/-0.00018 | 1.21 | 0.6 | 3980 |
| D (shared trend (no genuine co-failure)) | global | 0.10 | 0.00100 | 0.00358 | +0.00258+/-0.00136 | 3.57 | 14.3 | 4000 |
| D (shared trend (no genuine co-failure)) | global | 0.05 | 0.00013 | 0.00082 | +0.00070+/-0.00046 | 6.60 | 3.3 | 4000 |
| D (shared trend (no genuine co-failure)) | rolling | 0.10 | 0.00100 | 0.00141 | +0.00041+/-0.00079 | 1.41 | 5.6 | 3980 |
| D (shared trend (no genuine co-failure)) | rolling | 0.05 | 0.00013 | 0.00033 | +0.00020+/-0.00031 | 2.61 | 1.3 | 3980 |

## Lower-tail results (mean over 10 seeds)

| Scenario | Rank method | alpha | null alpha^K | joint-tail rate | excess (mean+/-sd) | lift | ~n_events | n_usable |
|----------|-------------|------:|------------:|---------------:|------------------:|-----:|----------:|---------:|
| A (genuine dependence) | global | 0.10 | 0.00100 | 0.01393 | +0.01293+/-0.00128 | 13.92 | 55.7 | 4000 |
| A (genuine dependence) | global | 0.05 | 0.00013 | 0.00495 | +0.00482+/-0.00112 | 39.60 | 19.8 | 4000 |
| A (genuine dependence) | rolling | 0.10 | 0.00100 | 0.01339 | +0.01239+/-0.00142 | 13.39 | 53.3 | 3980 |
| A (genuine dependence) | rolling | 0.05 | 0.00013 | 0.00485 | +0.00472+/-0.00104 | 38.79 | 19.3 | 3980 |
| B (drift-only null (independent)) | global | 0.10 | 0.00100 | 0.00135 | +0.00035+/-0.00192 | 1.35 | 5.4 | 4000 |
| B (drift-only null (independent)) | global | 0.05 | 0.00013 | 0.00015 | +0.00002+/-0.00021 | 1.20 | 0.6 | 4000 |
| B (drift-only null (independent)) | rolling | 0.10 | 0.00100 | 0.00078 | -0.00022+/-0.00065 | 0.78 | 3.1 | 3980 |
| B (drift-only null (independent)) | rolling | 0.05 | 0.00013 | 0.00003 | -0.00010+/-0.00008 | 0.20 | 0.1 | 3980 |
| D (shared trend (no genuine co-failure)) | global | 0.10 | 0.00100 | 0.00330 | +0.00230+/-0.00145 | 3.30 | 13.2 | 4000 |
| D (shared trend (no genuine co-failure)) | global | 0.05 | 0.00013 | 0.00053 | +0.00040+/-0.00045 | 4.20 | 2.1 | 4000 |
| D (shared trend (no genuine co-failure)) | rolling | 0.10 | 0.00100 | 0.00141 | +0.00041+/-0.00083 | 1.41 | 5.6 | 3980 |
| D (shared trend (no genuine co-failure)) | rolling | 0.05 | 0.00013 | 0.00018 | +0.00005+/-0.00017 | 1.41 | 0.7 | 3980 |

## Success criterion (upper tail)

| alpha | A global | A rolling | B rolling | D global | D rolling | verdict |
|------:|---------:|----------:|----------:|---------:|----------:|:-------:|
| 0.10 | +0.01277 (pos) | +0.01284 (pos) | +0.00003 (~0) | +0.00258 (SPURIOUS+ (t=3.2, 9/10)) | +0.00041 (fixed~0) | **MET** |
| 0.05 | +0.00468 (pos) | +0.00465 (pos) | +0.00003 (~0) | +0.00070 (SPURIOUS+ (t=2.5, 8/10)) | +0.00020 (fixed~0) | **MET** |

Legend: pos=significant positive (excess>2sd and >null); ns=not significant; ~0=consistent with null; SPURIOUS+=D-global excess exceeds the B-global baseline by a PAIRED same-seed contrast (t>2; shared trend fakes tail co-failure); fixed~0=causal ranks return D's excess to the null band.

## Caveats

- **Tail counts are small.** At alpha=0.05, K=3, the null is alpha^K=0.000125, i.e. expected ~0.50 joint events in N=4000 items under independence. Even under dependence only a handful of items land in the joint tail, so alpha=0.05 is noisy (wide sd, limited power). alpha=0.10 (null 0.0010, ~4.0 expected events) is the more reliable operating point here.
- **Rolling ranks lose rows** to the causal warm-up, shrinking the usable sample and further thinning the tail.
- Multi-seed mean+/-sd (ddof=1) quantifies the finite-sample noise; it does not confer any sequential/anytime validity.

*Empirical tail-coincidence diagnostic only. No anytime-valid claim.*
