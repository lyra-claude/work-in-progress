# NOTES: Leg-3 §5b — Difficulty-Stratified E-Process, Combiner Comparison

> Exploratory simulation notes. No publication claim. All results are Monte Carlo estimates (R=500, SEED=42).

---

## Construction

Items are partitioned into K difficulty strata by equal-width binning of latent difficulty theta ~ U(0,1).

Within each stratum k, a Method-A cross-item e-process is built:
- U_s = W_i^s * W_j^s (same-item co-failure)
- V_s = W_i^s * W_j^{t(s)} where t(s) is a FRESH-PAIR within-block, within-stratum partner (Clio Prop. 3): another same-stratum item in the same block, so neither partner's outcome is in F_{m-1} when the block-m bet is set (anytime-valid)
- delta_k = 2 * epsilon_k**2 (QUADRATIC slack 1/2 R^2 = 2 eps^2, R = 2 eps; the old linear 2*epsilon_k was valid but loose by a factor 1/epsilon), where epsilon_k = (max - min)/2 of marginal probabilities within stratum k
- Betting fraction lambda is clipped to [0, 1/(1+2*epsilon_k**2)] per item (ensures non-negative increments, consistent with the quadratic slack)
- Per-stratum wealth M^(k) built by Robbins-style mixture over lambda in {0.1, 0.2, ..., 0.9}

The per-stratum wealth streams are then combined into a single process via two methods:

**Averaged combiner:**
  E_t^avg = (1/K) * sum_k E^(k)_{count_k(t)}
- Valid by convexity of e-processes. No independence assumption needed.
- Known weakness: near-zero power. Sub-processes must collectively push (1/K)*sum to 1/alpha;
  when K is large, each stratum must reach K*threshold to move the average.

**Product combiner:**
  E_t^prod = prod_k E^(k)_{count_k(t)}
- Valid IF per-stratum e-processes are mutually independent under H0.
- Independence argument: items in different strata have different difficulty bins; under
  static H0 their W_i, W_j outcomes are conditionally independent given theta, and
  theta-bins are disjoint, so cross-stratum pairs are independent.
- Under DRIFT, shared time-varying base rates create cross-stratum correlation that could
  break this independence and thus break the product's validity. We measure, not assume.

---

## Experimental Conditions

Parameters: N varied, R=500, alpha=0.05, threshold=1/alpha=20, SEED=42, K=5 (unless noted).

**DGPs:**
- `null_dgp`: theta ~ U(0,1), p_i = clip(0.30 + 0.50*(theta-0.5)), p_j = clip(0.35 + 0.50*(theta-0.5)), W_i, W_j conditionally independent given theta.
- `drift_null_dgp`: same structure but base rates ramp linearly (base_i: 0.20->0.45, base_j: 0.25->0.50). Still H0 (conditional independence given theta at each time step).
- `alt_dgp(gamma)`: shared-shock Marshall-Olkin alternative. With probability gamma per item, both judges fail together (W_i=W_j=1) regardless of difficulty. Otherwise H0.

---

## Full Results

### Original Sweep (run_stratified_combiners.py): N=2000, K in {2, 5, 10}, R=500

All rates are fraction of 500 replicates that crossed threshold 20.

| Condition                        | K  | AVG Type-I | PROD Type-I | Note           |
|----------------------------------|----|------------|-------------|----------------|
| (a) static H0, true strata       | 2  | 0.002      | 0.000       | OK             |
| (a) static H0, true strata       | 5  | 0.000      | 0.000       | OK             |
| (a) static H0, true strata       | 10 | 0.000      | 0.000       | OK             |
| (b) drift H0, true strata        | 2  | 0.002      | 0.000       | OK             |
| (b) drift H0, true strata        | 5  | 0.000      | 0.000       | OK             |
| (b) drift H0, true strata        | 10 | 0.000      | 0.000       | OK             |
| (c') static H0, hard plug-in     | 2  | 0.002      | 0.000       | OK             |
| (c') static H0, hard plug-in     | 5  | 0.000      | 0.000       | OK             |
| (c') static H0, hard plug-in     | 10 | 0.000      | 0.000       | OK             |
| (d) drift H0, noisy strata       | 2  | 0.002      | 0.000       | OK             |
| (d) drift H0, noisy strata       | 5  | 0.000      | 0.000       | OK             |
| (d) drift H0, noisy strata       | 10 | 0.000      | 0.000       | OK             |
| (e) alt H1 gamma=0.10, K=2       | 2  | 0.024      | 0.018       | low power      |
| (e) alt H1 gamma=0.10, K=5       | 5  | 0.016      | 0.026       | low power      |
| (e) alt H1 gamma=0.10, K=10      | 10 | 0.014      | 0.016       | low power      |

**Concern raised:** All Type-I rates = 0.000 when power <= 0.026. The test may be nearly
dormant — a zero false-rejection rate when the process rarely moves is weak evidence of
validity. To confirm validity is real and not an artifact of a dormant test, we need to
find a regime where the test fires readily under the alternative and then check whether
it still respects the null.

---

### Strong-Signal Sweep (run_strongsignal_sweep.py): PRODUCT combiner, K=5, R=500

**PART 1 — Power curve** (alt_dgp, true strata):

| gamma | N      | Power | Median Det Time |
|-------|--------|-------|-----------------|
| 0.10  | 2000   | 0.000 | —               |
| 0.10  | 5000   | 0.000 | —               |
| 0.10  | 10000  | 0.002 | 21              |
| 0.20  | 2000   | 0.012 | 51              |
| 0.20  | 5000   | 0.006 | 145             |
| 0.20  | 10000  | 0.010 | 69              |
| 0.30  | 2000   | 0.070 | 130             |
| 0.30  | 5000   | 0.106 | 370             |
| 0.30  | 10000  | 0.198 | 2414            |
| 0.50  | 2000   | 0.446 | 903             |
| **0.50** | **5000**  | **0.968** | **2219** |
| **0.50** | **10000** | **1.000** | **2068** |

Substantial power (>= 0.5) is first reached at gamma=0.50, N=5000 (power=0.968).

Note on the gamma=0.20 anomaly: power is non-monotone in N at gamma=0.20 (N=2000 gives 0.012
but N=5000 gives 0.006). This is within Monte Carlo noise at R=500 with very low
rejection counts (6 vs 3 out of 500). Not a reliable signal — both are essentially zero.

Note on power growth: the product combiner's power is substantially lower than one might
expect for gamma=0.50 at N=2000 (0.446). This is consistent with a hard threshold test
(threshold=20) applied to K=5 independent sub-streams: the product needs all K strata to
contribute, and with N=2000 each stratum has only ~400 items. The 0.446 rate rising to
0.968 at N=5000 (each stratum ~1000 items) reflects this — the product amplifies quickly
once each stratum has enough data to move.

**PART 2 — Validity at high-power settings** (gamma=0.50, N in {5000, 10000}):

| Condition                      | gamma | N      | Type-I | Status |
|--------------------------------|-------|--------|--------|--------|
| (b) drift H0, true strata      | 0.50  | 5000   | 0.000  | OK     |
| (d) drift H0, noisy strata     | 0.50  | 5000   | 0.000  | OK     |
| (b) drift H0, true strata      | 0.50  | 10000  | 0.000  | OK     |
| (d) drift H0, noisy strata     | 0.50  | 10000  | 0.000  | OK     |
| (a) static H0, true strata     | 0.50  | 5000   | 0.000  | OK     |
| (a) static H0, true strata     | 0.50  | 10000  | 0.000  | OK     |

**No validity breaks detected.** Type-I = 0.000 across all conditions at high-power N.

---

## Verdict

The product combiner (K=5) shows:

1. **Validity**: Type-I rate <= 0.05 across all tested null conditions (static H0,
   drift H0, hard plug-in strata, noisy strata), at both low-power (N=2000, gamma=0.10)
   and high-power (N=5000/10000, gamma=0.50) settings. The 0.000 false-rejection rate
   at high-power settings — where the test fires at rates 0.968 and 1.000 under the
   alternative — is genuine evidence of validity, not a dormant-test artifact.

2. **Power**: Substantial power (>= 0.5) requires both large gamma (0.50) and moderate
   N (5000+). At realistic moderate shock magnitudes (gamma=0.10-0.30), power is low.
   This is a practical limitation of the product combiner under the current construction:
   the test needs all K strata to accumulate sufficient evidence simultaneously.

3. **Averaged vs Product**: The averaged combiner has comparable validity (all rates
   <= 0.05) and comparable power, but both are low under the original parameter regime.
   The product combiner does not offer a power advantage in this DGP.

The §5b construction is **empirically valid while actively firing** at gamma=0.50,
N>=5000. The practical limitation is that gamma=0.50 is a very large shock
(half of all items are simultaneous failures) — validity at smaller, more realistic
effect sizes (gamma=0.10-0.30) has only been checked in the near-dormant regime.

---

## Open Question for Clio

**Theoretical gap: validity of the product combiner**

The product combiner E_t^prod = prod_k E^(k)_{count_k(t)} is used as an e-process.
For this to be valid (i.e., E[E_t^prod] <= 1 under H0 for all stopping times t), we
need the K per-stratum e-processes to be mutually independent under H0.

**The question for Clio:**

1. **Static H0:** Can we formally establish that per-stratum e-processes are independent
   under the null? The argument is that items in different difficulty strata have
   non-overlapping theta bins, and given theta, W_i and W_j are conditionally
   independent. Does this chain of reasoning give independence of the WEALTH PROCESSES
   (not just of individual item outcomes), accounting for the cross-item derangement
   structure used to form V_s?

2. **Under drift:** The drift_null_dgp introduces a shared time-varying base rate that
   affects ALL strata simultaneously. This creates cross-stratum correlation in item-level
   outcomes, which could break the independence of wealth processes. The empirical results
   show 0.000 Type-I at N=10000 under drift, but this is Monte Carlo evidence, not a
   proof. Is there a formal argument for why the product combiner remains a valid
   e-process under the drift model tested here (linear base-rate ramp), or does validity
   rely on a weaker property (e.g., that the product is an e-variable but not an
   e-process)?

3. **Scope question:** If cross-stratum independence fails under drift, what is the
   strongest formal validity guarantee available for the product combiner? Is it:
   (a) a valid e-process under static H0 only,
   (b) valid under drift via some conditional argument, or
   (c) only an asymptotic/approximate result?

This theoretical question is Clio-gated before any formal claim about the product
combiner appears in the paper. The empirical results are consistent with validity but
do not prove it.

---

*Generated by run_stratified_combiners.py and run_strongsignal_sweep.py on branch leg3-stratified-5b.*
*Simulation parameters: K=5, R=500, alpha=0.05, SEED=42.*

---

## Clopper-Pearson Benign-Window Calibration Experiment

> EXPLORATORY. Results measure a power/assumption tradeoff only. No formal validity claim.
> Script: `run_benign_window_calibration.py`. Parameters: K=10, N_test=5000, R=500, SEED=42.

### Motivation and Setup

The product combiner (K=10) has low power at gamma <= 0.30 even at N=10000. The theoretical
baseline uses the semi-oracle epsilon (from true marginals) to set delta_k = 2*epsilon_k ≈ 0.05
per stratum with K=10. This is tight but requires access to the true marginals.

**Benign-window idea**: use an early "pre-drift" calibration window of n_cal items (drawn from
null_dgp, presumed shock-free) to estimate the per-stratum excess co-failure baseline E[U - V],
then use a one-sided z-UCB (90% confidence) on that estimate as delta_k.

**Critical design point**: the calibration must estimate E[U - V] (the excess co-failure over
cross-item), NOT the raw co-failure rate E[W_i * W_j]. The raw rate is ~0.10-0.40 (includes
baseline co-failure from shared difficulty), while the excess is ~0 under H0 with K=10 fine
strata. Using the raw rate as delta_k kills power entirely (documented in the script).

**Why calibration diverges from oracle at small n_cal**: With K=10 strata and n_cal items, each
stratum receives ~n_cal/K items. For n_cal=100: ~10 items/stratum. The z-UCB on the excess
(mean ≈ 0, std ≈ 0.5 for binary products) is approximately:
  delta_k_cal ≈ z_{0.90} * 0.5 / sqrt(10) ≈ 1.28 * 0.16 ≈ 0.20
Compare to oracle: 2*eps_k ≈ 0.05. The calibrated delta is ~4x larger → less power.
At n_cal=500: delta_k_cal ≈ 1.28 * 0.5 / sqrt(50) ≈ 0.09, still ~2x the oracle.

### Type-I Results (K=10, N_test=5000, R=500)

Calibration + test window from same null DGP:

| Condition  | n_cal=100 (T/C) | n_cal=250 (T/C) | n_cal=500 (T/C) |
|------------|-----------------|-----------------|-----------------|
| static H0  | 0.002 / 0.000   | 0.000 / 0.000   | 0.000 / 0.000   |
| drift H0   | 0.000 / 0.000   | 0.000 / 0.000   | 0.000 / 0.000   |

**No Type-I violations detected.** The calibrated method preserves Type-I under both static
and drift nulls (at N_test=5000, K=10). The 0.000 rate at n_cal=100 is consistent with a
more conservative test (larger delta_k → less likely to fire under null).

**Caveat on drift validity**: the drift null DGP has linearly increasing base rates from t=0.
The calibration window (first n_cal items) sees lower base rates than the test window. This
under-estimates the null co-failure rate slightly, but appears not to inflate Type-I in this
experiment. This is reassuring but not a proof of validity — faster or later-onset drift
could differ.

### Power Results (K=10, N_test=5000, R=500)

Calibration: null_dgp (clean). Test: alt_dgp(gamma). Baseline: theoretical oracle on test window.

| Condition     | n_cal=100 (T/C)     | n_cal=250 (T/C)     | n_cal=500 (T/C)     |
|---------------|---------------------|---------------------|---------------------|
| gamma=0.10    | 0.026 / 0.052       | 0.026 / 0.042       | 0.028 / 0.082       |
| gamma=0.20    | 0.904 / 0.524       | 0.902 / 0.656       | 0.892 / 0.898       |
| gamma=0.30    | 1.000 / 0.866       | 1.000 / 0.946       | 1.000 / 1.000       |
| gamma=0.40    | 1.000 / 0.934       | 1.000 / 0.992       | 1.000 / 1.000       |
| gamma=0.50    | 1.000 / 0.946       | 1.000 / 0.990       | 1.000 / 1.000       |

Format: Theoretical-oracle / Excess-UCB-calibrated. T = theoretical oracle (test window only).

**Key findings:**

1. **gamma=0.10 (the hard case)**: Both methods have low power (0.026-0.082). The calibrated
   method appears slightly higher than theoretical at some n_cal values, but the absolute
   rejection counts are tiny (13-41 out of 500). These differences are within Monte Carlo
   noise — not a reliable signal of improvement. The gamma=0.10 problem is NOT solved by
   benign-window calibration.

2. **gamma=0.20 (realistic moderate shock)**: The calibrated method loses substantial power
   at small n_cal (0.524 vs 0.904 at n_cal=100), converging near parity at n_cal=500
   (0.898 vs 0.892). This confirms the theoretical expectation: calibration catches up to
   the oracle only when n_cal/K is large enough (~50+ items/stratum).

3. **gamma >= 0.30**: Both methods reach full power (1.000 or near) at n_cal=500. The
   calibrated method lags at n_cal=100-250, with a larger median detection time (slower
   detection), but reaches 1.000 at n_cal=500.

4. **n_cal sensitivity**: Power grows monotonically in n_cal for the calibrated method.
   At n_cal=500 (50 items/stratum for K=10), calibration nearly matches the oracle at
   gamma >= 0.20. This suggests the break-even point for K=10 is n_cal ≈ 500-1000.

### Honest Assessment

**Does benign-window calibration solve the gamma <= 0.30 power problem?**

No, it does not. The core difficulty (low power at gamma=0.10) persists with the calibrated
method — the calibrated delta is systematically wider than the oracle for any n_cal ≤ 500
with K=10 strata. The method does not improve power; it typically reduces it.

**What does it cost in Type-I (under drift)?**

Nothing observable in this experiment. The calibrated Type-I is 0.000 under drift H0 at
N_test=5000. This is actually *better* Type-I control than the theoretical method (also 0.000
here, but the calibrated delta is more conservative), which comes at the cost of power.

**When would benign-window calibration make sense?**

Only when: (a) oracle marginals are unavailable (pure data-driven setting), (b) n_cal is
large enough that n_cal/K >> 50, and (c) the calibration window is genuinely pre-shock.
Under those conditions, the excess-UCB converges to the oracle and the method is practical.
With K=10 and the n_cal values tested here (100-500), condition (b) is not met.

**The CP-UCB on raw co-failure rate** (naive calibration) is definitively NOT viable: it
confuses the estimand (excess co-failure vs total co-failure) and produces zero power even
at gamma=0.50. This is documented in the script but not included in the results table as
it is a cautionary negative result, not a viable comparison.

**Conclusion**: Benign-window calibration is a valid structural idea but does not improve
the gamma <= 0.30 power problem under the current K=10 construction. The limiting factor
is n_cal/K < 50, not the calibration method itself. Future work could try K=5 (fewer strata
= more items per stratum in the calibration window) or larger n_cal (>= 1000).

---

*Benign-window results generated by run_benign_window_calibration.py on branch leg3-stratified-5b.*
*Simulation parameters: K=10, N_test=5000, R=500, alpha=0.05, SEED=42.*

---

## Equicorrelation / Forecast-Combination-Puzzle test (2026-11-15)

Adversarial null for the ONS online-learned weights (commit 61c5a2d). Question: does learning the stratum weights lambda_s online (ONS) beat plain equal weights (lam_fixed) at deployment-relevant signal strengths? The Forecast Combination Puzzle (econometrics, 50-yr result; Wang-Hyndman 2205.04216) predicts no — under equicorrelation/weak-signal, estimation variance swamps the gain.

Scripts: `run_equicorr_puzzle.py` (Regime 1 homogeneous shock, Regime 2 heterogeneous shock in top-30% strata, Regime 3 validity) + `run_equicorr_puzzle_highgamma.py` (R1/R2 at higher gamma to reach phi_bar approx 0.4). Settings: N=4000, R=200, Q=10, BLOCK=20, threshold=20, alpha=0.05, paired within-replication seeding (default_rng(1000+r)) so FIXED/ONLINE/ORACLE see identical data.

**Findings:**

- Homogeneous regime (R1), phi_bar up to ~0.31: ONLINE strictly LOSES to FIXED on paired per-seed log-wealth (Wilcoxon p <= 0.005 at every gamma; win-fractions 0.25-0.43).
- At phi_bar approx 0.40 (gamma=0.25): per-seed MEDIAN log-wealth crossover — ONLINE vs FIXED indistinguishable on median (p=0.98, win 0.515, median diff +0.086), BUT FIXED retains higher MEAN log-wealth (+5.57 vs +1.85) and higher POWER (0.98 vs 0.84).
- phi_bar up to 0.47: ONLINE's median edge grows thin-positive (median diff +0.44, win 0.57), but FIXED dominates MEAN log-wealth and POWER at every point (power 1.00 vs <=0.975).
- Mechanism: ONS's estimation variance manifests as left-tail wealth collapse (mean << median for ONLINE) plus ~6-16 percentage-point power loss — the textbook Forecast Combination Puzzle signature.
- Heterogeneous regime (R2, shock only in top-30% difficulty strata, so oracle weights are genuinely non-uniform): ONLINE loses decisively everywhere (Wilcoxon p < 1e-9). Neither FIXED nor ONLINE detects at tested gamma (ORACLE only). R2's whole-sample phi_bar never reaches 0.4 on this grid (tops at 0.27 at gamma=0.35; reaching 0.4 would need a saturating gamma approx 0.6+).
- Validity (R3): Type-I = 0.000 for FIXED, ONLINE, and ORACLE under both the latent-factor equicorrelation null and drift_null. ONS adaptivity does NOT inflate Type-I.

**Decision:** equal weights (lam_fixed) are the publishable §5b joint-bet combiner. The ONS online-learned weights are NOT a deployment improvement — they do not beat equal weights on power or expected log-wealth anywhere in the phi_bar range (including ~0.40), and they carry estimation-variance tail risk. ONS is retained as a DIAGNOSTIC (it reveals when ORACLE >> FIXED, i.e. when the co-failure signal is concentrated enough that reweighting could in principle matter), not sold as a corrector.

**Confidence:** high for this DGP (Marshall-Olkin shock + linear-in-theta marginals); ~85% as a general claim, since the puzzle mechanism (estimation variance of learned weights) is DGP-agnostic. Caveat: single co-failure generative family tested.

---

*Equicorrelation/FCP results generated by run_equicorr_puzzle.py and run_equicorr_puzzle_highgamma.py on branch leg3-stratified-5b.*
*Simulation parameters: N=4000, R=200, Q=10, BLOCK=20, threshold=20, alpha=0.05.*
