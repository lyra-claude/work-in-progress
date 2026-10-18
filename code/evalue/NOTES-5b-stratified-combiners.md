# NOTES: Leg-3 §5b — Difficulty-Stratified E-Process, Combiner Comparison

> Exploratory simulation notes. No publication claim. All results are Monte Carlo estimates (R=500, SEED=42).

---

## Construction

Items are partitioned into K difficulty strata by equal-width binning of latent difficulty theta ~ U(0,1).

Within each stratum k, a Method-A cross-item e-process is built:
- U_s = W_i^s * W_j^s (same-item co-failure)
- V_s = W_i^s * W_j^{t(s)} where t(s) is a derangement within the stratum
- delta_k = 2 * epsilon_k, where epsilon_k = (max - min)/2 of marginal probabilities within stratum k
- Betting fraction lambda is clipped to [0, 1/(1+2*epsilon_k)] per item (ensures non-negative increments)
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
