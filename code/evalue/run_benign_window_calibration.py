"""
run_benign_window_calibration.py — Clopper-Pearson benign-window calibration experiment.

PURPOSE:
  The product combiner (K=10) has low power at realistic gamma values (<= 0.30).
  Idea (motivated by 2610.11843): estimate the per-stratum excess co-failure baseline
  from an early "benign window" of n_cal items, then use a UCB on that estimate as
  delta_k in the betting statistic instead of the oracle theoretical epsilon.

BACKGROUND — WHY CP-UCB ON CO-FAILURE RATE FAILS (exploratory finding):
  A naive approach would estimate delta_k as the CP upper confidence bound on the raw
  co-failure rate P(W_i = W_j = 1) from the calibration window. This FAILS because:
  - The betting statistic measures EXCESS co-failure: E[U - V] = E[W_i*W_j] - E[W_i]*E[W_j]
  - The theoretical epsilon (2*eps_k ~ 0.05 for K=10 fine strata) is the maximum EXCESS,
    not the total co-failure rate (~ 0.10-0.40 for difficult items).
  - Setting delta_k ~ 0.10-0.40 (the raw co-failure UCB) rather than ~ 0.05 (the excess UCB)
    makes the betting increment negative in expectation even under H1, killing power entirely.

CORRECT APPROACH — Excess-UCB:
  Estimate E[U - V] per stratum from the calibration window using a derangement within
  that window to form V. Use a one-sided z-UCB on the mean excess as delta_k:
    delta_k = max(0, mean(U_cal - V_cal) + z_{1-alpha} * std(U_cal - V_cal) / sqrt(n_k_cal))
  Under H0 the mean excess ~ 0, so this UCB ≈ z * std/sqrt(n_k_cal).
  With K=10 strata and n_cal items, each stratum gets ~n_cal/K items.
  For n_cal=100, K=10: ~10 items/stratum, UCB ≈ 1.28*0.5/sqrt(10) ≈ 0.20.
  Theoretical oracle: 2*eps_k ≈ 0.05. So the excess UCB is ~4x larger → less power.
  For n_cal=1000, K=10: ~100 items/stratum, UCB ≈ 1.28*0.5/sqrt(100) ≈ 0.064 ≈ oracle.

MAIN HYPOTHESIS: calibration helps only when n_cal/K is large. With K=10 and
  n_cal ≤ 500, the excess UCB will be systematically wider than the oracle, giving
  LOWER power than the theoretical combiner. We test whether this holds empirically.

DESIGN:
  Methods compared (all use product combiner, K=10):
    A) Theoretical: oracle epsilon from true marginals (semi-oracle, test window only)
    B) Excess-UCB: UCB on mean(U-V) from n_cal clean calibration items

  Type-I: both calibration AND test window from null DGP (static + drift)
  Power: calibration from null_dgp (clean), test from alt_dgp(gamma)
    — this models the intended use: shocks begin AFTER the calibration window

  n_cal: {100, 250, 500}
  K=10, N_test=5000, R=500, SEED=42
  gamma: {0.10, 0.20, 0.30, 0.40, 0.50}

OUTPUT:
  results_benign_window.csv
  Findings appended to NOTES-5b-stratified-combiners.md

CAVEATS (stated plainly):
  - Method B is NOT oracle-free; it requires a clean calibration window.
  - Excess-UCB uses normal-theory intervals; these are approximate for small n_k.
  - With K=10 and n_cal ≤ 500, the UCB width dominates the oracle epsilon, so no power
    gain is expected. The experiment MEASURES this; it does not assume an improvement.
  - All results are EXPLORATORY. No formal validity claim is made.
"""

import csv
import sys
import numpy as np
from scipy.stats import norm as scipy_norm

sys.path.insert(0, '/home/lyra/projects/evalue-sheaf/code/evalue')

from dgp import null_dgp, alt_dgp, drift_null_dgp
from eprocess import (
    assign_strata,
    stratum_epsilon,
    accumulate_wealth_mixture,
    ville_reject,
    THRESHOLD,
)

# ── Parameters ────────────────────────────────────────────────────────────────

K          = 10
R          = 500
ALPHA      = 0.05
SEED       = 42
N_TEST     = 5000

CAL_ALPHA  = 0.10   # one-sided confidence for UCBs

N_CAL_GRID = [100, 250, 500]
GAMMA_GRID = [0.10, 0.20, 0.30, 0.40, 0.50]

print(f"SEED = {SEED}")
print(f"Parameters: K={K}, N_test={N_TEST}, R={R}, alpha={ALPHA}, threshold={THRESHOLD}")
print(f"Calibration: one-sided z-UCB at (1 - {CAL_ALPHA}) = {1-CAL_ALPHA:.2f}")
print(f"n_cal grid: {N_CAL_GRID}")
print(f"Gamma grid: {GAMMA_GRID}")
print()


# ── Calibration: Excess-UCB ───────────────────────────────────────────────────

def calibrate_excess_ucb(
    W_i_cal: np.ndarray,
    W_j_cal: np.ndarray,
    strata_cal: np.ndarray,
    K_val: int,
    rng: np.random.Generator,
    alpha: float = CAL_ALPHA,
) -> np.ndarray:
    """
    Compute per-stratum delta_k as a UCB on the excess co-failure E[U - V].
    For each stratum: derange within the stratum, compute (U_s - V_s) per item,
    and use mean + z_{1-alpha} * std / sqrt(n_k) as the UCB.
    Clipped to [0, 1].
    """
    z       = scipy_norm.ppf(1.0 - alpha)
    delta_k = np.zeros(K_val)

    for k in range(K_val):
        mask = np.where(strata_cal == k)[0]
        n_k  = len(mask)
        if n_k < 2:
            delta_k[k] = 1.0   # maximally conservative
            continue
        W_i_k      = W_i_cal[mask]
        W_j_k      = W_j_cal[mask]
        local_idx  = np.arange(n_k)
        local_perm = local_idx.copy()
        while True:
            rng.shuffle(local_perm)
            if not np.any(local_perm == local_idx):
                break
        U_k        = W_i_k * W_j_k                  # same-item
        V_k        = W_i_k * W_j_k[local_perm]      # cross-item
        excess_k   = (U_k - V_k).astype(float)
        mean_exc   = excess_k.mean()
        std_exc    = excess_k.std(ddof=1)
        delta_k[k] = float(np.clip(mean_exc + z * std_exc / np.sqrt(n_k), 0.0, 1.0))

    return delta_k


# ── Per-stratum wealth builders ───────────────────────────────────────────────

def build_wealth_theoretical(
    W_i: np.ndarray,
    W_j: np.ndarray,
    p_i: np.ndarray,
    p_j: np.ndarray,
    strata: np.ndarray,
    K_val: int,
    rng: np.random.Generator,
) -> list:
    """Semi-oracle: epsilon from true marginals, test window only."""
    eps_k = stratum_epsilon(p_i, p_j, strata, Q=K_val)
    per_M = []
    for k in range(K_val):
        mask = np.where(strata == k)[0]
        n_k  = len(mask)
        if n_k <= 1:
            per_M.append(np.ones(max(n_k, 1))); continue
        W_i_k = W_i[mask]; W_j_k = W_j[mask]
        perm  = np.arange(n_k)
        while True:
            rng.shuffle(perm)
            if not np.any(perm == np.arange(n_k)): break
        V = W_i_k * W_j_k[perm]; U = W_i_k * W_j_k
        eps_item = np.full(n_k, eps_k[k]); delta = 2.0 * eps_item
        per_M.append(accumulate_wealth_mixture(U, V, delta, clip_lam=True, eps_per_item=eps_item))
    return per_M


def build_wealth_calibrated(
    W_i: np.ndarray,
    W_j: np.ndarray,
    strata: np.ndarray,
    K_val: int,
    rng: np.random.Generator,
    delta_k: np.ndarray,
) -> list:
    """Calibrated: use pre-computed delta_k from calibration window."""
    per_M = []
    for k in range(K_val):
        mask = np.where(strata == k)[0]
        n_k  = len(mask)
        if n_k <= 1:
            per_M.append(np.ones(max(n_k, 1))); continue
        W_i_k = W_i[mask]; W_j_k = W_j[mask]
        perm  = np.arange(n_k)
        while True:
            rng.shuffle(perm)
            if not np.any(perm == np.arange(n_k)): break
        V        = W_i_k * W_j_k[perm]; U = W_i_k * W_j_k
        eps_item = np.full(n_k, delta_k[k] / 2.0)
        delta    = np.full(n_k, delta_k[k])
        per_M.append(accumulate_wealth_mixture(U, V, delta, clip_lam=True, eps_per_item=eps_item))
    return per_M


def interleave_product(sub_M: list, strata: np.ndarray, K_val: int) -> np.ndarray:
    N_total = len(strata)
    ptr = np.zeros(K_val, dtype=int); cur = np.ones(K_val); E = np.ones(N_total)
    for t in range(N_total):
        k_t = strata[t]
        if ptr[k_t] < len(sub_M[k_t]):
            cur[k_t] = sub_M[k_t][ptr[k_t]]; ptr[k_t] += 1
        E[t] = cur.prod()
    return E


def med_det(lst):
    fired = [x for x in lst if x is not None]
    return int(np.median(fired)) if fired else None

def med_str(v):
    return str(v) if v is not None else "—"


# ── Core runner ───────────────────────────────────────────────────────────────

def run_one(
    cal_dgp_fn,
    test_dgp_fn,
    N_val: int,
    R_val: int,
    seed: int,
    n_cal: int,
) -> dict:
    """
    Run R replications with calibration from cal_dgp_fn and testing on test_dgp_fn.
    Returns rates and medians for both theoretical (oracle) and excess-UCB calibrated.
    """
    rng = np.random.default_rng(seed)

    reject_theo  = []; reject_cal  = []
    det_theo     = []; det_cal     = []

    for _ in range(R_val):
        # Calibration window
        cal_data    = cal_dgp_fn(n_cal, rng)
        strata_cal  = assign_strata(cal_data['theta'], Q=K)
        delta_k     = calibrate_excess_ucb(cal_data['W_i'], cal_data['W_j'], strata_cal, K, rng)

        # Test window
        test_data   = test_dgp_fn(N_val, rng)
        strata_test = assign_strata(test_data['theta'], Q=K)

        # Theoretical
        sub_M_theo = build_wealth_theoretical(
            test_data['W_i'], test_data['W_j'],
            test_data['p_i'], test_data['p_j'],
            strata_test, K, rng,
        )
        E_theo = interleave_product(sub_M_theo, strata_test, K)
        f_t, t_t = ville_reject(E_theo)

        # Excess-UCB calibrated
        sub_M_cal = build_wealth_calibrated(
            test_data['W_i'], test_data['W_j'],
            strata_test, K, rng, delta_k,
        )
        E_cal = interleave_product(sub_M_cal, strata_test, K)
        f_c, t_c = ville_reject(E_cal)

        reject_theo.append(f_t); det_theo.append(t_t + 1 if t_t is not None else None)
        reject_cal.append(f_c);  det_cal.append(t_c  + 1 if t_c is not None else None)

    return dict(
        rate_theo = float(np.mean(reject_theo)),
        rate_cal  = float(np.mean(reject_cal)),
        med_theo  = med_det(det_theo),
        med_cal   = med_det(det_cal),
        n_fired_theo = int(sum(reject_theo)),
        n_fired_cal  = int(sum(reject_cal)),
    )


# ── PART 1: Type-I ────────────────────────────────────────────────────────────

print("=" * 90)
print("PART 1 — TYPE-I ERROR")
print("  Calibration + test window both from null DGP (same condition).")
print(f"  N_test={N_TEST}, K={K}, R={R}")
print("=" * 90)
print()

typeI_rows = []

for null_label, dgp_fn, seed_offset in [
    ("static H0", null_dgp,       1000),
    ("drift H0",  drift_null_dgp, 2000),
]:
    print(f"  --- {null_label} ---")
    print(f"  {'n_cal':>6}  {'Theo Type-I':>12}  {'Cal Type-I':>12}  {'Cal Status':>32}")
    print("  " + "-" * 72)
    for n_cal in N_CAL_GRID:
        seed = SEED + seed_offset + n_cal
        res  = run_one(dgp_fn, dgp_fn, N_TEST, R, seed, n_cal)
        cal_status = (f"*** VIOLATION ({res['rate_cal']:.3f}>{ALPHA}) ***" if res['rate_cal'] > ALPHA else "OK")
        print(f"  n_cal={n_cal:>4}  theo={res['rate_theo']:.3f}  cal={res['rate_cal']:.3f}  {cal_status}")
        typeI_rows.append(dict(
            part='typeI', condition=null_label, n_cal=n_cal, gamma='null',
            rate_theo=res['rate_theo'], rate_cal=res['rate_cal'],
            med_theo=res['med_theo'], med_cal=res['med_cal'],
            n_fired_theo=res['n_fired_theo'], n_fired_cal=res['n_fired_cal'],
            flag_cal=('VIOLATION' if res['rate_cal'] > ALPHA else ''),
        ))
    print()


# ── PART 2: Power ─────────────────────────────────────────────────────────────

print("=" * 90)
print("PART 2 — POWER")
print("  Calibration: null_dgp (clean). Test: alt_dgp(gamma).")
print("  Baseline: theoretical oracle on test window only (fair comparison).")
print(f"  N_test={N_TEST}, K={K}, R={R}")
print("=" * 90)
print()

power_rows = []

for gamma in GAMMA_GRID:
    test_dgp_fn = lambda N, rng, g=gamma: alt_dgp(N, g, rng)
    print(f"  --- gamma={gamma:.2f} ---")
    print(f"  {'n_cal':>6}  {'Theo':>7}  {'Med Theo':>10}  {'Cal':>6}  {'Med Cal':>9}  {'Delta':>8}")
    print("  " + "-" * 60)
    for n_cal in N_CAL_GRID:
        seed = SEED + 3000 + int(gamma * 100) * 10 + N_CAL_GRID.index(n_cal)
        res  = run_one(null_dgp, test_dgp_fn, N_TEST, R, seed, n_cal)
        print(
            f"  n_cal={n_cal:>4}  "
            f"theo={res['rate_theo']:.3f}  med={med_str(res['med_theo']):>9}  "
            f"cal={res['rate_cal']:.3f}  med={med_str(res['med_cal']):>8}  "
            f"delta={res['rate_cal']-res['rate_theo']:+.3f}"
        )
        power_rows.append(dict(
            part='power', condition=f"alt H1 gamma={gamma:.2f}", n_cal=n_cal, gamma=gamma,
            rate_theo=res['rate_theo'], rate_cal=res['rate_cal'],
            med_theo=res['med_theo'], med_cal=res['med_cal'],
            n_fired_theo=res['n_fired_theo'], n_fired_cal=res['n_fired_cal'],
            flag_cal='',
        ))
    print()


# ── SUMMARY ───────────────────────────────────────────────────────────────────

print("=" * 90)
print("COMPACT SUMMARY: Theoretical (oracle) vs Excess-UCB Calibrated")
print(f"  K={K}, N_test={N_TEST}, R={R}  Format: theo/cal  (*=Type-I violation)")
print("=" * 90)
print()
print(f"  Type-I:    {'n_cal=100':>12} | {'n_cal=250':>12} | {'n_cal=500':>12}")
print("  " + "-" * 54)
for null_label in ["static H0", "drift H0"]:
    parts = []
    for n_cal in N_CAL_GRID:
        row = next(r for r in typeI_rows if r['condition']==null_label and r['n_cal']==n_cal)
        fv  = '*' if row['rate_cal'] > ALPHA else ' '
        parts.append(f"{row['rate_theo']:.3f}/{fv}{row['rate_cal']:.3f}")
    print(f"  {null_label:<12} " + " | ".join(f"{p:>12}" for p in parts))
print()
print(f"  Power:     {'n_cal=100':>12} | {'n_cal=250':>12} | {'n_cal=500':>12}")
print("  " + "-" * 54)
for gamma in GAMMA_GRID:
    parts = []
    for n_cal in N_CAL_GRID:
        row = next(r for r in power_rows if abs(r['gamma']-gamma)<1e-9 and r['n_cal']==n_cal)
        parts.append(f"{row['rate_theo']:.3f}/{row['rate_cal']:.3f}")
    print(f"  gamma={gamma:.2f}      " + " | ".join(f"{p:>12}" for p in parts))
print()

violations = [r for r in typeI_rows if r['flag_cal'] == 'VIOLATION']
if violations:
    print("  *** VIOLATIONS: ***")
    for v in violations: print(f"    {v['condition']} n_cal={v['n_cal']} cal_rate={v['rate_cal']:.4f}")
else:
    print("  Calibrated Type-I: all <= 0.05 — no violations.")
print()


# ── Write CSV ─────────────────────────────────────────────────────────────────

all_rows = typeI_rows + power_rows
csv_path = '/home/lyra/projects/evalue-sheaf/code/evalue/results_benign_window.csv'
fieldnames = [
    'part', 'condition', 'n_cal', 'gamma',
    'rate_theo', 'rate_cal', 'med_theo', 'med_cal',
    'n_fired_theo', 'n_fired_cal', 'flag_cal',
]
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
print(f"CSV written to {csv_path}")
print("Done.")
