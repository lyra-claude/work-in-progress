"""
run_benign_window_pastonly.py — §5b benign-window calibration with PAST-ONLY pairing.

Re-runs the same experiment as run_benign_window_calibration.py, but replaces the
within-stratum full-stream derangement (make_derangement_within_strata) with the
predictable, online past-only pairing (make_online_past_pairing).  This satisfies
the F_{n-1}-measurability requirement for anytime/sequential validity.

DESIGN — identical parameters to run_benign_window_calibration.py:
  K=10, R=500, ALPHA=0.05, THRESHOLD=20, SEED=42, N_TEST=5000
  N_CAL_GRID = [100, 250, 500]
  GAMMA_GRID  = [0.10, 0.20, 0.30, 0.40, 0.50]
  Null labels: "static H0" (null_dgp), "drift H0" (drift_null_dgp)
  Seeds: same offsets (+1000, +2000, +3000) as the original.

The two methods compared (product combiner, K=10):
  A) Theoretical (oracle): oracle epsilon from true marginals + past-only pairing
  B) Excess-UCB calibrated: UCB on mean(U-V) from n_cal clean items + past-only pairing

Pairing design for per-stratum wealth builds:
  - The online past-only pairing t is computed over the FULL test stream.
  - has_partner[s] = (t[s] != s) gives the F_{s-1}-measurable sentinel mask (global).
  - Per stratum k: mask = items in stratum k; t_k = t[mask] (global partner indices).
  - Sentinel check per stratum: has_partner_k = (t_k != mask) (compare global indices).
  - V_k[i] = W_j[t_k[i]] uses the global W_j array; sentinel entries are skipped.

OUTPUT: results_benign_window_pastonly.csv (does NOT overwrite results_benign_window.csv)
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
    make_online_past_pairing,
    accumulate_wealth_online,
    accumulate_wealth_online_mixture,
    ville_reject,
    THRESHOLD,
)

# ── Parameters (identical to run_benign_window_calibration.py) ────────────────

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
print(f"Pairing: PAST-ONLY (make_online_past_pairing)")
print()


# ── Calibration: Excess-UCB using past-only pairing ──────────────────────────

def calibrate_excess_ucb_pastonly(
    W_i_cal: np.ndarray,
    W_j_cal: np.ndarray,
    strata_cal: np.ndarray,
    K_val: int,
    rng: np.random.Generator,
    alpha: float = CAL_ALPHA,
) -> np.ndarray:
    """
    Per-stratum delta_k as UCB on excess co-failure, using past-only pairing
    over the calibration window.

    Generates t_cal over the full calibration stream, then per stratum:
      - extracts sentinel mask (t_cal[s] == s in global sense: t_k[i] == mask[i])
      - computes excess (U - V) only for items with valid past partners
      - returns UCB = mean + z * std / sqrt(n_valid), clipped to [0,1]
    """
    n_cal    = len(W_i_cal)
    z        = scipy_norm.ppf(1.0 - alpha)
    delta_k  = np.zeros(K_val)
    t_cal    = make_online_past_pairing(strata_cal, rng, Q=K_val)

    for k in range(K_val):
        mask = np.where(strata_cal == k)[0]
        n_k  = len(mask)
        if n_k < 2:
            delta_k[k] = 1.0   # maximally conservative
            continue

        t_k           = t_cal[mask]            # global partner indices for stratum k items
        has_partner_k = (t_k != mask)          # sentinel: t_k[i] == mask[i] means no partner
        valid         = np.where(has_partner_k)[0]

        if len(valid) < 2:
            delta_k[k] = 1.0
            continue

        W_i_k    = W_i_cal[mask][valid]
        W_j_k_s  = W_j_cal[mask][valid]        # same-item j
        W_j_k_t  = W_j_cal[t_k[valid]]         # partner item j (global index into W_j_cal)

        U_k      = W_i_k * W_j_k_s             # same-item co-failure
        V_k      = W_i_k * W_j_k_t             # cross-item baseline
        excess_k = (U_k - V_k).astype(float)
        mean_exc = excess_k.mean()
        std_exc  = excess_k.std(ddof=1)
        n_valid  = len(valid)
        delta_k[k] = float(np.clip(mean_exc + z * std_exc / np.sqrt(n_valid), 0.0, 1.0))

    return delta_k


# ── Per-stratum wealth builders (past-only pairing) ───────────────────────────

def build_wealth_theoretical_pastonly(
    W_i: np.ndarray,
    W_j: np.ndarray,
    p_i: np.ndarray,
    p_j: np.ndarray,
    strata: np.ndarray,
    K_val: int,
    rng: np.random.Generator,
) -> list:
    """
    Semi-oracle: epsilon from true marginals, past-only pairing, test window only.

    Generates t over the FULL test stream (global indices), then per stratum extracts
    the sentinel mask using global-index comparison (t_k[i] != mask[i]).
    """
    eps_k = stratum_epsilon(p_i, p_j, strata, Q=K_val)
    t     = make_online_past_pairing(strata, rng, Q=K_val)

    per_M = []
    for k in range(K_val):
        mask = np.where(strata == k)[0]
        n_k  = len(mask)
        if n_k == 0:
            per_M.append(np.ones(1)); continue

        W_i_k         = W_i[mask]
        W_j_k_s       = W_j[mask]              # same-item j outcomes
        t_k           = t[mask]                # global partner indices
        has_partner_k = (t_k != mask)          # True where a valid past partner exists

        # V[i] = W_j[t_k[i]] for all i; sentinel values ignored via has_partner_k
        V_k           = W_i_k * W_j[t_k]
        U_k           = W_i_k * W_j_k_s

        eps_item  = np.full(n_k, eps_k[k])
        delta_arr = 2.0 * eps_item
        per_M.append(accumulate_wealth_online_mixture(
            U_k, V_k, delta_arr, has_partner_k,
            clip_lam=True, eps_per_item=eps_item,
        ))
    return per_M


def build_wealth_calibrated_pastonly(
    W_i: np.ndarray,
    W_j: np.ndarray,
    strata: np.ndarray,
    K_val: int,
    rng: np.random.Generator,
    delta_k: np.ndarray,
) -> list:
    """
    Calibrated: use pre-computed delta_k from calibration window, past-only pairing.
    Same global-pairing / per-stratum sentinel logic as theoretical variant.
    """
    t     = make_online_past_pairing(strata, rng, Q=K_val)
    per_M = []
    for k in range(K_val):
        mask = np.where(strata == k)[0]
        n_k  = len(mask)
        if n_k == 0:
            per_M.append(np.ones(1)); continue

        W_i_k         = W_i[mask]
        W_j_k_s       = W_j[mask]
        t_k           = t[mask]
        has_partner_k = (t_k != mask)

        V_k       = W_i_k * W_j[t_k]
        U_k       = W_i_k * W_j_k_s
        eps_item  = np.full(n_k, delta_k[k] / 2.0)
        delta_arr = np.full(n_k, delta_k[k])
        per_M.append(accumulate_wealth_online_mixture(
            U_k, V_k, delta_arr, has_partner_k,
            clip_lam=True, eps_per_item=eps_item,
        ))
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
    Run R replications with calibration from cal_dgp_fn, testing on test_dgp_fn.
    Uses past-only pairing throughout.
    """
    rng = np.random.default_rng(seed)

    reject_theo  = []; reject_cal  = []
    det_theo     = []; det_cal     = []

    for _ in range(R_val):
        # Calibration window
        cal_data   = cal_dgp_fn(n_cal, rng)
        strata_cal = assign_strata(cal_data['theta'], Q=K)
        delta_k    = calibrate_excess_ucb_pastonly(
            cal_data['W_i'], cal_data['W_j'], strata_cal, K, rng,
        )

        # Test window
        test_data   = test_dgp_fn(N_val, rng)
        strata_test = assign_strata(test_data['theta'], Q=K)

        # Theoretical (oracle epsilon + past-only pairing)
        sub_M_theo = build_wealth_theoretical_pastonly(
            test_data['W_i'], test_data['W_j'],
            test_data['p_i'], test_data['p_j'],
            strata_test, K, rng,
        )
        E_theo = interleave_product(sub_M_theo, strata_test, K)
        f_t, t_t = ville_reject(E_theo)

        # Excess-UCB calibrated (past-only pairing)
        sub_M_cal = build_wealth_calibrated_pastonly(
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
print("PART 1 — TYPE-I ERROR (past-only pairing)")
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
        cal_status = (f"*** VIOLATION ({res['rate_cal']:.3f}>{ALPHA}) ***"
                      if res['rate_cal'] > ALPHA else "OK")
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
print("PART 2 — POWER (past-only pairing)")
print("  Calibration: null_dgp (clean). Test: alt_dgp(gamma).")
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
print("COMPACT SUMMARY (past-only pairing): Theoretical (oracle) vs Excess-UCB Calibrated")
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
    for v in violations:
        print(f"    {v['condition']} n_cal={v['n_cal']} cal_rate={v['rate_cal']:.4f}")
else:
    print("  Calibrated Type-I: all <= 0.05 — no violations.")
print()


# ── Write CSV ─────────────────────────────────────────────────────────────────

all_rows = typeI_rows + power_rows
csv_path = '/home/lyra/projects/evalue-sheaf/code/evalue/results_benign_window_pastonly.csv'
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
