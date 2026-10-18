"""
run_strongsignal_sweep.py — Strong-signal sweep for the PRODUCT combiner (K=5).

PURPOSE:
  The earlier run_stratified_combiners.py showed product-combiner power <= 0.026 at
  gamma=0.10, N=2000 — too low to meaningfully test validity while the test is active.
  This script sweeps larger gamma and N to find a regime where power is substantial
  (>= 0.5), then re-measures Type-I false-rejection at those same settings to verify
  that validity holds while the test is actively firing.

PARTS:
  PART 1 — POWER CURVE
    PRODUCT combiner, K=5.
    gamma in [0.10, 0.20, 0.30, 0.50] x N in [2000, 5000, 10000].
    DGP: alt_dgp (shared-shock alternative), true strata.
    Reports power and median detection time per cell.

  PART 2 — VALIDITY UNDER ACTIVE FIRING
    At each (gamma, N) cell with power >= 0.5:
      (b) Drift H0, true strata  — drift_null_dgp + true_strata_fn
      (d) Drift H0, noisy strata — drift_null_dgp + noisy_strata_fn
    Reports Type-I false-rejection rate.  A value > 0.05 is a validity BREAK.

Parameters: K=5, R=500, alpha=0.05, SEED=42.
Results saved to code/evalue/results_stratified_strongsignal.csv.
"""

import csv
import sys
import numpy as np

# Adjust path so we can import from this directory
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

K           = 5
R           = 500
ALPHA       = 0.05
SEED        = 42
NOISE_SIGMA = 0.20

GAMMA_GRID = [0.10, 0.20, 0.30, 0.50]
N_GRID     = [2000, 5000, 10000]

POWER_THRESHOLD = 0.5   # what counts as "substantial power"

print(f"SEED = {SEED}")
print(f"Parameters: K={K}, R={R}, alpha={ALPHA}, threshold={THRESHOLD}")
print(f"Gamma grid: {GAMMA_GRID}")
print(f"N grid: {N_GRID}")
print()


# ── Per-stratum wealth builder (PRODUCT combiner) ─────────────────────────────

def build_per_stratum_wealth(
    W_i: np.ndarray,
    W_j: np.ndarray,
    p_i: np.ndarray,
    p_j: np.ndarray,
    strata: np.ndarray,
    K_val: int,
    rng: np.random.Generator,
) -> list:
    """Build per-stratum wealth using semi-oracle epsilon (true p)."""
    eps_k = stratum_epsilon(p_i, p_j, strata, Q=K_val)
    per_stratum_M = []

    for k in range(K_val):
        mask = np.where(strata == k)[0]
        n_k  = len(mask)

        if n_k <= 1:
            per_stratum_M.append(np.ones(max(n_k, 1)))
            continue

        W_i_k = W_i[mask]
        W_j_k = W_j[mask]

        local_idx  = np.arange(n_k)
        local_perm = local_idx.copy()
        while True:
            rng.shuffle(local_perm)
            if not np.any(local_perm == local_idx):
                break

        V        = W_i_k * W_j_k[local_perm]
        U        = W_i_k * W_j_k
        eps_item = np.full(n_k, eps_k[k])
        delta    = 2.0 * eps_item

        M_k = accumulate_wealth_mixture(U, V, delta, clip_lam=True, eps_per_item=eps_item)
        per_stratum_M.append(M_k)

    return per_stratum_M


def interleave_product(sub_M: list, strata: np.ndarray, K_val: int) -> np.ndarray:
    """
    Product combiner: E_t = prod_k E^(k)_{count_k(t)}.
    Valid IF per-stratum e-processes are independent under H0.
    """
    N_total   = len(strata)
    ptr       = np.zeros(K_val, dtype=int)
    current_W = np.ones(K_val)
    E         = np.ones(N_total)

    for t in range(N_total):
        k_t = strata[t]
        if ptr[k_t] < len(sub_M[k_t]):
            current_W[k_t] = sub_M[k_t][ptr[k_t]]
            ptr[k_t] += 1
        E[t] = current_W.prod()

    return E


# ── Strata factories ──────────────────────────────────────────────────────────

def true_strata_fn(data: dict, K_val: int, rng: np.random.Generator) -> np.ndarray:
    return assign_strata(data['theta'], Q=K_val)


def noisy_strata_fn(data: dict, K_val: int, rng: np.random.Generator) -> np.ndarray:
    theta_hat = np.clip(
        data['theta'] + rng.normal(0.0, NOISE_SIGMA, size=len(data['theta'])),
        0.0, 1.0
    )
    return assign_strata(theta_hat, Q=K_val)


# ── Core runner ───────────────────────────────────────────────────────────────

def run_product(
    dgp_fn,
    strata_fn,
    N: int,
    R: int,
    seed: int,
) -> dict:
    """
    Run R simulations of the PRODUCT combiner.
    Returns rejection rate, median detection time, and all detection times.
    """
    rng = np.random.default_rng(seed)
    reject = []
    det    = []

    for _ in range(R):
        data   = dgp_fn(N, rng)
        strata = strata_fn(data, K, rng)

        sub_M = build_per_stratum_wealth(
            data['W_i'], data['W_j'],
            data['p_i'], data['p_j'],
            strata, K, rng,
        )

        E_prod = interleave_product(sub_M, strata, K)
        fired, t = ville_reject(E_prod)

        reject.append(fired)
        det.append(t + 1 if t is not None else None)

    rate   = float(np.mean(reject))
    fired_dets = [d for d in det if d is not None]
    med_det    = int(np.median(fired_dets)) if fired_dets else None
    return dict(rate=rate, med_det=med_det, n_fired=len(fired_dets))


def median_str(v) -> str:
    return str(v) if v is not None else "—"


# ── PART 1: Power curve ───────────────────────────────────────────────────────

print("=" * 80)
print("PART 1 — POWER CURVE: PRODUCT combiner, K=5")
print("  Alt: alt_dgp (shared-shock), true strata")
print("=" * 80)
print()
print(f"  {'gamma':>6}  {'N':>7}  {'Power':>7}  {'Med Det Time':>13}  {'N fired':>8}")
print("  " + "-" * 52)

power_results = []
strong_signal_cells = []   # cells with power >= POWER_THRESHOLD

for gamma in GAMMA_GRID:
    dgp_fn = lambda N_arg, rng, g=gamma: alt_dgp(N_arg, g, rng)
    for N in N_GRID:
        seed = SEED + 200 + int(gamma * 100) * 10 + N_GRID.index(N)
        res  = run_product(dgp_fn, true_strata_fn, N, R, seed)
        rate = res['rate']
        med  = res['med_det']
        print(
            f"  gamma={gamma:.2f}  N={N:>6}  power={rate:.3f}  med_det={median_str(med):>8}  n_fired={res['n_fired']}"
        )
        row = dict(
            part='power',
            gamma=gamma, N=N,
            condition='alt H1',
            strata_type='true',
            rate=rate,
            med_det=med,
            n_fired=res['n_fired'],
            flag=('HIGH_POWER' if rate >= POWER_THRESHOLD else ''),
        )
        power_results.append(row)
        if rate >= POWER_THRESHOLD:
            strong_signal_cells.append((gamma, N))

print()
print(f"  Cells with power >= {POWER_THRESHOLD}: {strong_signal_cells}")
print()


# ── PART 2: Validity under active firing ──────────────────────────────────────

print("=" * 80)
print("PART 2 — TYPE-I VALIDITY at high-power (gamma, N) cells")
print("  Nulls tested: (b) drift H0 true strata, (d) drift H0 noisy strata")
print(f"  NOTE: rate > {ALPHA} = VALIDITY BREAK")
print("=" * 80)
print()

validity_results = []
violations_found = []

if not strong_signal_cells:
    print("  No cells reached power >= 0.5 — no high-power validity check performed.")
    print("  Consider raising gamma or N further.")
else:
    print(f"  {'Condition':<35}  {'gamma':>6}  {'N':>7}  {'Type-I':>8}  {'Status':>20}")
    print("  " + "-" * 82)

    for (gamma, N) in strong_signal_cells:
        for (cond_label, dgp_fn, strata_fn, seed_offset) in [
            ('(b) drift H0 true strata',   drift_null_dgp, true_strata_fn,  300),
            ('(d) drift H0 noisy strata',  drift_null_dgp, noisy_strata_fn, 400),
        ]:
            seed = SEED + seed_offset + int(gamma * 100) * 10 + N_GRID.index(N)
            res  = run_product(dgp_fn, strata_fn, N, R, seed)
            rate = res['rate']

            if rate > ALPHA:
                status = f"*** VIOLATION ({rate:.3f} > {ALPHA}) ***"
                violations_found.append((cond_label, gamma, N, rate))
            else:
                status = "OK"

            print(f"  {cond_label:<35}  gamma={gamma:.2f}  N={N:>6}  rate={rate:.3f}  {status}")

            row = dict(
                part='validity',
                gamma=gamma, N=N,
                condition=cond_label,
                strata_type=('noisy' if 'noisy' in cond_label else 'true'),
                rate=rate,
                med_det=res['med_det'],
                n_fired=res['n_fired'],
                flag=('VIOLATION' if rate > ALPHA else ''),
            )
            validity_results.append(row)

print()


# ── PART 2b: Also test static null at high-power cells ───────────────────────
# (matches condition (a) from the original script)

static_results = []
if strong_signal_cells:
    print("  Static H0 validity check (for completeness):")
    print(f"  {'Condition':<35}  {'gamma':>6}  {'N':>7}  {'Type-I':>8}  {'Status':>20}")
    print("  " + "-" * 82)
    for (gamma, N) in strong_signal_cells:
        seed = SEED + 500 + int(gamma * 100) * 10 + N_GRID.index(N)
        res  = run_product(null_dgp, true_strata_fn, N, R, seed)
        rate = res['rate']
        status = (f"*** VIOLATION ({rate:.3f} > {ALPHA}) ***" if rate > ALPHA else "OK")
        if rate > ALPHA:
            violations_found.append(('(a) static H0 true strata', gamma, N, rate))
        print(f"  {'(a) static H0 true strata':<35}  gamma={gamma:.2f}  N={N:>6}  rate={rate:.3f}  {status}")
        row = dict(
            part='validity',
            gamma=gamma, N=N,
            condition='(a) static H0 true strata',
            strata_type='true',
            rate=rate,
            med_det=res['med_det'],
            n_fired=res['n_fired'],
            flag=('VIOLATION' if rate > ALPHA else ''),
        )
        static_results.append(row)
    print()


# ── SUMMARY ───────────────────────────────────────────────────────────────────

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

if violations_found:
    print("  *** TYPE-I VIOLATIONS DETECTED ***")
    for (cond, g, n, rate) in violations_found:
        print(f"    {cond}  gamma={g:.2f}  N={n}  rate={rate:.4f}")
else:
    print("  Type-I: ALL rates <= 0.05 — validity holds at high-power settings.")

print()
print("  Power summary (cells >= 0.5):")
for row in power_results:
    if row['rate'] >= POWER_THRESHOLD:
        print(f"    gamma={row['gamma']:.2f}  N={row['N']}  power={row['rate']:.3f}  med_det={median_str(row['med_det'])}")

if not strong_signal_cells:
    print("    (none — power did not reach 0.5 in this sweep)")

print()


# ── Write CSV ─────────────────────────────────────────────────────────────────

all_rows = power_results + validity_results + static_results
csv_path = '/home/lyra/projects/evalue-sheaf/code/evalue/results_stratified_strongsignal.csv'
fieldnames = ['part', 'gamma', 'N', 'condition', 'strata_type', 'rate', 'med_det', 'n_fired', 'flag']
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
print(f"CSV written to {csv_path}")
print("Done.")
