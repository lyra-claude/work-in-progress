"""
run_stratified_combiners.py — Side-by-side comparison of averaged vs product combiners
for the difficulty-stratified e-process portfolio.

CONSTRUCTION (per stratum, unchanged from run_stratified_experiment.py):
  Items partitioned into K difficulty strata.
  Within each stratum k, Method A cross-item e-process:
    V = W_i^s * W_j^{t(s)}, t(s) a derangement within the stratum.
  Per-stratum wealth E^(k)_1, ..., E^(k)_{n_k}.

COMBINERS COMPARED:

  Averaged (existing):
    E_t^avg = (1/K) * sum_k E^(k)_{count_k(t)}
    Valid by convexity of e-processes (no independence assumption needed).
    Known issue: near-zero power (sub-processes must each reach K*threshold).

  Product (new):
    E_t^prod = prod_k E^(k)_{count_k(t)}
    Valid IF per-stratum e-processes are independent under H0.
    Independence argument: items assigned to different strata have different difficulty
    bins, and under H0 their W_i, W_j outcomes are conditionally independent.
    HOWEVER: under DRIFT, the shared time-varying base rates create cross-stratum
    correlation. This could break the product's validity. We MEASURE, not assume.

CONDITIONS TESTED (matching run_stratified_experiment.py exactly):
  (a) Static H0, true strata
  (b) Drift  H0, true strata
  (c') Static H0, hard plug-in estimated strata + epsilon
  (d) Drift  H0, noisy strata (semi-oracle epsilon)
  (e) Alt H1 gamma=0.10, true strata — power

Parameters: N=2000, R=500, alpha=0.05, SEED=42 (matching run_stratified_experiment.py).
K grid: 2, 5, 10.

Results saved to code/evalue/results_stratified_combiners.csv.
"""

import csv
import numpy as np

from dgp import null_dgp, alt_dgp, drift_null_dgp
from eprocess import (
    assign_strata,
    stratum_epsilon,
    accumulate_wealth_mixture,
    ville_reject,
    THRESHOLD,
)

# ── Parameters ────────────────────────────────────────────────────────────────────

N            = 2000
R            = 500
ALPHA        = 0.05
SEED         = 42
K_GRID       = [2, 5, 10]
NOISE_SIGMA  = 0.20   # matching run_stratified_experiment.py

print(f"SEED = {SEED}")
print(f"Parameters: N={N}, R={R}, alpha={ALPHA}, threshold={THRESHOLD}")
print(f"K grid: {K_GRID}, noise sigma: {NOISE_SIGMA}")
print()


# ── Core per-stratum builder (identical to run_stratified_experiment.py) ──────────

def build_per_stratum_wealth(
    W_i: np.ndarray,
    W_j: np.ndarray,
    p_i: np.ndarray,
    p_j: np.ndarray,
    strata: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> list:
    """
    Build the per-stratum wealth arrays using the standard semi-oracle epsilon.
    Returns a list of K arrays (one per stratum).
    """
    eps_k = stratum_epsilon(p_i, p_j, strata, Q=K)   # true p, shape (K,)
    per_stratum_M = []

    for k in range(K):
        mask = np.where(strata == k)[0]
        n_k  = len(mask)

        if n_k <= 1:
            per_stratum_M.append(np.ones(max(n_k, 1)))
            continue

        W_i_k = W_i[mask]
        W_j_k = W_j[mask]

        # Derangement within stratum
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


def build_per_stratum_wealth_hard_plugin(
    W_i: np.ndarray,
    W_j: np.ndarray,
    strata: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> list:
    """
    Hard plug-in variant: epsilon estimated from empirical W within each stratum.
    """
    eps_k = _stratum_epsilon_empirical(W_i, W_j, strata, K)
    per_stratum_M = []

    for k in range(K):
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

        U        = W_i_k * W_j_k
        V        = W_i_k * W_j_k[local_perm]
        eps_item = np.full(n_k, eps_k[k])
        delta    = 2.0 * eps_item

        M_k = accumulate_wealth_mixture(U, V, delta, clip_lam=True, eps_per_item=eps_item)
        per_stratum_M.append(M_k)

    return per_stratum_M


def _stratum_epsilon_empirical(
    W_i: np.ndarray,
    W_j: np.ndarray,
    strata: np.ndarray,
    K: int,
) -> np.ndarray:
    """Estimate epsilon per stratum from empirical W (no access to true p)."""
    eps = np.zeros(K)
    for k in range(K):
        mask = strata == k
        n_k  = mask.sum()
        if n_k < 2:
            eps[k] = 0.0
            continue
        idx_k = np.where(mask)[0]
        half  = max(1, n_k // 2)
        range_i = abs(W_i[idx_k[:half]].mean() - W_i[idx_k[half:]].mean()) / 2.0
        range_j = abs(W_j[idx_k[:half]].mean() - W_j[idx_k[half:]].mean()) / 2.0
        eps[k]  = max(range_i, range_j)
    return eps


# ── Combiners: interleave per-stratum wealth into a full-stream process ───────────

def interleave_averaged(sub_M: list, strata: np.ndarray, K: int) -> np.ndarray:
    """
    Averaged combiner: E_t = (1/K) * sum_k E^(k)_{count_k(t)}.
    Valid by convexity — no independence assumption.
    """
    N_total   = len(strata)
    ptr       = np.zeros(K, dtype=int)
    current_W = np.ones(K)
    E         = np.ones(N_total)

    for t in range(N_total):
        k_t = strata[t]
        if ptr[k_t] < len(sub_M[k_t]):
            current_W[k_t] = sub_M[k_t][ptr[k_t]]
            ptr[k_t] += 1
        E[t] = current_W.mean()

    return E


def interleave_product(sub_M: list, strata: np.ndarray, K: int) -> np.ndarray:
    """
    Product combiner: E_t = prod_k E^(k)_{count_k(t)}.
    Valid IF per-stratum e-processes are independent under H0.
    VALIDITY IS NOT GUARANTEED under drift — measured empirically.
    """
    N_total   = len(strata)
    ptr       = np.zeros(K, dtype=int)
    current_W = np.ones(K)
    E         = np.ones(N_total)

    for t in range(N_total):
        k_t = strata[t]
        if ptr[k_t] < len(sub_M[k_t]):
            current_W[k_t] = sub_M[k_t][ptr[k_t]]
            ptr[k_t] += 1
        E[t] = current_W.prod()

    return E


# ── Strata factories ──────────────────────────────────────────────────────────────

def true_strata_fn(data: dict, K: int, rng: np.random.Generator) -> np.ndarray:
    return assign_strata(data['theta'], Q=K)


def noisy_strata_fn(data: dict, K: int, rng: np.random.Generator) -> np.ndarray:
    theta_hat = np.clip(data['theta'] + rng.normal(0.0, NOISE_SIGMA, size=len(data['theta'])), 0.0, 1.0)
    return assign_strata(theta_hat, Q=K)


# ── Monte-Carlo runner ────────────────────────────────────────────────────────────

def run_condition(
    dgp_fn,
    K: int,
    strata_fn,
    R: int,
    N: int,
    seed: int,
    use_hard_plugin: bool = False,
) -> dict:
    """
    Run R simulations and return rejection arrays for BOTH combiners.
    For each replicate, per-stratum wealth is built ONCE and fed to both combiners.
    This ensures both combiners see identical per-stratum sub-processes.
    """
    rng = np.random.default_rng(seed)

    reject_avg  = []
    reject_prod = []
    det_avg     = []
    det_prod    = []

    for _ in range(R):
        data   = dgp_fn(N, rng)
        strata = strata_fn(data, K, rng)

        if use_hard_plugin:
            sub_M = build_per_stratum_wealth_hard_plugin(
                data['W_i'], data['W_j'], strata, K, rng
            )
        else:
            sub_M = build_per_stratum_wealth(
                data['W_i'], data['W_j'],
                data['p_i'], data['p_j'],
                strata, K, rng,
            )

        E_avg  = interleave_averaged(sub_M, strata, K)
        E_prod = interleave_product(sub_M, strata, K)

        fired_a, t_a = ville_reject(E_avg)
        fired_p, t_p = ville_reject(E_prod)

        reject_avg.append(fired_a)
        reject_prod.append(fired_p)
        det_avg.append(t_a + 1 if t_a is not None else None)
        det_prod.append(t_p + 1 if t_p is not None else None)

    return dict(
        reject_avg  = np.array(reject_avg),
        reject_prod = np.array(reject_prod),
        det_avg     = det_avg,
        det_prod    = det_prod,
    )


def median_det(det_list: list) -> str:
    fired = [t for t in det_list if t is not None]
    if not fired:
        return "—"
    return str(int(np.median(fired)))


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 100)
    print("DIFFICULTY-STRATIFIED E-PROCESS: AVERAGED vs PRODUCT COMBINER COMPARISON")
    print(f"  N={N}, R={R}, alpha={ALPHA}, threshold={THRESHOLD}, SEED={SEED}")
    print("=" * 100)
    print()

    dgp_null   = null_dgp
    dgp_drift  = drift_null_dgp
    dgp_alt010 = lambda N, rng: alt_dgp(N, 0.10, rng)

    # Condition seeds matching run_stratified_experiment.py offsets
    seeds = {
        'a_static_null':  SEED + 100,
        'b_drift_null':   SEED + 101,
        'cp_hard_plugin': SEED + 105,
        'd_drift_noisy':  SEED + 103,
        'e_alt010':       SEED + 104,
    }

    conditions = [
        # (label, dgp_fn, strata_fn, seed_key, metric, use_hard_plugin)
        ('(a) static H0, true strata',      dgp_null,   true_strata_fn,  'a_static_null',  'Type-I', False),
        ('(b) drift  H0, true strata',      dgp_drift,  true_strata_fn,  'b_drift_null',   'Type-I', False),
        ("(c') static H0, hard plug-in",    dgp_null,   noisy_strata_fn, 'cp_hard_plugin', 'Type-I', True),
        ('(d) drift  H0, noisy strata',     dgp_drift,  noisy_strata_fn, 'd_drift_noisy',  'Type-I', False),
        ('(e) alt H1 gamma=0.10',           dgp_alt010, true_strata_fn,  'e_alt010',       'Power ', False),
    ]

    all_rows = []

    header = (
        f"{'Condition':<44} {'K':>3}  "
        f"{'AVG Rate':>9}  {'AVG Status':>18}  "
        f"{'PROD Rate':>10}  {'PROD Status':>18}"
    )
    print(header)
    print("-" * 110)

    for (label, dgp_fn, strata_fn, seed_key, metric_label, use_plugin) in conditions:
        print()
        for K in K_GRID:
            res = run_condition(
                dgp_fn, K, strata_fn,
                R, N, seeds[seed_key],
                use_hard_plugin=use_plugin,
            )

            rate_avg  = res['reject_avg'].mean()
            rate_prod = res['reject_prod'].mean()
            med_avg   = median_det(res['det_avg'])
            med_prod  = median_det(res['det_prod'])

            is_typeI = (metric_label.strip() == 'Type-I')

            def status(rate, is_t1):
                if is_t1:
                    if rate > ALPHA:
                        return f"*** VIOLATION ({rate:.3f}>{ALPHA}) ***"
                    return "OK"
                return f"power={rate:.3f}"

            s_avg  = status(rate_avg,  is_typeI)
            s_prod = status(rate_prod, is_typeI)

            print(
                f"  {label:<44} K={K:>2}  "
                f"{metric_label}={rate_avg:.3f}  {s_avg:<20}  "
                f"{metric_label}={rate_prod:.3f}  {s_prod:<20}"
            )

            all_rows.append(dict(
                condition       = label,
                K               = K,
                metric          = metric_label.strip(),
                avg_rate        = f"{rate_avg:.4f}",
                avg_med_det     = med_avg,
                avg_status      = s_avg,
                prod_rate       = f"{rate_prod:.4f}",
                prod_med_det    = med_prod,
                prod_status     = s_prod,
            ))

    # ── Compact summary table ──────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("SUMMARY TABLE: AVERAGED vs PRODUCT COMBINER")
    print("  Format: avg_rate / prod_rate   (* = violation > 0.05)")
    print("=" * 100)
    print()
    print(f"  {'Condition':<44} | {'K=2':>14} | {'K=5':>14} | {'K=10':>14}")
    print("  " + "-" * 92)

    # Group rows by condition
    by_label = {}
    for row in all_rows:
        key = row['condition']
        if key not in by_label:
            by_label[key] = {}
        by_label[key][row['K']] = row

    for (label, *_) in conditions:
        cells = {}
        for K in K_GRID:
            row = by_label.get(label, {}).get(K)
            if row is None:
                cells[K] = '   —   /   —   '
                continue
            ra  = float(row['avg_rate'])
            rp  = float(row['prod_rate'])
            is_typeI = (row['metric'] == 'Type-I')
            fa  = '*' if (is_typeI and ra > ALPHA) else ' '
            fp  = '*' if (is_typeI and rp > ALPHA) else ' '
            cells[K] = f"{fa}{ra:.3f} /{fp}{rp:.3f}"
        print(f"  {label:<44} | {cells[2]:>14} | {cells[5]:>14} | {cells[10]:>14}")

    print()
    print("  Columns: avg / prod  — * = Type-I violation > 0.05")
    print()

    # ── TYPE-I VALIDITY SUMMARY ────────────────────────────────────────────────────
    print("VALIDITY VERDICT:")
    prod_violations = [
        (row['condition'], row['K'], float(row['prod_rate']))
        for row in all_rows
        if row['metric'] == 'Type-I' and float(row['prod_rate']) > ALPHA
    ]
    avg_violations = [
        (row['condition'], row['K'], float(row['avg_rate']))
        for row in all_rows
        if row['metric'] == 'Type-I' and float(row['avg_rate']) > ALPHA
    ]

    if not avg_violations:
        print("  Averaged combiner: ALL Type-I rates <= 0.05 (valid)")
    else:
        print(f"  Averaged combiner: {len(avg_violations)} violation(s) — {avg_violations}")

    if not prod_violations:
        print("  Product combiner:  ALL Type-I rates <= 0.05 (valid)")
    else:
        print(f"  Product combiner:  *** {len(prod_violations)} VIOLATION(S) ***")
        for cond, K, rate in prod_violations:
            print(f"    {cond}  K={K}  rate={rate:.4f}")

    print()

    # ── Write CSV ──────────────────────────────────────────────────────────────────
    csv_path = '/home/lyra/projects/evalue-sheaf/code/evalue/results_stratified_combiners.csv'
    fieldnames = ['condition', 'K', 'metric', 'avg_rate', 'avg_med_det', 'avg_status',
                  'prod_rate', 'prod_med_det', 'prod_status']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"CSV written to {csv_path}")
    print("Done.")


if __name__ == '__main__':
    main()
