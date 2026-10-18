"""
run_stratified_experiment.py — Difficulty-stratified averaged e-process experiment.

CONSTRUCTION:
  Items carry latent difficulty d_i = theta_i (already in the DGPs as theta ~ U[0,1]).
  Partition the N items into K difficulty strata.
  Within each stratum k, run the existing per-pair cross-item e-process (Method A
  construction: V = W_i^s * W_j^{t(s)} with t(s) a derangement within the stratum).
  This gives a per-stratum wealth process E_t^(k).

  Averaged portfolio:
    E_t = (1/K) * sum_k E_t^(k)

  A convex combination of e-processes is itself an e-process (Ville's inequality
  passes through linearity: E[E_t] <= 1 if E[E_t^(k)] <= 1 for all k).
  Reject the FIRST TIME E_t >= 1/alpha = 20 (anytime-valid Ville stopping rule).

CONDITIONS TESTED:
  (a) Static H0 — null_dgp, TRUE difficulty strata (oracle theta).
  (b) Drifting H0 — drift_null_dgp, TRUE difficulty strata.
  (c) Static H0 — null_dgp, NOISY strata (d_hat = theta + N(0, sigma)).
      IMPORTANT: epsilon (and thus delta) is computed from TRUE p_i,p_j values
      within the noisily-assigned bins. So the delta correction adapts to inflated
      within-stratum variation. This is a SEMI-oracle — not the pure plug-in trap.
  (c') Static H0 — HARD PLUG-IN: both strata AND epsilon estimated from data.
      Epsilon computed from empirical W_i, W_j rather than true p values.
      This is the genuine plug-in danger zone: if estimated epsilon is systematically
      underestimated, E[e_s] > 1 is possible, breaking the e-process property.
  (d) Drifting H0 — drift_null_dgp, NOISY strata (semi-oracle as in (c)).
  (e) Alternative H1 — alt_dgp(gamma=0.10), TRUE strata — to measure power.

Note on (c) vs (c'):
  In conditions (c) and (d), epsilon is still computed from TRUE p_i,p_j but within
  wrong bins, causing inflated epsilon (more conservative, not less). This preserves
  validity but kills power further. The hard plug-in (c') estimates epsilon from
  empirical W values, which can underestimate the true range, potentially breaking
  the conservative guarantee.

Parameters: N=2000, R=500, alpha=0.05, SEED=42 (matching run_experiment.py).
K grid: 2, 5, 10.
Noise level for estimated strata: sigma=0.20 (~1 stratum width for K=5).

Results saved to code/evalue/results_stratified.csv.
"""

import csv
import sys
import numpy as np

from dgp import null_dgp, alt_dgp, drift_null_dgp
from eprocess import (
    assign_strata,
    stratum_epsilon,
    accumulate_wealth,
    accumulate_wealth_mixture,
    ville_reject,
    THRESHOLD,
)

# ── Parameters ────────────────────────────────────────────────────────────────────

N       = 2000
R       = 500
ALPHA   = 0.05
SEED    = 42
K_GRID  = [2, 5, 10]
NOISE_SIGMA = 0.20   # Gaussian noise added to theta to simulate estimated difficulty

print(f"SEED = {SEED}")
print(f"Parameters: N={N}, R={R}, alpha={ALPHA}, threshold={THRESHOLD}")
print(f"K grid: {K_GRID}, noise sigma for estimated strata: {NOISE_SIGMA}")
print()


# ── Per-stratum e-process builder ─────────────────────────────────────────────────

def make_derangement_within_stratum_mask(
    idx: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Given a sorted array of item indices belonging to one stratum,
    return a derangement permutation of those indices (no fixed points).
    For size <= 1, returns the indices unchanged (can't derange a singleton).
    """
    if len(idx) <= 1:
        return idx.copy()
    perm = idx.copy()
    while True:
        rng.shuffle(perm)
        if not np.any(perm == idx):
            return perm


def per_stratum_wealth(
    W_i: np.ndarray,
    W_j: np.ndarray,
    p_i: np.ndarray,
    p_j: np.ndarray,
    strata: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> dict:
    """
    For each stratum k in {0,..,K-1}, compute an independent sub-e-process
    using the Method A construction (cross-item within stratum, delta = 2*eps_k).

    Per-stratum wealth process E^(k) is computed from item indices in stratum k.
    The items are presented in the order they appear in the stream — this preserves
    the anytime-valid property since the e-process is adapted to the filtration.

    Returns:
        per_stratum_M:  list of K wealth arrays (each of variable length)
        per_stratum_eps: array of K epsilon values
    """
    eps_k = stratum_epsilon(p_i, p_j, strata, Q=K)   # shape (K,)

    per_stratum_M = []

    for k in range(K):
        mask = np.where(strata == k)[0]   # item indices in stratum k (in stream order)
        n_k  = len(mask)

        if n_k <= 1:
            # Degenerate stratum: E^(k) stays at 1 (no updates possible)
            per_stratum_M.append(np.ones(max(n_k, 1)))
            continue

        W_i_k = W_i[mask]
        W_j_k = W_j[mask]
        p_i_k = p_i[mask]
        p_j_k = p_j[mask]

        # Derangement within this stratum
        local_idx  = np.arange(n_k)
        local_perm = local_idx.copy()
        while True:
            rng.shuffle(local_perm)
            if not np.any(local_perm == local_idx):
                break

        U   = W_i_k * W_j_k
        V   = W_i_k * W_j_k[local_perm]   # i on item s, j on different item t(s)

        eps_item = np.full(n_k, eps_k[k])
        delta    = 2.0 * eps_item

        # Use the mixture lambda (Robbins-style) for robustness, matching Method A
        M_k = accumulate_wealth_mixture(U, V, delta, clip_lam=True, eps_per_item=eps_item)
        per_stratum_M.append(M_k)

    return dict(per_stratum_M=per_stratum_M, eps_k=eps_k)


def averaged_portfolio_wealth(
    W_i: np.ndarray,
    W_j: np.ndarray,
    p_i: np.ndarray,
    p_j: np.ndarray,
    strata: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Construct the averaged portfolio e-process over K strata.

    Each stratum k yields a wealth process E^(k)_1, ..., E^(k)_{n_k} where n_k
    is the number of items in stratum k.  We interleave these back into the
    original item order to form the full-stream averaged process:

      E_t = (1/K) * sum_k E^(k)_{count_k(t)}

    where count_k(t) is the number of items up to time t that belong to stratum k.
    At time t, only strata with at least one item seen contribute their latest wealth;
    unseen strata are initialized at 1.

    This interleaving preserves the e-process property: at each time t, the
    averaged process is a convex combination of the individual sub-e-processes
    evaluated at their own local clock, each of which is a supermartingale <= 1
    under H0 given the filtration of stratum-k items seen so far.

    Returns: E of shape (N,) — the averaged portfolio wealth after each item.
    """
    res     = per_stratum_wealth(W_i, W_j, p_i, p_j, strata, K, rng)
    sub_M   = res['per_stratum_M']    # list of K arrays

    N_total = len(W_i)
    E       = np.ones(N_total)

    # Pointers into each per-stratum wealth array
    ptr = np.zeros(K, dtype=int)
    # Current wealth of each stratum (initialized at 1.0)
    current_W = np.ones(K)

    for t in range(N_total):
        k_t = strata[t]
        if ptr[k_t] < len(sub_M[k_t]):
            current_W[k_t] = sub_M[k_t][ptr[k_t]]
            ptr[k_t] += 1
        # Averaged portfolio value at time t
        E[t] = current_W.mean()

    return E


# ── Strata assignment helpers ─────────────────────────────────────────────────────

def true_strata(data: dict, K: int) -> np.ndarray:
    """Strata from the true (oracle) theta values."""
    return assign_strata(data['theta'], Q=K)


def estimated_strata(data: dict, K: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Strata from a noisy estimate of difficulty: d_hat = theta + N(0, sigma), clipped to [0,1].
    NOTE: epsilon is still computed from TRUE p_i,p_j within these noisy bins.
    This is a semi-oracle — the bins are wrong but the within-bin range is true.
    Typically inflates epsilon (more conservative), so Type-I control is maintained.
    """
    theta_hat = np.clip(data['theta'] + rng.normal(0.0, sigma, size=len(data['theta'])), 0.0, 1.0)
    return assign_strata(theta_hat, Q=K)


# ── Hard plug-in: estimated epsilon from empirical W values ───────────────────────

def stratum_epsilon_empirical(
    W_i: np.ndarray,
    W_j: np.ndarray,
    strata: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Estimate epsilon per stratum from the EMPIRICAL win rates of W_i, W_j
    rather than the true probabilities p_i, p_j.

    Within each stratum k: compute running sample means of W_i and W_j
    and use (max-min)/2 of those running means as the epsilon estimate.
    In practice, use the full-stratum sample mean range as an approximation.

    WARNING: empirical estimates are noisy, especially for small strata.
    Under-estimation is possible, which could make E[e_s] > 1, breaking
    the e-process supermartingale property. This is the plug-in danger zone.
    """
    eps = np.zeros(K)
    for k in range(K):
        mask = strata == k
        n_k  = mask.sum()
        if n_k < 2:
            eps[k] = 0.0
            continue
        # Within the stratum, estimate local win rates in two halves
        # to get a within-stratum range estimate from data alone.
        idx_k = np.where(mask)[0]
        half  = max(1, n_k // 2)
        first_half  = idx_k[:half]
        second_half = idx_k[half:]
        # Range of empirical means between first and second half
        range_i = abs(W_i[first_half].mean() - W_i[second_half].mean()) / 2.0
        range_j = abs(W_j[first_half].mean() - W_j[second_half].mean()) / 2.0
        eps[k]  = max(range_i, range_j)
    return eps


def per_stratum_wealth_hard_plugin(
    W_i: np.ndarray,
    W_j: np.ndarray,
    strata: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> list:
    """
    Like per_stratum_wealth but uses empirically estimated epsilon (hard plug-in).
    No access to true p_i, p_j — delta is set from data-estimated eps.
    This is the genuine plug-in case: may under-estimate delta, risking Type-I failure.
    """
    eps_k = stratum_epsilon_empirical(W_i, W_j, strata, K)   # from data, not truth

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

        # Hard plug-in: don't assert non-negativity — we WANT to see if it breaks
        # We suppress the assertion by clipping lambda conservatively.
        # But note: if eps_k is underestimated, delta may be too small, meaning
        # the increment 1+lam*(U-V-delta) could have E > 1, invalidating the e-process.
        M_k = accumulate_wealth_mixture(U, V, delta, clip_lam=True, eps_per_item=eps_item)
        per_stratum_M.append(M_k)

    return per_stratum_M


def averaged_portfolio_wealth_hard_plugin(
    W_i: np.ndarray,
    W_j: np.ndarray,
    strata: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Averaged portfolio with HARD PLUG-IN epsilon (estimated from W, not true p).
    """
    sub_M   = per_stratum_wealth_hard_plugin(W_i, W_j, strata, K, rng)
    N_total = len(W_i)
    E       = np.ones(N_total)
    ptr     = np.zeros(K, dtype=int)
    current_W = np.ones(K)

    for t in range(N_total):
        k_t = strata[t]
        if ptr[k_t] < len(sub_M[k_t]):
            current_W[k_t] = sub_M[k_t][ptr[k_t]]
            ptr[k_t] += 1
        E[t] = current_W.mean()

    return E


# ── Monte-Carlo runner ────────────────────────────────────────────────────────────

def run_condition_stratified(
    dgp_fn,
    K: int,
    strata_fn,   # callable(data, K, rng) -> strata array
    R: int,
    N: int,
    seed: int,
) -> dict:
    """
    Run R independent simulations using the SEMI-ORACLE averaged portfolio
    (epsilon computed from true p_i,p_j within possibly-noisy strata).
    Returns reject array (bool, length R) and detection times.
    """
    rng          = np.random.default_rng(seed)
    reject       = []
    det_times    = []

    for _ in range(R):
        data    = dgp_fn(N, rng)
        strata  = strata_fn(data, K, rng)

        E = averaged_portfolio_wealth(
            data['W_i'], data['W_j'],
            data['p_i'], data['p_j'],
            strata, K, rng,
        )

        fired, t_fire = ville_reject(E)
        reject.append(fired)
        det_times.append(t_fire + 1 if t_fire is not None else None)

    return dict(
        reject    = np.array(reject),
        det_times = det_times,
    )


def run_condition_hard_plugin(
    dgp_fn,
    K: int,
    strata_fn,   # callable(data, K, rng) -> strata array
    R: int,
    N: int,
    seed: int,
) -> dict:
    """
    Run R independent simulations using the HARD PLUG-IN averaged portfolio:
    epsilon is estimated from empirical W_i, W_j values (no access to true p).
    This tests whether estimating delta from data breaks Type-I control.
    """
    rng          = np.random.default_rng(seed)
    reject       = []
    det_times    = []

    for _ in range(R):
        data   = dgp_fn(N, rng)
        strata = strata_fn(data, K, rng)

        E = averaged_portfolio_wealth_hard_plugin(
            data['W_i'], data['W_j'],
            strata, K, rng,
        )

        fired, t_fire = ville_reject(E)
        reject.append(fired)
        det_times.append(t_fire + 1 if t_fire is not None else None)

    return dict(
        reject    = np.array(reject),
        det_times = det_times,
    )


def median_det(det_list: list) -> str:
    fired = [t for t in det_list if t is not None]
    if not fired:
        return "—"
    return str(int(np.median(fired)))


# ── Main experiment ────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("DIFFICULTY-STRATIFIED AVERAGED PORTFOLIO E-PROCESS EXPERIMENT")
    print(f"  N={N}, R={R}, alpha={ALPHA}, threshold={THRESHOLD}, SEED={SEED}")
    print("=" * 90)
    print()

    # DGP factories
    dgp_null      = null_dgp
    dgp_drift     = drift_null_dgp
    dgp_alt010    = lambda N, rng: alt_dgp(N, 0.10, rng)

    # Strata factories
    true_strata_fn  = lambda data, K, rng: true_strata(data, K)
    noisy_strata_fn = lambda data, K, rng: estimated_strata(data, K, NOISE_SIGMA, rng)

    # Condition seeds (offset from SEED to avoid overlap with run_experiment.py seeds)
    seeds = {
        'a_static_null':    SEED + 100,
        'b_drift_null':     SEED + 101,
        'c_noisy_null':     SEED + 102,
        'cp_hard_plugin':   SEED + 105,
        'd_drift_noisy':    SEED + 103,
        'e_alt010':         SEED + 104,
    }

    conditions = [
        # (label, dgp_fn, strata_fn, seed_key, metric_label, runner)
        # runner: 'standard' uses run_condition_stratified (semi-oracle eps)
        #         'plugin'   uses run_condition_hard_plugin (fully data-estimated eps)
        ('(a) static H0, true strata',     dgp_null,   true_strata_fn,  'a_static_null',  'Type-I', 'standard'),
        ('(b) drift  H0, true strata',     dgp_drift,  true_strata_fn,  'b_drift_null',   'Type-I', 'standard'),
        ('(c) static H0, noisy strata',    dgp_null,   noisy_strata_fn, 'c_noisy_null',   'Type-I', 'standard'),
        ("(c') static H0, hard plug-in",   dgp_null,   noisy_strata_fn, 'cp_hard_plugin', 'Type-I', 'plugin'),
        ('(d) drift  H0, noisy strata',    dgp_drift,  noisy_strata_fn, 'd_drift_noisy',  'Type-I', 'standard'),
        ('(e) alt H1 gamma=0.10, true str',dgp_alt010, true_strata_fn,  'e_alt010',       'Power ', 'standard'),
    ]

    all_rows = []   # for CSV

    print(f"{'Condition':<44} {'K':>3}  {'Rate':>8}  {'MedDet':>8}  {'Status'}")
    print("-" * 95)

    for (label, dgp_fn, strata_fn, seed_key, metric_label, runner) in conditions:
        print()
        for K in K_GRID:
            if runner == 'standard':
                res = run_condition_stratified(dgp_fn, K, strata_fn, R, N, seeds[seed_key])
            else:
                res = run_condition_hard_plugin(dgp_fn, K, strata_fn, R, N, seeds[seed_key])

            rate    = res['reject'].mean()
            med_det = median_det(res['det_times'])

            # Flag violations clearly
            is_typeI = (metric_label.strip() == 'Type-I')
            if is_typeI and rate > ALPHA:
                status = f"*** TYPE-I VIOLATION ({rate:.3f} > {ALPHA}) ***"
            elif is_typeI:
                status = "OK"
            else:
                status = f"power={rate:.3f}"

            print(
                f"  {label:<44} K={K:>2}  "
                f"{metric_label}={rate:.3f}  MedDet={med_det:>8}  {status}"
            )

            all_rows.append(dict(
                condition   = label,
                K           = K,
                metric      = metric_label.strip(),
                rate        = f"{rate:.4f}",
                med_det     = med_det,
                status      = status,
            ))

    print()
    print("=" * 95)
    print("SUMMARY — False-rejection rates vs 0.05 threshold")
    print("=" * 95)
    print("  (c) noisy strata: epsilon from TRUE p within noisy bins — conservatively inflated.")
    print("  (c') hard plug-in: epsilon from empirical W within noisy bins — may underestimate.")
    print()

    # Print compact summary table
    header = f"  {'Condition':<44} | {'K=2':>6} | {'K=5':>6} | {'K=10':>6}"
    print(header)
    print("  " + "-" * 72)

    seen_labels = {}
    for row in all_rows:
        key = row['condition']
        if key not in seen_labels:
            seen_labels[key] = {}
        seen_labels[key][row['K']] = float(row['rate'])

    for (label, *rest) in conditions:
        r2  = seen_labels[label].get(2,  float('nan'))
        r5  = seen_labels[label].get(5,  float('nan'))
        r10 = seen_labels[label].get(10, float('nan'))

        is_typeI = rest[2].strip() == 'Type-I'

        def fmt(v, is_t1):
            if is_t1 and v > ALPHA:
                return f"*{v:.3f}*"
            return f"{v:.3f}"

        print(f"  {label:<44} | {fmt(r2, is_typeI):>6} | {fmt(r5, is_typeI):>6} | {fmt(r10, is_typeI):>6}")

    print()
    print("  Note: * ... * marks Type-I violations > 0.05")
    print()

    # ── Write CSV ──────────────────────────────────────────────────────────────────
    csv_path = '/home/lyra/projects/evalue-sheaf/code/evalue/results_stratified.csv'
    fieldnames = ['condition', 'K', 'metric', 'rate', 'med_det', 'status']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"CSV written to {csv_path}")
    print()
    print("Done.")


if __name__ == '__main__':
    main()
