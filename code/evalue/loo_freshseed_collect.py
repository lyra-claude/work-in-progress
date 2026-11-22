"""
loo_freshseed_run90000.py — Fresh-seed reproduction for Clio's peer-review request.

Three-arm PAIRED power sweep: seed block 90000+r, R=2000.
Arms:
  (A) cyclic  + delta_2  (second_order slack for cyclic)      -> BASELINE
  (B) LOO     + delta_LOO (second_order slack for LOO, Prop.4) -> LOO-matched slack
  (C) LOO     + delta_2   (second_order slack for cyclic)     -> isolates pairing vs slack

PAIRED: same data draws used for all three arms per replication.
SEED base: 90000  (seed = 90000 + r for r in 0..R-1).

make_within_block_pairing is DETERMINISTIC (no rng): cyclic roll within cell.
LOO V is computed analytically from W_j cell, also deterministic given data.
=> All three arms get the same (W_i, W_j, theta, p_i, p_j) for each r.
=> Pairwise differences are valid Monte-Carlo estimates of power difference.

The delta_2 for cyclic:  delta_2 = (1/2) R_i R_j = 2 * range_i * range_j
The delta_LOO for LOO:   delta_LOO = (n/(n-1)) * range_i * range_j
The "LOO + delta_2" arm: use LOO pairing but set delta = 2 * range_i * range_j

Does NOT modify dgp.py / eprocess.py / run_online_weights.py.
"""

import sys
import numpy as np

sys.path.insert(0, '/home/lyra/projects/evalue-sheaf/code/evalue')

from dgp import null_dgp, alt_dgp
from eprocess import THRESHOLD, assign_strata
import run_online_weights as rw
from run_online_weights import lam_fixed, wealth_path

# ── Parameters ──────────────────────────────────────────────────────────────────
Q           = 10
BLOCK       = 20
N           = 4000
R           = 2000
ALPHA       = 0.05
THRESH      = THRESHOLD        # 20.0
SEED_BASE   = 90000
GAMMA_GRID  = [0.00, 0.04, 0.06, 0.08, 0.10]

# Patch module-level params
rw.Q     = Q
rw.BLOCK = BLOCK


def build_increment_three_arms(data: dict):
    """
    Build X, b for three arms from the SAME data dict.
    Returns: (X_cyc, b_cyc), (X_loo_dloo, b_loo_dloo), (X_loo_d2, b_loo_d2)

    We re-implement the within_block path inline (no rng needed: deterministic).
    Uses rw.Q and rw.BLOCK module-level values.
    """
    from eprocess import assign_strata, make_within_block_pairing

    W_i, W_j = data['W_i'], data['W_j']
    p_i, p_j = data['p_i'], data['p_j']
    theta    = data['theta']
    Nloc     = len(W_i)

    q_local = rw.Q
    blk_sz  = rw.BLOCK
    M = Nloc // blk_sz
    block_ids = np.arange(Nloc) // blk_sz

    strata = assign_strata(theta, Q=q_local)
    U = (W_i * W_j).astype(float)

    # Cyclic V  (pairing_rule='cyclic')
    t = make_within_block_pairing(strata, block_ids, Q=q_local)
    has_partner = (t != np.arange(Nloc))
    V_cyc = (W_i * W_j[t]).astype(float)

    # LOO V  (pairing_rule='loo')
    V_loo = np.zeros(Nloc)
    for m in range(M):
        sel_m = block_ids == m
        for k in range(q_local):
            cell_idx = np.where(sel_m & (strata == k))[0]
            n = len(cell_idx)
            if n < 2:
                continue
            wj = W_j[cell_idx].astype(float)
            sumj = wj.sum()
            loo_mean = (sumj - wj) / (n - 1)
            V_loo[cell_idx] = W_i[cell_idx].astype(float) * loo_mean

    # Delta arrays for the three arms
    # (A) cyclic + delta_2
    # (B) LOO    + delta_LOO = (n/(n-1)) * range_i * range_j
    # (C) LOO    + delta_2   = 2 * range_i * range_j
    delta_A = np.zeros(Nloc)   # cyclic + delta_2
    delta_B = np.zeros(Nloc)   # LOO + delta_LOO
    delta_C = np.zeros(Nloc)   # LOO + delta_2

    for m in range(M):
        sel_m = block_ids == m
        for k in range(q_local):
            cell = sel_m & (strata == k) & has_partner
            ncell = int(cell.sum())
            if ncell < 1:
                continue
            range_i = (p_i[cell].max() - p_i[cell].min()) / 2.0
            range_j = (p_j[cell].max() - p_j[cell].min()) / 2.0
            d2  = 2.0 * range_i * range_j            # (1/2) R_i R_j
            if ncell >= 2:
                d_loo = (ncell / (ncell - 1.0)) * range_i * range_j
            else:
                d_loo = d2  # degenerate fallback
            delta_A[cell] = d2
            delta_B[cell] = d_loo
            delta_C[cell] = d2   # LOO pairing but cyclic/d2 slack

    # Increment arrays
    inc_A = np.where(has_partner, U - V_cyc - delta_A, 0.0)
    inc_B = np.where(has_partner, U - V_loo - delta_B, 0.0)
    inc_C = np.where(has_partner, U - V_loo - delta_C, 0.0)
    valid = has_partner.astype(float)

    b_A = 1.0 + float(delta_A.max())
    b_B = 1.0 + float(delta_B.max())
    b_C = 1.0 + float(delta_C.max())

    def to_X(inc_arr):
        X = np.zeros((M, q_local))
        for m in range(M):
            sl = slice(m * blk_sz, (m + 1) * blk_sz)
            s_blk = strata[sl]
            inc_blk = inc_arr[sl]
            val_blk = valid[sl]
            for s in range(q_local):
                sel = (s_blk == s) & (val_blk > 0)
                if sel.any():
                    X[m, s] = inc_blk[sel].mean()
        return X

    X_A = to_X(inc_A)
    X_B = to_X(inc_B)
    X_C = to_X(inc_C)

    return (X_A, b_A), (X_B, b_B), (X_C, b_C)


def run_paired_sweep(gamma: float, R: int) -> dict:
    """
    Run R paired replications at given gamma.
    Returns dicts of rejection indicators for each arm.
    """
    rej_A = np.zeros(R, dtype=bool)
    rej_B = np.zeros(R, dtype=bool)
    rej_C = np.zeros(R, dtype=bool)

    for r in range(R):
        seed = SEED_BASE + r
        rng = np.random.default_rng(seed)
        if gamma == 0.0:
            data = null_dgp(N, rng)
        else:
            data = alt_dgp(N, gamma, rng)

        (X_A, b_A), (X_B, b_B), (X_C, b_C) = build_increment_three_arms(data)

        lf_A = lam_fixed(b_A)
        lf_B = lam_fixed(b_B)
        lf_C = lam_fixed(b_C)

        Mp_A = wealth_path(X_A, lf_A)
        Mp_B = wealth_path(X_B, lf_B)
        Mp_C = wealth_path(X_C, lf_C)

        rej_A[r] = np.any(Mp_A >= THRESH)
        rej_B[r] = np.any(Mp_B >= THRESH)
        rej_C[r] = np.any(Mp_C >= THRESH)

    pA = rej_A.mean()
    pB = rej_B.mean()
    pC = rej_C.mean()

    # Paired differences (B - A) and (C - A)
    diff_BA = rej_B.astype(float) - rej_A.astype(float)   # LOO_dLOO - cyclic_d2
    diff_CA = rej_C.astype(float) - rej_A.astype(float)   # LOO_d2   - cyclic_d2

    # SE of paired mean difference = std(diff) / sqrt(R)
    se_BA = diff_BA.std() / np.sqrt(R)
    se_CA = diff_CA.std() / np.sqrt(R)

    return dict(
        gamma=gamma,
        pA=pA, pB=pB, pC=pC,
        diff_BA=diff_BA.mean(), se_BA=se_BA,
        diff_CA=diff_CA.mean(), se_CA=se_CA,
    )


def ci95_half(p: float, R: int) -> float:
    return 1.96 * np.sqrt(p * (1.0 - p) / R)


def main():
    import time
    t0 = time.time()

    print("=" * 100)
    print("LOO vs CYCLIC fresh-seed reproduction — Clio peer-review request")
    print(f"  SEED_BASE={SEED_BASE}, seed=SEED_BASE+r for r=0..{R-1}")
    print(f"  R={R}, N={N}, Q={Q}, BLOCK={BLOCK}, alpha={ALPHA}, THRESH={THRESH}")
    print(f"  Arms:")
    print(f"    A = cyclic  + delta_2  (second_order, cyclic formula)")
    print(f"    B = LOO     + delta_LOO (second_order, LOO/Prop.4 formula = (n/(n-1))*range_i*range_j)")
    print(f"    C = LOO     + delta_2   (second_order, cyclic formula — isolates pairing from slack)")
    print(f"  PAIRED: same data per replication for all three arms.")
    print(f"  Gammas: {GAMMA_GRID}")
    print("=" * 100)

    all_res = {}
    for gamma in GAMMA_GRID:
        print(f"\n  Running gamma={gamma:.2f} ...", flush=True)
        res = run_paired_sweep(gamma, R)
        all_res[gamma] = res
        note_A = ""
        if gamma == 0.0:
            note_A = " <- TYPE-I" + (" *** EXCEEDS 0.05 ***" if res['pA'] > 0.05 else " OK")
        print(f"    A(cyclic+d2) = {res['pA']:.4f}  B(LOO+dLOO) = {res['pB']:.4f}  "
              f"C(LOO+d2) = {res['pC']:.4f}{note_A}")
        print(f"    B-A (LOO_dLOO - cyclic_d2) = {res['diff_BA']:+.4f} +/- {res['se_BA']:.4f}")
        print(f"    C-A (LOO_d2   - cyclic_d2) = {res['diff_CA']:+.4f} +/- {res['se_CA']:.4f}")

    elapsed = time.time() - t0

    # ── Summary table ────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (R={:d}, seed=90000+r, paired)".format(R))
    print(f"  {'gamma':>6} | {'A:cyc+d2':>10} | {'B:LOO+dLOO':>10} | {'C:LOO+d2':>10} | "
          f"{'B-A':>8} {'±SE':>6} | {'C-A':>8} {'±SE':>6}")
    print("  " + "-" * 85)
    for gamma in GAMMA_GRID:
        r = all_res[gamma]
        note = " <- TYPE-I" if gamma == 0.0 else ""
        print(f"  {gamma:>6.2f} | {r['pA']:>10.4f} | {r['pB']:>10.4f} | {r['pC']:>10.4f} | "
              f"{r['diff_BA']:>+8.4f} {r['se_BA']:>6.4f} | {r['diff_CA']:>+8.4f} {r['se_CA']:>6.4f}{note}")

    # ── Verdict ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("VERDICT — Clio's claimed edge: LOO+dLOO - cyclic+d2 ~ +0.028 +/- 0.007 at the edge gamma")
    print("=" * 100)

    # Find gamma with max |B-A| (excluding null)
    alt_gammas = [g for g in GAMMA_GRID if g > 0.0]
    max_g = max(alt_gammas, key=lambda g: abs(all_res[g]['diff_BA']))
    r_max = all_res[max_g]
    print(f"  Max |LOO_dLOO - cyclic_d2| is at gamma={max_g:.2f}: "
          f"{r_max['diff_BA']:+.4f} +/- {r_max['se_BA']:.4f}")

    # Check against 0.028 +/- 0.007
    target = 0.028
    target_se = 0.007
    actual = r_max['diff_BA']
    actual_se = r_max['se_BA']
    # Within ~2 SE of target?
    gap = abs(actual - target)
    combined_se = np.sqrt(target_se**2 + actual_se**2)
    reproduces = gap < 2.0 * combined_se
    print(f"  Target: {target:+.3f} +/- {target_se:.3f}")
    print(f"  Fresh:  {actual:+.4f} +/- {actual_se:.4f}")
    print(f"  |gap| = {gap:.4f}, combined_SE = {combined_se:.4f}, gap/combined_SE = {gap/combined_se:.2f}")
    print(f"  REPRODUCES (+0.028 +/- 0.007 edge): {'YES' if reproduces else 'NO'}")

    # Type-I check
    print("\n  Type-I sanity (gamma=0, all arms must be <= 0.05):")
    r0 = all_res[0.0]
    for arm, pval in [('A(cyclic+d2)', r0['pA']), ('B(LOO+dLOO)', r0['pB']), ('C(LOO+d2)', r0['pC'])]:
        status = "OK" if pval <= 0.05 else "*** VIOLATION ***"
        print(f"    {arm}: {pval:.4f} -> {status}")

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print("Done.")


if __name__ == '__main__':
    main()
