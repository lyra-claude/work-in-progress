"""
verify_prop3_freshpair.py — ACCEPTANCE TEST for Clio's Prop. 3 fresh-pair repair.

Bug (Prop. 2, verified): the PAST-ONLY pairing in make_online_past_pairing sets the
partner t(s) to an EARLIER same-stratum item, so W_j[t(s)] is F_{s-1}-measurable. On
the observable event {W_j[t(s)]=0} the increment U - V - delta has positive conditional
mean, and a predictable bettor that bets the whole cap on strata s>=5 whose partners all
have W_j(t)=0 drives Type-I to 1.000 under the STATIC null.

Fix (Prop. 3, fresh-pair within-block): pair each item with another SAME-stratum item in
the SAME block. Neither partner's outcome is in F_{m-1} when the block bet is set, so
E[X | F_{m-1}] <= 0 for EVERY predictable lambda. delta is computed per (block, stratum)
as 2 * within-block-within-stratum p-range.

This script uses the SAME machinery (build_increment_stream + the joint linear bet
wealth_path) so the test is honest. cap = 1/b as deployed.
"""

import sys
import numpy as np

sys.path.insert(0, '/home/lyra/projects/evalue-sheaf/code/evalue')

from dgp import null_dgp, alt_dgp, drift_null_dgp
from eprocess import (
    assign_strata, THRESHOLD,
    make_online_past_pairing, make_within_block_pairing,
)
import run_online_weights as rw
from run_online_weights import (
    build_increment_stream, lam_fixed, ons_weights, oracle_weights, wealth_path,
)
from run_equicorr_puzzle import latent_factor_null_dgp

# Settings matched to the §5b harness
Q      = 10
BLOCK  = 20
N      = 4000
R      = 1000
ALPHA  = 0.05
THRESH = THRESHOLD   # 20.0
GAMMA_GRID = [0.02, 0.04, 0.06, 0.10, 0.15]

rw.Q = Q
rw.BLOCK = BLOCK


# ── Clio's adversarial predictable bettor ──────────────────────────────────────
# For each block m, set lambda_{s,m} = cap for strata s >= 5 whose block partners all
# have W_j(t) = 0, else 0. Under past-only the partner value is F_{m-1}-measurable, so
# this rule exploits the observable {W_j[t]=0} event to win under H0 (the bug). Under
# fresh-pair the partner is in the SAME block, so the same rule cannot win: the within-
# block pairing makes E[X | F_{m-1}] <= 0 regardless.
def adversarial_wealth(data, rng, pairing):
    """
    Build the increment stream for `pairing`, then run the adversarial bettor on it.
    Returns (rejected, terminal_logwealth). cap = 1/b.
    """
    W_j   = data['W_j']
    theta = data['theta']
    strata = assign_strata(theta, Q=Q)

    # Build X and get the EXACT partner map t used to generate it (honest: same stream).
    X, b, t, has_partner = build_increment_stream(data, rng, pairing=pairing, return_t=True)
    cap = 1.0 / b
    Wjt = W_j[t]  # partner outcome per item

    M = X.shape[0]
    lam_seq = np.zeros((M, Q))
    for m in range(M):
        sl = slice(m * BLOCK, (m + 1) * BLOCK)
        s_blk = strata[sl]
        hp_blk = has_partner[sl]
        wjt_blk = Wjt[sl]
        for s in range(5, Q):             # strata s >= 5
            cell = (s_blk == s) & hp_blk
            if cell.any() and np.all(wjt_blk[cell] == 0):
                lam_seq[m, s] = cap
    # enforce feasibility sum_s lambda <= cap: if several strata fire in a block, scale.
    row_sums = lam_seq.sum(axis=1, keepdims=True)
    scale = np.where(row_sums > cap, cap / np.maximum(row_sums, 1e-12), 1.0)
    lam_seq = lam_seq * scale

    Mp = wealth_path(X, lam_seq)
    rej = bool(np.any(Mp >= THRESH))
    return rej, float(np.log(max(Mp[-1], 1e-300)))


def run_adversarial(dgp_fn, pairing, R, seed0=1000):
    rej = 0
    lws = []
    for r in range(R):
        rng = np.random.default_rng(seed0 + r)
        data = dgp_fn(N, rng)
        rr, lw = adversarial_wealth(data, rng, pairing)
        rej += int(rr)
        lws.append(lw)
    return rej / R, float(np.mean(lws))


# ── Standard FIXED / ONS Type-I and power on a stream ───────────────────────────
def run_standard(dgp_fn, R, seed0=1000, pairing='within_block'):
    """Returns dict of Type-I/power + mean log-wealth for FIXED and ONS."""
    rej_fx = rej_on = 0
    lw_fx = []; lw_on = []
    empty_cells = []
    for r in range(R):
        rng = np.random.default_rng(seed0 + r)
        data = dgp_fn(N, rng)
        X, b = build_increment_stream(data, rng, pairing=pairing)
        empty_cells.append(float(np.mean(X == 0.0)))
        lf = lam_fixed(b)
        lam_on_seq, _ = ons_weights(X, b)
        M_fx = wealth_path(X, lf)
        M_on = wealth_path(X, lam_on_seq)
        rej_fx += int(np.any(M_fx >= THRESH))
        rej_on += int(np.any(M_on >= THRESH))
        lw_fx.append(float(np.log(max(M_fx[-1], 1e-300))))
        lw_on.append(float(np.log(max(M_on[-1], 1e-300))))
    return dict(fixed=rej_fx / R, online=rej_on / R,
                lw_fixed=float(np.mean(lw_fx)), lw_online=float(np.mean(lw_on)),
                empty_frac=float(np.mean(empty_cells)))


def main():
    print("=" * 92)
    print("ACCEPTANCE TEST — Prop. 3 fresh-pair repair")
    print(f"  Q={Q}, N={N}, BLOCK={BLOCK} (M={N//BLOCK} blocks), R={R}, alpha={ALPHA}, cap=1/b")
    print("=" * 92)

    # (A) ADVERSARIAL bettor on the STATIC null, both designs.
    print("\n(A) ADVERSARIAL predictable bettor on STATIC null_dgp:")
    ti_past, lw_past = run_adversarial(null_dgp, 'past_only', R)
    print(f"    PAST-ONLY (buggy)  : Type-I = {ti_past:.3f}   mean logW = {lw_past:+.3f}   "
          f"(expect ~1.000, the bug)")
    ti_fresh, lw_fresh = run_adversarial(null_dgp, 'within_block', R)
    print(f"    FRESH-PAIR (Prop.3): Type-I = {ti_fresh:.3f}   mean logW = {lw_fresh:+.3f}   "
          f"(must be ~0.000)")
    if ti_fresh > ALPHA:
        print("    *** FRESH-PAIR ADVERSARIAL Type-I EXCEEDS alpha — FIX IS WRONG, STOP ***")

    # (B) Standard FIXED and ONS Type-I on three nulls, fresh-pair.
    print("\n(B) Standard Type-I (FIXED, ONS) on three nulls, FRESH-PAIR:")
    print(f"    {'null':>28} | {'FIXED':>7} {'ONS':>7} | {'lw_fx':>8} {'lw_on':>8} {'empty%':>7}")
    print("    " + "-" * 74)
    for name, fn in [("static null", null_dgp),
                     ("latent-factor equicorr", latent_factor_null_dgp),
                     ("drift_null", drift_null_dgp)]:
        res = run_standard(fn, R)
        print(f"    {name:>28} | {res['fixed']:>7.3f} {res['online']:>7.3f} | "
              f"{res['lw_fixed']:>+8.3f} {res['lw_online']:>+8.3f} {100*res['empty_frac']:>6.1f}%")

    # (C) Power: Regime 1 (homogeneous shock) and Regime 3 (drift) over gamma grid.
    #     Regime 1 = alt_dgp(gamma). Regime 3 "drift" power here = alt+drift is not defined;
    #     the prompt's "Regime 3 (drift)" power means re-running the drift-null AND the
    #     homogeneous alt under fresh-pair. Drift is a NULL (power = Type-I, already in B);
    #     we report Regime-1 power across gamma, and the drift-null rejection (should be ~0).
    print("\n(C) POWER — Regime 1 (homogeneous alt_dgp) across gamma, FRESH-PAIR:")
    print(f"    {'gamma':>6} | {'FIXED pow':>9} {'ONS pow':>8} | {'lw_fx':>8} {'lw_on':>8} {'empty%':>7}")
    print("    " + "-" * 62)
    for g in GAMMA_GRID:
        fn = (lambda Nn, rng, gg=g: alt_dgp(Nn, gg, rng))
        res = run_standard(fn, R)
        print(f"    {g:>6.2f} | {res['fixed']:>9.3f} {res['online']:>8.3f} | "
              f"{res['lw_fixed']:>+8.3f} {res['lw_online']:>+8.3f} {100*res['empty_frac']:>6.1f}%")

    print("\n    Regime 3 (drift NULL) rejection is reported in (B) as drift_null (must be ~0).")
    print("\nDone.")


if __name__ == '__main__':
    main()
