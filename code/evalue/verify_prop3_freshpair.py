"""
verify_prop3_freshpair.py — ACCEPTANCE TEST for Clio's Prop. 3 fresh-pair repair,
with Clio's PREDICTABLE adversarial bettors (2026-11-19 c2 review) and the
SECOND-ORDER slack (Prop. 2).

WHAT CHANGED (Clio's 2026-11-19 c2 review, UID 2174)
----------------------------------------------------
(1) The old adversarial bettor in this file was LOOK-AHEAD.  It set
      lam_{m,s} = cap  when  all( W_j[t(s)] == 0 )  over the block-m cell,
    and under WITHIN-BLOCK pairing t(s) is an item of the SAME block m, so
    Wjt = W_j[t] is NOT F_{m-1}-measurable: the bet reads the current block's
    outcomes.  Worse, because t permutes the cell, "every partner has W_j=0" is
    the same event as "every item of the cell has W_j=0", on which U = V = 0 and
    X = -delta exactly.  So the rule fired ONLY when X = -delta and was bound to
    lose (mean logW = -1.804, which merely says delta > 0).  It was no test of
    validity.  Table (A) built on it was no evidence.  (Clio, Prop. 1 / section 1.1.)

(2) Replacement: SIX genuinely PREDICTABLE in-budget adversaries (Clio's).  A
    predictable bettor may use theta for blocks <= m (strata are fixed in advance,
    so the live/empty pattern of block m is known before block m is observed) and
    OUTCOMES of blocks < m only — never the current block's W.  All stay <= alpha
    (the Ville bound).  'peek' is a look-ahead positive control that SHOULD reject
    in every run, proving the harness can see an invalid bettor.

(3) SECOND-ORDER slack (Prop. 2): delta_2 = (1/2) R_i R_j in place of
    delta_1 = max(R_i, R_j).  The within-cell pairing is a permutation of the cell,
    so the partner sum equals the cell's own sum; E[X|F_{m-1}] <= (1/2) R_i R_j - delta.
    delta_2 already makes E[X|F_{m-1}] <= 0 at no cost to validity, and restores
    Regime-1 FIXED power (delta_1 slack of O(R) swamped small gamma; delta_2 is O(R^2)).

VALIDITY IS PROVEN (Clio, Prop. 1): within-block cyclic pairing, per-cell
delta >= (1/2) R_i R_j, lambda >= 0 predictable, sum lambda <= 1/b, gives a test
supermartingale under H0.  This script is an HONEST empirical demonstration, not the
validity argument.
"""

import sys
import numpy as np

sys.path.insert(0, '/home/lyra/projects/evalue-sheaf/code/evalue')

from dgp import null_dgp, alt_dgp, drift_null_dgp
from eprocess import assign_strata, THRESHOLD
import run_online_weights as rw
from run_online_weights import build_increment_stream, lam_fixed, wealth_path
from run_equicorr_puzzle import latent_factor_null_dgp

# Settings matched to the §5b harness
Q      = 10
BLOCK  = 20
N      = 4000
R      = 1000        # nulls
R_POW  = 500         # power
ALPHA  = 0.05
THRESH = THRESHOLD   # 20.0
GAMMA_GRID = [0.02, 0.04, 0.06, 0.10, 0.15]

rw.Q = Q
rw.BLOCK = BLOCK
M_BLOCKS = N // BLOCK


# ── Per-(block, stratum) cell structure, all F_{m-1}-predictable except W-sums ──
def cell_structure(data):
    """
    Return, per block m and stratum k:
        live[m,k]  : bool, cell has n >= 2 (a bet is placed there)       [PREDICTABLE: theta only]
        ncell[m,k] : int, number of items in the cell                    [PREDICTABLE: theta only]
    plus arrays needed by specific bettors (see below).  `theta`/strata fix the
    live/empty pattern in advance, so `live` and `ncell` are F_{m-1}-measurable.
    """
    theta  = data['theta']
    W_j    = data['W_j']
    strata = assign_strata(theta, Q=Q)
    block_ids = np.arange(N) // BLOCK

    live  = np.zeros((M_BLOCKS, Q), dtype=bool)
    ncell = np.zeros((M_BLOCKS, Q), dtype=int)
    # per-cell sum of W_j, used by 'replay' but only ever READ for blocks < m.
    wj_zero = np.zeros((M_BLOCKS, Q), dtype=bool)   # True iff cell live AND all W_j == 0
    for m in range(M_BLOCKS):
        sl = slice(m * BLOCK, (m + 1) * BLOCK)
        s_blk = strata[sl]
        wj_blk = W_j[sl]
        for k in range(Q):
            idx = np.where(s_blk == k)[0]
            n = len(idx)
            ncell[m, k] = n
            if n >= 2:
                live[m, k] = True
                wj_zero[m, k] = bool(np.all(wj_blk[idx] == 0))
    return strata, live, ncell, wj_zero


# ── Clio's six predictable bettors + the peek positive control ──────────────────
def build_lambda(bettor, X, b, live, ncell, wj_zero, delta_cell):
    """
    Return lam_seq (M, Q), predictable except for 'peek' (the look-ahead control).

    delta_cell[m,k] : per-cell slack (for 'min-slack'); 0 for dead cells.
    All non-peek bettors use ONLY: live/ncell/delta_cell (theta-determined, so known
    in advance) and, for data-adaptive ones, X and wj_zero from blocks STRICTLY < m.
    """
    M, Qn = X.shape
    cap = 1.0 / b
    lam = np.zeros((M, Qn))

    if bettor == 'fixed':
        # cap/Q on every stratum, constant. (Clio: 'fixed'.)
        lam[:, :] = cap / Qn

    elif bettor == 'fixed-live':
        # spread cap over the LIVE cells of block m, so no money on empty cells.
        for m in range(M):
            live_k = np.where(live[m])[0]
            if len(live_k):
                lam[m, live_k] = cap / len(live_k)

    elif bettor == 'max-n':
        # whole cap on the LARGEST live cell of block m.
        for m in range(M):
            live_k = np.where(live[m])[0]
            if len(live_k):
                kstar = live_k[np.argmax(ncell[m, live_k])]
                lam[m, kstar] = cap

    elif bettor == 'min-slack':
        # whole cap on the live cell with the SMALLEST delta — the fairest game.
        for m in range(M):
            live_k = np.where(live[m])[0]
            if len(live_k):
                kstar = live_k[np.argmin(delta_cell[m, live_k])]
                lam[m, kstar] = cap

    elif bettor == 'replay':
        # whole cap on strata >= 5 whose MOST RECENT PAST live cell had every W_j = 0.
        # This is the 11-18 attack made PREDICTABLE: uses wj_zero[<m] only.
        last_zero = np.zeros(Qn, dtype=bool)   # state carried forward from past blocks
        for m in range(M):
            # bet using state from blocks < m:
            for k in range(5, Qn):
                if live[m, k] and last_zero[k]:
                    lam[m, k] = cap
            # normalize if several strata fire (keep sum <= cap)
            s = lam[m].sum()
            if s > cap:
                lam[m] *= cap / s
            # AFTER betting, update state with block m's OWN observation (for block m+1)
            for k in range(Qn):
                if live[m, k]:
                    last_zero[k] = wj_zero[m, k]

    elif bettor == 'momentum':
        # whole cap on the live stratum with the largest PAST sum of X (blocks < m).
        cumX = np.zeros(Qn)
        for m in range(M):
            live_k = np.where(live[m])[0]
            if len(live_k):
                kstar = live_k[np.argmax(cumX[live_k])]
                lam[m, kstar] = cap
            cumX += X[m]   # update AFTER betting => uses blocks < m only

    elif bettor == 'peek':
        # LOOK-AHEAD positive control: bet proportional to 1[X_{m,k} > 0]. NOT valid.
        for m in range(M):
            pos = (X[m] > 0).astype(float)
            s = pos.sum()
            if s > 0:
                lam[m] = cap * pos / s
    else:
        raise ValueError(f"unknown bettor {bettor!r}")

    return lam


BETTORS = ['fixed', 'fixed-live', 'max-n', 'min-slack', 'replay', 'momentum', 'peek']


def per_cell_delta(data, slack, pairing_rule):
    """Recover the per-(block,stratum) delta used by build_increment_stream, for min-slack."""
    from run_online_weights import Q as RWQ  # noqa
    theta = data['theta']; p_i = data['p_i']; p_j = data['p_j']
    strata = assign_strata(theta, Q=Q)
    block_ids = np.arange(N) // BLOCK
    dc = np.zeros((M_BLOCKS, Q))
    for m in range(M_BLOCKS):
        sel_m = block_ids == m
        for k in range(Q):
            idx = np.where(sel_m & (strata == k))[0]
            n = len(idx)
            if n < 2:
                continue
            ri = (p_i[idx].max() - p_i[idx].min()) / 2.0
            rj = (p_j[idx].max() - p_j[idx].min()) / 2.0
            if slack == 'first_order':
                dc[m, k] = 2.0 * max(ri, rj)
            elif pairing_rule == 'loo':
                dc[m, k] = (n / (n - 1.0)) * ri * rj
            else:
                dc[m, k] = 2.0 * ri * rj
    return dc


def run_bettors(dgp_fn, R, slack, pairing_rule, seed0=1000):
    """Return dict bettor -> Type-I rejection fraction."""
    rej = {bt: 0 for bt in BETTORS}
    for r in range(R):
        rng = np.random.default_rng(seed0 + r)
        data = dgp_fn(N, rng)
        X, b = build_increment_stream(data, rng, pairing='within_block',
                                      slack=slack, pairing_rule=pairing_rule)
        strata, live, ncell, wj_zero = cell_structure(data)
        delta_cell = per_cell_delta(data, slack, pairing_rule)
        for bt in BETTORS:
            lam = build_lambda(bt, X, b, live, ncell, wj_zero, delta_cell)
            Mp = wealth_path(X, lam)
            if np.any(Mp >= THRESH):
                rej[bt] += 1
    return {bt: rej[bt] / R for bt in BETTORS}


# ── Standard FIXED power on Regime-1 alt, by slack / pairing_rule ────────────────
def run_power_fixed(gamma, R, slack, pairing_rule, seed0=1000):
    rej = 0
    for r in range(R):
        rng = np.random.default_rng(seed0 + r)
        data = alt_dgp(N, gamma, rng)
        X, b = build_increment_stream(data, rng, pairing='within_block',
                                      slack=slack, pairing_rule=pairing_rule)
        lf = lam_fixed(b)
        Mp = wealth_path(X, lf)
        if np.any(Mp >= THRESH):
            rej += 1
    return rej / R


def main():
    print("=" * 100)
    print("ACCEPTANCE TEST — Prop. 3 fresh-pair repair (Clio 2026-11-19 c2 review)")
    print(f"  Q={Q}, N={N}, BLOCK={BLOCK} (M={M_BLOCKS} blocks), R_null={R}, R_pow={R_POW}, "
          f"alpha={ALPHA}, cap=1/b")
    print("=" * 100)

    # ── (A) PREDICTABLE adversarial bettors, Type-I, three nulls × (delta1, delta2) ──
    print("\n(A) PREDICTABLE adversarial bettors — Type-I (must be <= alpha=0.05).")
    print("    peek is a LOOK-AHEAD positive control: it SHOULD reject ~1.000.")
    print("    OLD look-ahead bettor (removed): fired only on X=-delta, mean logW=-1.804 — no test.\n")
    hdr = f"    {'null / slack':>28} | " + " ".join(f"{bt:>10}" for bt in BETTORS)
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    nulls = [("static", null_dgp),
             ("latent", latent_factor_null_dgp),
             ("drift",  drift_null_dgp)]
    for name, fn in nulls:
        for slack, tag in [('first_order', 'd1'), ('second_order', 'd2')]:
            res = run_bettors(fn, R, slack, 'cyclic')
            label = f"{name}, {tag} cyclic"
            print(f"    {label:>28} | " + " ".join(f"{res[bt]:>10.3f}" for bt in BETTORS))
        # LOO row under d2
        res = run_bettors(fn, R, 'second_order', 'loo')
        label = f"{name}, d2 LOO"
        print(f"    {label:>28} | " + " ".join(f"{res[bt]:>10.3f}" for bt in BETTORS))

    # ── (B) Power recovery: Regime-1 FIXED power, delta1 vs delta2 (cyclic) + d2 LOO ──
    print("\n(B) POWER — Regime 1 (homogeneous alt_dgp), FIXED weights, cap=1/b.")
    print("    Clio's target: delta2 restores FIXED power at gamma=0.06 from 0.000 to ~0.53.\n")
    print(f"    {'gamma':>6} | {'d1 cyclic':>10} {'d2 cyclic':>10} {'d2 LOO':>10}")
    print("    " + "-" * 44)
    for g in GAMMA_GRID:
        p1 = run_power_fixed(g, R_POW, 'first_order', 'cyclic')
        p2 = run_power_fixed(g, R_POW, 'second_order', 'cyclic')
        pl = run_power_fixed(g, R_POW, 'second_order', 'loo')
        print(f"    {g:>6.2f} | {p1:>10.3f} {p2:>10.3f} {pl:>10.3f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
