"""
loo_vs_cyclic_sweep.py — Throwaway measurement script.

Runs LOO vs cyclic within-block pairing at R=10000, delta_2 (second_order slack),
across gamma in {0.00, 0.04, 0.06, 0.08, 0.10} with a fixed RNG seed.

Reports power (rejection rate at alpha=0.05) and 95% Monte-Carlo CIs.
Gamma=0.00 is the static-null Type-I check.

Does NOT edit dgp.py / eprocess.py / verify_prop3_freshpair.py.
Calls build_increment_stream and lam_fixed / wealth_path from run_online_weights.
"""

import sys
import numpy as np

sys.path.insert(0, '/home/lyra/projects/evalue-sheaf/code/evalue')

from dgp import null_dgp, alt_dgp
from eprocess import THRESHOLD, assign_strata
import run_online_weights as rw
from run_online_weights import build_increment_stream, lam_fixed, wealth_path

# ── Parameters ────────────────────────────────────────────────────────────────
Q           = 10
BLOCK       = 20
N           = 4000
R           = 5000
ALPHA       = 0.05
THRESH      = THRESHOLD        # 20.0  (= 1/0.05)
SEED        = 20261120         # fixed seed — recorded in output below
GAMMA_GRID  = [0.00, 0.04, 0.06, 0.08, 0.10]
SLACK       = 'second_order'   # delta_2 = R_i * R_j / 2

# Patch module-level params that build_increment_stream reads
rw.Q     = Q
rw.BLOCK = BLOCK


def run_scheme(scheme: str, gamma: float, R: int, seed: int) -> float:
    """
    Returns rejection rate (power or Type-I) at THRESH for a given scheme and gamma.
    scheme: 'cyclic' or 'loo'
    gamma=0.0 -> null_dgp (Type-I check), else alt_dgp.
    """
    rej = 0
    for r in range(R):
        rng = np.random.default_rng(seed + r)
        if gamma == 0.0:
            data = null_dgp(N, rng)
        else:
            data = alt_dgp(N, gamma, rng)
        X, b = build_increment_stream(data, rng,
                                      pairing='within_block',
                                      slack=SLACK,
                                      pairing_rule=scheme)
        lf = lam_fixed(b)
        Mp = wealth_path(X, lf)
        if np.any(Mp >= THRESH):
            rej += 1
    return rej / R


def ci95(p: float, R: int) -> float:
    """Half-width of 95% Monte-Carlo CI."""
    return 1.96 * np.sqrt(p * (1.0 - p) / R)


def main():
    print("=" * 90)
    print("LOO vs CYCLIC within-block pairing — power sweep at R=10000")
    print(f"  Seed={SEED}, slack={SLACK} (delta_2=R_i*R_j/2), alpha={ALPHA}, THRESH={THRESH}")
    print(f"  Q={Q}, N={N}, BLOCK={BLOCK} (M={N//BLOCK} blocks), R={R}")
    print(f"  pairing_rule='cyclic'  -> run_online_weights.build_increment_stream(..., pairing_rule='cyclic')")
    print(f"  pairing_rule='loo'     -> run_online_weights.build_increment_stream(..., pairing_rule='loo')")
    print("=" * 90)

    # Header
    print(f"\n{'scheme':>8} | {'gamma':>6} | {'power':>7} | {'95% CI':>18} | note")
    print("-" * 60)

    results = {}
    for scheme in ['cyclic', 'loo']:
        for gamma in GAMMA_GRID:
            p = run_scheme(scheme, gamma, R, SEED)
            hw = ci95(p, R)
            lo = max(0.0, p - hw)
            hi = min(1.0, p + hw)
            note = ""
            if gamma == 0.0:
                note = "TYPE-I" + (" *** EXCEEDS 0.05 — VALIDITY PROBLEM ***" if p > 0.05 else " OK")
            results[(scheme, gamma)] = (p, lo, hi)
            print(f"  {scheme:>6} | {gamma:>6.2f} | {p:>7.4f} | [{lo:.4f}, {hi:.4f}] | {note}")

    # Summary analysis
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Type-I check
    for scheme in ['cyclic', 'loo']:
        p_null, lo_null, hi_null = results[(scheme, 0.0)]
        status = "OK (<=0.05)" if p_null <= 0.05 else "*** EXCEEDS 0.05 — VALIDITY PROBLEM ***"
        print(f"  Type-I [{scheme}]: {p_null:.4f}  CI=[{lo_null:.4f},{hi_null:.4f}]  -> {status}")

    # Key comparison at gamma=0.06
    p_cyc, lo_cyc, hi_cyc = results[('cyclic', 0.06)]
    p_loo, lo_loo, hi_loo = results[('loo', 0.06)]
    overlap = lo_loo < hi_cyc and lo_cyc < hi_loo
    diff = p_loo - p_cyc
    print(f"\n  gamma=0.06:  cyclic={p_cyc:.4f} CI=[{lo_cyc:.4f},{hi_cyc:.4f}]")
    print(f"               loo   ={p_loo:.4f} CI=[{lo_loo:.4f},{hi_loo:.4f}]")
    print(f"               LOO - cyclic = {diff:+.4f}")
    print(f"               CI OVERLAP: {'YES (edge is noise)' if overlap else 'NO (clear separation)'}")

    # Sign consistency across gamma > 0
    pos_gammas = [g for g in GAMMA_GRID if g > 0.0]
    diffs = [(g, results[('loo', g)][0] - results[('cyclic', g)][0]) for g in pos_gammas]
    signs = [d > 0 for _, d in diffs]
    print(f"\n  LOO - cyclic power difference by gamma:")
    for g, d in diffs:
        sign_str = "(LOO>cyclic)" if d > 0 else "(cyclic>LOO)"
        print(f"    gamma={g:.2f}: {d:+.4f}  {sign_str}")
    all_same_sign = all(signs) or not any(signs)
    print(f"\n  Consistent same-sign edge across gamma>0: {'YES' if all_same_sign else 'NO (sign flips)'}")

    print("\nDone.")


if __name__ == '__main__':
    main()
