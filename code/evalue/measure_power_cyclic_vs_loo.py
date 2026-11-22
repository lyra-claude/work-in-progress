"""
measure_power_cyclic_vs_loo.py — THROWAWAY measurement script (do NOT edit harness logic).

Runs both cyclic and LOO within-block pairing at R=10000 replications across
gamma in {0, 0.04, 0.06, 0.08, 0.10} with delta_2 = second_order slack.

Reports power (rejection rate at alpha=0.05) with 95% Monte-Carlo CI (binomial).
Gamma=0 is the Type-I sanity check.

SEED: 20261120  (chosen for reproducibility; date-stamped)
"""

import sys
import numpy as np

sys.path.insert(0, '/home/lyra/projects/evalue-sheaf/code/evalue')

from dgp import null_dgp, alt_dgp
from eprocess import THRESHOLD
from run_online_weights import build_increment_stream, lam_fixed, wealth_path, Q, BLOCK

# ── Configuration ──────────────────────────────────────────────────────────────
SEED       = 20261120
N          = 4000
R          = 10000
ALPHA      = 0.05
THRESH     = THRESHOLD   # 20.0
SLACK      = 'second_order'
GAMMAS     = [0.0, 0.04, 0.06, 0.08, 0.10]
SCHEMES    = ['cyclic', 'loo']


def ci95(p, n):
    half = 1.96 * np.sqrt(p * (1 - p) / n)
    return p - half, p + half


def run_power(gamma, scheme, seed0, R, N):
    """Return rejection rate for scheme at given gamma."""
    rej = 0
    dgp_fn = null_dgp if gamma == 0.0 else (lambda n, rng: alt_dgp(n, gamma, rng))
    for r in range(R):
        rng = np.random.default_rng(seed0 + r)
        data = dgp_fn(N, rng)
        X, b = build_increment_stream(data, rng, pairing='within_block',
                                      slack=SLACK, pairing_rule=scheme)
        lf = lam_fixed(b)
        Mp = wealth_path(X, lf)
        if np.any(Mp >= THRESH):
            rej += 1
    return rej / R


def main():
    print("=" * 90)
    print("POWER MEASUREMENT: cyclic vs LOO, delta_2 (second_order), R=10000")
    print(f"  Q={Q}, N={N}, BLOCK={BLOCK}, R={R}, alpha={ALPHA}, threshold={THRESH}")
    print(f"  SEED={SEED}  (seed0+r for r in 0..{R-1})")
    print(f"  Gammas: {GAMMAS}")
    print("=" * 90)

    results = {}  # (scheme, gamma) -> (power, lo_ci, hi_ci)

    for gamma in GAMMAS:
        for scheme in SCHEMES:
            label = f"gamma={gamma:.2f}, {scheme}"
            print(f"  Running {label} ...", flush=True)
            p = run_power(gamma, scheme, SEED, R, N)
            lo, hi = ci95(p, R)
            results[(scheme, gamma)] = (p, lo, hi)
            print(f"    power={p:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]", flush=True)

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print(f"  {'scheme':<8} {'gamma':>6} {'power':>8} {'CI_lo':>8} {'CI_hi':>8}")
    print("  " + "-" * 46)
    for gamma in GAMMAS:
        for scheme in SCHEMES:
            p, lo, hi = results[(scheme, gamma)]
            tag = "  << TYPE-I" if gamma == 0.0 else ""
            print(f"  {scheme:<8} {gamma:>6.2f} {p:>8.4f} {lo:>8.4f} {hi:>8.4f}{tag}")
        print()

    # ── Verdict: gamma=0.06 CI overlap ──────────────────────────────────────
    print("=" * 90)
    print("VERDICT: gamma=0.06")
    p_cyc, lo_cyc, hi_cyc = results[('cyclic', 0.06)]
    p_loo, lo_loo, hi_loo = results[('loo', 0.06)]
    overlap = lo_loo < hi_cyc and lo_cyc < hi_loo
    print(f"  cyclic: {p_cyc:.4f} [{lo_cyc:.4f}, {hi_cyc:.4f}]")
    print(f"  LOO:    {p_loo:.4f} [{lo_loo:.4f}, {hi_loo:.4f}]")
    print(f"  CIs overlap: {overlap}")
    if overlap:
        print("  => The ~0.05 power edge is NOT resolved at R=10000; could be Monte-Carlo noise.")
    else:
        print("  => CIs DO NOT overlap; LOO power edge is statistically real at R=10000.")

    # ── Verdict: sign consistency across gammas ──────────────────────────────
    print("\nSIGN CONSISTENCY (LOO power minus cyclic power):")
    for gamma in GAMMAS:
        if gamma == 0.0:
            continue
        p_cyc, _, _ = results[('cyclic', gamma)]
        p_loo, _, _ = results[('loo', gamma)]
        diff = p_loo - p_cyc
        print(f"  gamma={gamma:.2f}: LOO - cyclic = {diff:+.4f}")

    # ── Type-I sanity ─────────────────────────────────────────────────────────
    print("\nTYPE-I SANITY (gamma=0, must be <= 0.05):")
    p_cyc0, lo_cyc0, hi_cyc0 = results[('cyclic', 0.0)]
    p_loo0, lo_loo0, hi_loo0 = results[('loo', 0.0)]
    print(f"  cyclic: {p_cyc0:.4f} [{lo_cyc0:.4f}, {hi_cyc0:.4f}] => {'OK' if p_cyc0 <= 0.05 else 'VIOLATION'}")
    print(f"  LOO:    {p_loo0:.4f} [{lo_loo0:.4f}, {hi_loo0:.4f}] => {'OK' if p_loo0 <= 0.05 else 'VIOLATION'}")
    if p_cyc0 > 0.05 or p_loo0 > 0.05:
        print("\n  !!! TYPE-I VIOLATION — STOP. This is a validity problem, not a power question. !!!")
    else:
        print("  Both Type-I rates within tolerance.")

    print("\nDone.")


if __name__ == '__main__':
    main()
