"""
run_cap09_logwealth.py — TASK B: log-wealth re-run with cap = 0.9/b.

Clio showed that cap = 1/b clamps the mean log-wealth to a ruin FLOOR (~ -690.8 =
1000 * log(1e-300) averaged in, i.e. wealth paths that hit 0 and are floored). At the
Vovk-Wang cap 1/b a single block with <lam, X_m> = -1 sends the factor to 0 and the whole
path to the floor, masking any ONS edge: both FIXED and ONS-type collapse to the floor on
those seeds and the mean is dominated by the floor, not by the growth differences.

Re-run the FIXED vs ONS-type log-wealth comparison with cap = 0.9/b instead of 1/b, on the
EXISTING Regime-1 DGP (alt_dgp, uniform Marshall-Olkin shock) — NOT the new equicorr DGP —
to keep it comparable to prior runs. A smaller cap keeps every factor 1 + <lam,X_m> >=
1 - 0.9 = 0.1 > 0 (since |<lam,X_m>| <= cap*b = 0.9), so no path ruins; the mean log-wealth
then reflects genuine growth and any ONS edge can appear.

MECHANISM: cap=0.9/b is obtained by feeding the weight functions an inflated bound
b' = b/0.9, so their internal cap = 1/b' = 0.9/b. This is strictly SAFER (smaller cap) and
keeps feasibility/ONS constants self-consistent. The increment stream X and its true bound
b are unchanged (same data), so the comparison to the cap=1/b run is apples-to-apples.

Reports: mean log-wealth FIXED vs ONS-type at cap=0.9/b across the gamma grid (alongside the
cap=1/b baseline), whether the ONS-type edge appears and at which gamma, and Type-I at
cap=0.9/b (must stay ~0).
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/lyra/projects/evalue-sheaf/code/evalue')

from dgp import null_dgp, alt_dgp
from eprocess import ville_reject, THRESHOLD
import run_online_weights as rw
from run_online_weights import (
    build_increment_stream, lam_fixed, ons_weights, oracle_weights, wealth_path,
)

Q      = 10
BLOCK  = 20
N      = 4000
R      = 1000
ALPHA  = 0.05
THRESH = THRESHOLD
SEED0  = 1000
GAMMA_GRID = [0.02, 0.04, 0.06, 0.10, 0.15, 0.25]
CAP_SCALE = 0.9        # cap = CAP_SCALE / b

rw.Q = Q
rw.BLOCK = BLOCK


def paired_run(data, rng, cap_scale):
    """
    Build the increment stream once (true bound b), then evaluate FIXED and ONS-type at
    cap = cap_scale / b by passing b_eff = b / cap_scale to the weight functions.
    """
    X, b = build_increment_stream(data, rng, pairing='within_block')
    b_eff = b / cap_scale                       # => internal cap = 1/b_eff = cap_scale/b
    lf = lam_fixed(b_eff)
    lam_on_seq, _ = ons_weights(X, b_eff)
    M_fx = wealth_path(X, lf)
    M_on = wealth_path(X, lam_on_seq)
    def tlw(Mp): return float(np.log(max(Mp[-1], 1e-300)))
    rej_fx, _ = ville_reject(M_fx, THRESH)
    rej_on, _ = ville_reject(M_on, THRESH)
    # floor diagnostic: fraction of paths that hit the 1e-300 wealth floor (ruin)
    ruin_fx = M_fx[-1] <= 1e-299
    ruin_on = M_on[-1] <= 1e-299
    return dict(lw_fx=tlw(M_fx), lw_on=tlw(M_on),
                rej_fx=rej_fx, rej_on=rej_on,
                ruin_fx=ruin_fx, ruin_on=ruin_on)


def run_condition(dgp_fn, R, cap_scale, seed0=SEED0):
    rows = []
    for r in range(R):
        rng = np.random.default_rng(seed0 + r)
        data = dgp_fn(N, rng)
        rows.append(paired_run(data, rng, cap_scale))
    return rows


def agg(rows):
    lw_fx = np.array([r['lw_fx'] for r in rows])
    lw_on = np.array([r['lw_on'] for r in rows])
    return dict(
        lw_fx=lw_fx.mean(), lw_on=lw_on.mean(),
        med_fx=float(np.median(lw_fx)), med_on=float(np.median(lw_on)),
        win_on=float(np.mean(lw_on > lw_fx)),
        pow_fx=np.mean([r['rej_fx'] for r in rows]),
        pow_on=np.mean([r['rej_on'] for r in rows]),
        ruin_fx=np.mean([r['ruin_fx'] for r in rows]),
        ruin_on=np.mean([r['ruin_on'] for r in rows]),
    )


def main():
    t0 = time.time()
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("=" * 100)
    out("TASK B — log-wealth re-run with cap = 0.9/b (vs baseline cap = 1/b), EXISTING Regime-1 DGP (alt_dgp)")
    out(f"  Q={Q}, N={N}, BLOCK={BLOCK} (M={N//BLOCK} blocks), R={R}, alpha={ALPHA}, "
        f"FRESH-PAIR (Prop.3), seed base=default_rng({SEED0}+r)")
    out("=" * 100)

    for cap_scale, tag in [(1.0, "cap=1/b  (baseline)"), (0.9, "cap=0.9/b (Task B)")]:
        out(f"\n### {tag} ###")
        out(f"    {'gamma':>6} | {'FIX lw':>10} {'ONS lw':>10} | {'FIX med':>9} {'ONS med':>9} "
            f"| {'ONS win%':>8} | {'FIX pow':>8} {'ONS pow':>8} | {'FIX ruin%':>9} {'ONS ruin%':>9}")
        out("    " + "-" * 104)
        for g in GAMMA_GRID:
            fn = (lambda Nn, rng, gg=g: alt_dgp(Nn, gg, rng))
            s = agg(run_condition(fn, R, cap_scale))
            out(f"    {g:>6.2f} | {s['lw_fx']:>+10.3f} {s['lw_on']:>+10.3f} | "
                f"{s['med_fx']:>+9.3f} {s['med_on']:>+9.3f} | {100*s['win_on']:>7.1f}% | "
                f"{s['pow_fx']:>8.3f} {s['pow_on']:>8.3f} | "
                f"{100*s['ruin_fx']:>8.1f}% {100*s['ruin_on']:>8.1f}%")

    # Type-I at cap=0.9/b on the static null (must be ~0)
    out("\n### Type-I at cap=0.9/b on STATIC null (must be ~0) ###")
    s = agg(run_condition(null_dgp, R, 0.9))
    out(f"    FIXED Type-I = {s['pow_fx']:.4f}   ONS-type Type-I = {s['pow_on']:.4f}   "
        f"(target <= {ALPHA})")

    out(f"\nTOTAL RUNTIME: {time.time()-t0:.1f}s")
    out("Done.")

    with open('/home/lyra/projects/evalue-sheaf/results/cap09_logwealth.txt', 'w') as f:
        f.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    main()
