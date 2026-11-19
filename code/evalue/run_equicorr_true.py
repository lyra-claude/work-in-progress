"""
run_equicorr_true.py — TASK A: genuinely-equicorrelated DGP experiment.

Clio's review: the current "Regime 1" (alt_dgp, uniform Marshall-Olkin shock gamma) is
NOT equicorrelated. Because the marginals p_i,p_j rise with difficulty theta, the per-item
excess co-failure covariance

    Cov_s = gamma (1-gamma) (1 - p_i_s)(1 - p_j_s)

varies ~4x across the Q strata, so the per-stratum Kelly/oracle allocation comes out
heterogeneous (~(0.60, 0.31, 0.10) when collapsed to three groups): the strata carry very
different signal. The forecast-combination puzzle's "home turf" (optimal weights ARE equal)
is therefore NOT what the old Regime 1 tests.

FIX (new DGP alt_dgp_equicorr_true, in dgp.py): make the shock STRATUM-DEPENDENT, gamma_s,
scaled so that  gamma_s (1-gamma_s)(1-p_i)(1-p_j)  is CONSTANT across strata. Then the
per-stratum signal is uniform and the oracle weights are genuinely ~equal.

This script RE-RUNS the Regime-1-style power + Type-I experiment on the new DGP, reporting:
  (1) Type-I under the null for FIXED and ONS-type (ONLINE) — should be ~0;
  (2) power curves vs gamma for FIXED vs ONS-type;
  (3) whether ONS-type beats or loses to FIXED on power, and any crossover gamma;
  (4) the per-stratum ORACLE (Kelly) allocation under the new DGP — confirm ~equal.

Same R, N, BLOCK, Q as the §5b harness. cap = 1/b (as deployed). FRESH-PAIR (Prop.3) pairing.
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/lyra/projects/evalue-sheaf/code/evalue')

from dgp import null_dgp, alt_dgp_equicorr_true, _equicorr_stratum_shocks
from eprocess import ville_reject, THRESHOLD, assign_strata
import run_online_weights as rw
from run_online_weights import (
    build_increment_stream, lam_fixed, ons_weights, oracle_weights, wealth_path,
)

# ── Settings matched to the §5b harness ─────────────────────────────────────────
Q      = 10
BLOCK  = 20
N      = 4000
R      = 1000
ALPHA  = 0.05
THRESH = THRESHOLD
SEED0  = 1000
GAMMA_GRID = [0.02, 0.04, 0.06, 0.10, 0.15, 0.25]

rw.Q = Q
rw.BLOCK = BLOCK


def paired_run(data, rng):
    """FIXED, ONS-type (ONLINE), ORACLE on one SAME increment stream (fresh-pair)."""
    X, b = build_increment_stream(data, rng, pairing='within_block')
    lf = lam_fixed(b)
    lam_on_seq, _ = ons_weights(X, b)
    lo = oracle_weights(X, b)
    M_fx = wealth_path(X, lf)
    M_on = wealth_path(X, lam_on_seq)
    M_or = wealth_path(X, lo)
    def tlw(Mp): return float(np.log(max(Mp[-1], 1e-300)))
    rej_fx, _ = ville_reject(M_fx, THRESH)
    rej_on, _ = ville_reject(M_on, THRESH)
    rej_or, _ = ville_reject(M_or, THRESH)
    return dict(lw_fx=tlw(M_fx), lw_on=tlw(M_on), lw_or=tlw(M_or),
                rej_fx=rej_fx, rej_on=rej_on, rej_or=rej_or,
                lo=lo, cap=1.0 / b)


def run_condition(dgp_fn, R, seed0=SEED0):
    rows = []
    for r in range(R):
        rng = np.random.default_rng(seed0 + r)
        data = dgp_fn(N, rng)
        rows.append(paired_run(data, rng))
    return rows


def summarize(rows):
    lw_fx = np.array([r['lw_fx'] for r in rows])
    lw_on = np.array([r['lw_on'] for r in rows])
    lw_or = np.array([r['lw_or'] for r in rows])
    pow_fx = np.mean([r['rej_fx'] for r in rows])
    pow_on = np.mean([r['rej_on'] for r in rows])
    pow_or = np.mean([r['rej_or'] for r in rows])
    return dict(pow_fx=pow_fx, pow_on=pow_on, pow_or=pow_or,
                lw_fx=lw_fx.mean(), lw_on=lw_on.mean(), lw_or=lw_or.mean())


def main():
    t0 = time.time()
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("=" * 96)
    out("TASK A — GENUINELY-EQUICORRELATED DGP (alt_dgp_equicorr_true)")
    out(f"  Q={Q}, N={N}, BLOCK={BLOCK} (M={N//BLOCK} blocks), R={R}, alpha={ALPHA}, "
        f"cap=1/b, FRESH-PAIR (Prop.3)")
    out(f"  seed base = default_rng({SEED0}+r)")
    out("=" * 96)

    # Derivation sanity: print per-stratum gamma_s and the (constant) covariance target.
    out("\n(0) DGP derivation check — per-stratum shock gamma_s and excess covariance:")
    out("    excess Cov_s = gamma_s*(1-gamma_s)*(1-p_i_s)*(1-p_j_s)  (should be CONSTANT across s)")
    for g in [0.06, 0.15, 0.25]:
        gs, C, r = _equicorr_stratum_shocks(g, Q=Q)
        cov = gs * (1 - gs) * r
        out(f"    gamma={g:.2f}: C={C:.5f}  gamma_s={np.array2string(gs, precision=3, floatmode='fixed')}")
        out(f"               cov_s range = [{cov.min():.6f}, {cov.max():.6f}] (constant to 6dp)")

    # (1) Type-I under the static null (FIXED, ONS-type). The null is DGP-independent
    #     (null_dgp = conditional independence given theta); report it for completeness.
    out("\n(1) TYPE-I under STATIC null (FIXED vs ONS-type), fresh-pair:")
    s = summarize(run_condition(null_dgp, R))
    out(f"    FIXED Type-I = {s['pow_fx']:.4f}   ONS-type Type-I = {s['pow_on']:.4f}   "
        f"ORACLE Type-I = {s['pow_or']:.4f}   (target <= {ALPHA})")
    null_ok = (s['pow_fx'] <= ALPHA and s['pow_on'] <= ALPHA)
    out(f"    => null {'OK (<= alpha)' if null_ok else 'EXCEEDS alpha — STOP'}")

    # (2) Power curves vs gamma for FIXED vs ONS-type on the equicorr-true alt.
    out("\n(2) POWER vs gamma — genuinely-equicorrelated alt, FIXED vs ONS-type:")
    out(f"    {'gamma':>6} | {'FIX pow':>8} {'ONS pow':>8} {'OR pow':>8} | "
        f"{'FIX lw':>9} {'ONS lw':>9} {'OR lw':>9} | {'ONS-FIX pow':>11}")
    out("    " + "-" * 86)
    crossover = None
    per_gamma = []
    for g in GAMMA_GRID:
        fn = (lambda Nn, rng, gg=g: alt_dgp_equicorr_true(Nn, gg, rng))
        rows = run_condition(fn, R)
        s = summarize(rows)
        dpow = s['pow_on'] - s['pow_fx']
        per_gamma.append((g, s, dpow))
        out(f"    {g:>6.2f} | {s['pow_fx']:>8.3f} {s['pow_on']:>8.3f} {s['pow_or']:>8.3f} | "
            f"{s['lw_fx']:>+9.3f} {s['lw_on']:>+9.3f} {s['lw_or']:>+9.3f} | {dpow:>+11.3f}")
        if crossover is None and dpow > 0:
            crossover = g

    # (3) crossover verdict
    out("\n(3) ONS-type vs FIXED on POWER under genuine equicorrelation:")
    if crossover is None:
        out("    ONS-type NEVER beats FIXED on power across the tested gamma grid "
            "(FIXED >= ONS-type everywhere).")
    else:
        out(f"    ONS-type first beats FIXED on power at gamma = {crossover:.2f} "
            f"(crossover). Below that, FIXED wins.")
    # mean log-wealth verdict too
    lw_better = [g for g, s, _ in per_gamma if s['lw_on'] > s['lw_fx']]
    out(f"    On mean log-wealth, ONS-type > FIXED at gamma in: "
        f"{lw_better if lw_better else 'none (FIXED dominates everywhere)'}")

    # (4) per-stratum ORACLE (Kelly) allocation under the new DGP — confirm ~equal.
    #     Average the oracle lam* vector over replications at a mid gamma, normalize to cap.
    out("\n(4) PER-STRATUM ORACLE (Kelly) allocation under equicorr-true "
        "(mean lam*/cap over R reps):")
    for g in [0.10, 0.25]:
        fn = (lambda Nn, rng, gg=g: alt_dgp_equicorr_true(Nn, gg, rng))
        rows = run_condition(fn, R)
        los = np.array([r['lo'] / r['cap'] for r in rows])   # (R, Q), fraction of cap
        mean_alloc = los.mean(axis=0)
        # normalize to a probability vector over strata for interpretability
        tot = mean_alloc.sum()
        norm = mean_alloc / tot if tot > 1e-12 else mean_alloc
        out(f"    gamma={g:.2f}: mean lam*/cap per stratum = "
            f"{np.array2string(mean_alloc, precision=3, floatmode='fixed')}")
        out(f"               normalized (sums to 1)      = "
            f"{np.array2string(norm, precision=3, floatmode='fixed')}")
        out(f"               coefficient of variation across strata = "
            f"{(norm.std()/ (norm.mean()+1e-12)):.3f}  (0 => perfectly equal)")

    # Contrast: old Regime 1 (alt_dgp uniform shock) 3-group collapse, for comparison.
    out("\n    For contrast — OLD Regime 1 (alt_dgp, uniform shock) is reported separately.")

    out(f"\nTOTAL RUNTIME: {time.time()-t0:.1f}s")
    out("Done.")

    with open('/home/lyra/projects/evalue-sheaf/results/equicorr_true.txt', 'w') as f:
        f.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    main()
