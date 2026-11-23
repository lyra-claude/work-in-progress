"""
mixwin_exchangeable_control.py — EXCHANGEABLE-DGP CONTROL for the Regime-1
Eq-vs-MixWin / per-component diagnostics.

MOTIVATION
----------
Under the Regime-1 DGP (`alt_dgp`, uniform Marshall-Olkin shock gamma), the
per-stratum excess co-failure covariance  gamma*(1-gamma)*(1-p_i_s)*(1-p_j_s)
DECREASES monotonically with stratum index because the judge marginals p_i, p_j
rise with difficulty theta (ratio stratum0/stratum9 ~= 4.0).  Prior diagnostics
found the 9 MixWin components have monotonically decreasing mean terminal
log-wealth by window index and equal-weights is almost never the per-seed argmax.
That is exactly what a CONCENTRATED (not exchangeable) signal predicts.

This control swaps `alt_dgp` for `alt_dgp_exchangeable`, which flattens the
marginals to constants (p_i=0.30, p_j=0.35 = judge_probs(0.5)) while keeping the
SAME uniform shock gamma, so every stratum has the SAME per-stratum covariance
(genuinely stratum-exchangeable) and the SAME total drift is spread uniformly
across all 10 strata.  Total signal matches `alt_dgp` to within ~4% at a given
gamma.

Everything else is IDENTICAL to mixwin_perseed_regime1.py /
mixwin_components_regime1.py: Q=10, N=4000, BLOCK=20, THRESH=20, R=600 paired
seeds, seed base 50000+r, same BUILD_KW, same combiners (Eq + 8 windows).

PREDICTION being tested (under exchangeability):
  - the 8 window components have ~equal mean terminal log-wealth (symmetric),
  - equal-weights (Eq) is the per-seed argmax MUCH more often than under Regime 1,
  - Eq has the HIGHEST mean terminal log-wealth of the 9 components.

This script ONLY measures and prints numbers. It does not interpret them.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, '/home/lyra/projects/evalue-sheaf/code/evalue')

from dgp import alt_dgp_exchangeable
from reproduce_mixwin_regime1_weak import (
    Q, BLOCK, N, ALPHA, THRESH, SEED_BASE, BUILD_KW,
    eq_wealth, mixwin_wealth, mixwin_components,
)
from run_online_weights import build_increment_stream, wealth_path
from eprocess import ville_reject

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
    _HAVE_SCIPY_WILCOXON = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY_WILCOXON = False

# ── Runtime settings (match the Regime-1 diagnostics) ───────────────────────────
R = 600
GAMMAS = [0.05, 0.07]        # same total-signal levels as Regime 1; adjust if power saturates
N_COMP = Q - 1               # = 9 (Eq + 8 windows)


def component_labels():
    labs = ["Eq (component 0)"]
    for k0 in range(Q - 2):                       # 0..7
        labs.append(f"win{{{k0},{k0 + 1},{k0 + 2}}} (component {k0 + 1})")
    return labs


def run_gamma(gamma, R):
    """R paired seeds at one gamma on the EXCHANGEABLE DGP.

    Returns arrays:
        rej_eq, rej_mw         : (R,) bool Ville-reject indicators
        log_eq, log_mw         : (R,) terminal log-wealth, Eq and MixWin mixture
        L_comp                 : (R, N_COMP) terminal log-wealth per component
    """
    rej_eq = np.empty(R, dtype=bool); rej_mw = np.empty(R, dtype=bool)
    log_eq = np.empty(R); log_mw = np.empty(R)
    L_comp = np.empty((R, N_COMP))

    for r in range(R):
        rng = np.random.default_rng(SEED_BASE + r)
        data = alt_dgp_exchangeable(N, gamma, rng)
        X, b = build_increment_stream(data, rng, **BUILD_KW)

        M_eq = eq_wealth(X, b)
        M_mw = mixwin_wealth(X, b)
        re, _ = ville_reject(M_eq, THRESH)
        rm, _ = ville_reject(M_mw, THRESH)
        rej_eq[r] = re; rej_mw[r] = rm
        log_eq[r] = np.log(M_eq[-1]); log_mw[r] = np.log(M_mw[-1])

        comps = mixwin_components(b)
        paths = np.array([wealth_path(X, w) for w in comps])  # (9, M)
        L_comp[r] = np.log(paths[:, -1])

    return dict(rej_eq=rej_eq, rej_mw=rej_mw, log_eq=log_eq, log_mw=log_mw,
                L_comp=L_comp)


def pct(a, q):
    return float(np.percentile(a, q))


def summarize(gamma, D, labels, emit):
    emit("=" * 88)
    emit(f"gamma = {gamma:.2f}   (R = {R} paired seeds, seed base {SEED_BASE}+r)")
    emit(f"  DGP: alt_dgp_exchangeable (flat marginals, uniform shock) | build={BUILD_KW}")
    emit("=" * 88)

    rej_eq, rej_mw = D['rej_eq'], D['rej_mw']
    log_eq, log_mw = D['log_eq'], D['log_mw']
    L_comp = D['L_comp']

    # ── [1] reject rates + power gap ─────────────────────────────────────────────
    rr_eq = float(rej_eq.mean()); rr_mw = float(rej_mw.mean())
    emit("[1] Reject rate (fraction ever-crossing THRESH=%.1f)" % THRESH)
    emit(f"      Eq      : {rr_eq:.4f}  ({int(rej_eq.sum())}/{R})")
    emit(f"      MixWin  : {rr_mw:.4f}  ({int(rej_mw.sum())}/{R})")
    emit(f"      power gap (Eq - MixWin) : {rr_eq - rr_mw:+.4f}")

    # ── [2] terminal log-wealth Eq vs MixWin + paired diff ──────────────────────
    emit("[2] Terminal log-wealth  (mean, median)")
    emit(f"      Eq      : mean={log_eq.mean():+.4f}  median={np.median(log_eq):+.4f}")
    emit(f"      MixWin  : mean={log_mw.mean():+.4f}  median={np.median(log_mw):+.4f}")
    diff = log_mw - log_eq
    med_diff = float(np.median(diff))
    frac_mw_gt = float(np.mean(log_mw > log_eq))
    nz = diff[diff != 0.0]
    if _HAVE_SCIPY_WILCOXON and len(nz) > 0:
        try:
            _, pval = _scipy_wilcoxon(nz)
        except Exception as e:
            pval = float('nan'); emit(f"      (scipy wilcoxon raised: {e})")
    else:
        pval = float('nan')
    emit("    Paired difference (MixWin - Eq):")
    emit(f"      median paired diff : {med_diff:+.4f}")
    emit(f"      Wilcoxon p-value   : {pval:.3e}  [scipy; n_nonzero={len(nz)}]")
    emit(f"      frac(MixWin > Eq)  : {frac_mw_gt:.4f}")

    # ── [3] per-seed argmax over the 9 components ───────────────────────────────
    argmax = np.argmax(L_comp, axis=1)
    emit("[3] Per-seed ARGMAX across the 9 components")
    frac_eq = float(np.mean(argmax == 0))
    emit(f"      argmax = Eq (component 0) : {frac_eq:.4f}  ({int(np.sum(argmax == 0))}/{R})")
    emit(f"      argmax in windows 1..8    : {float(np.mean(argmax >= 1)):.4f}  "
         f"({int(np.sum(argmax >= 1))}/{R})")
    emit("      per-component argmax breakdown:")
    for j in range(N_COMP):
        cnt = int(np.sum(argmax == j))
        emit(f"        [{j}] {labels[j]:<28}: {cnt / R:.4f}  ({cnt}/{R})")

    # ── [4] mean terminal log-wealth per component ──────────────────────────────
    emit("[4] Mean terminal log-wealth per component (across all seeds)")
    means = L_comp.mean(axis=0)
    order = np.argsort(-means)
    for j in range(N_COMP):
        tag = "  <- highest MEAN" if j == order[0] else ""
        emit(f"        [{j}] {labels[j]:<28}: mean={means[j]:+.6f}  "
             f"median={np.median(L_comp[:, j]):+.6f}  "
             f"sd={L_comp[:, j].std(ddof=1):.6f}{tag}")
    # window-symmetry read: spread of the 8 window means
    win_means = means[1:]
    emit(f"      window means (1..8): min={win_means.min():+.6f}  "
         f"max={win_means.max():+.6f}  range={win_means.max() - win_means.min():.6f}  "
         f"sd_across_windows={win_means.std(ddof=1):.6f}")
    emit(f"      Eq mean - mean(window means) : {means[0] - win_means.mean():+.6f}")
    emit("")
    return rr_eq, rr_mw


def main():
    t0 = time.time()
    lines = []

    def emit(s):
        print(s); lines.append(s)

    labels = component_labels()

    emit("#" * 88)
    emit("EXCHANGEABLE-DGP CONTROL — Eq vs MixWin + per-component terminal log-wealth")
    emit(f"  Q={Q}, N={N}, BLOCK={BLOCK} (M={N // BLOCK} blocks), alpha={ALPHA}, THRESH={THRESH}")
    emit(f"  R={R} paired seeds, seed base {SEED_BASE}+r | build={BUILD_KW}")
    emit(f"  DGP: alt_dgp_exchangeable (flat marginals p_i=0.30 p_j=0.35, uniform shock gamma)")
    emit(f"  MixWin = (1/9) mean of [Eq + 8 windows {{0,1,2}}..{{7,8,9}}]")
    emit(f"  scipy.stats.wilcoxon available: {_HAVE_SCIPY_WILCOXON}")
    emit("#" * 88)
    emit("")

    for g in GAMMAS:
        tg = time.time()
        D = run_gamma(g, R)
        summarize(g, D, labels, emit)
        emit(f"  (gamma={g:.2f} runtime: {time.time() - tg:.1f}s, R={R})")
        emit("")

    emit("=" * 88)
    emit(f"TOTAL RUNTIME: {time.time() - t0:.1f}s  (R={R} per gamma, {len(GAMMAS)} gammas)")
    emit("=" * 88)

    logdir = '/home/lyra/projects/evalue-sheaf/code/evalue/logs'
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, 'mixwin_exchangeable_control.log'), 'w') as f:
        f.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    main()
