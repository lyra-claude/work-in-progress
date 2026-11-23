"""
mixwin_components_regime1.py — PER-COMPONENT terminal log-wealth diagnostics on
the Regime-1 uniform-shock DGP.

Companion to mixwin_perseed_regime1.py. Where that script retains only the Eq
and MixWin-mixture terminal quantities, this script decomposes the mixture into
its 9 INDIVIDUAL components and measures each one's terminal log-wealth per seed,
alongside the mixture's terminal log-wealth.

Component indexing (from mixwin_components in reproduce_mixwin_regime1_weak):
    index 0      : equal-weights (Eq)
    index 1..8   : the 8 three-stratum sliding windows {0,1,2}..{7,8,9}

Same paired DGP, same BUILD_KW, N/Q/BLOCK/THRESH, same paired seed base (50000+r)
as mixwin_perseed_regime1.py. This script ONLY measures and prints numbers; it
does not interpret them.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, '/home/lyra/projects/evalue-sheaf/code/evalue')

from dgp import alt_dgp
from reproduce_mixwin_regime1_weak import (
    Q, BLOCK, N, ALPHA, THRESH, SEED_BASE, BUILD_KW,
    mixwin_components,
)
from run_online_weights import build_increment_stream, wealth_path

N_COMP = Q - 1          # = 9 for Q=10 (Eq + 8 windows)

# ── Runtime settings ───────────────────────────────────────────────────────────
R = 600                 # paired seeds per gamma (drop to 400 if too slow)
GAMMAS = [0.05, 0.07]


def component_labels():
    labs = ["Eq (component 0)"]
    for k0 in range(Q - 2):        # 0..7
        labs.append(f"win{{{k0},{k0 + 1},{k0 + 2}}} (component {k0 + 1})")
    return labs


def run_gamma(gamma, R):
    """R paired seeds at one gamma.

    Returns:
        L_comp : (R, N_COMP) terminal log-wealth of each component per seed.
        L_mix  : (R,)        terminal log-wealth of the (1/9)-mean-of-wealths mixture.
    """
    L_comp = np.empty((R, N_COMP))
    L_mix = np.empty(R)

    for r in range(R):
        rng = np.random.default_rng(SEED_BASE + r)
        data = alt_dgp(N, gamma, rng)
        X, b = build_increment_stream(data, rng, **BUILD_KW)

        comps = mixwin_components(b)                         # 9 weight vectors
        # Full wealth PROCESSES for each component (same as mixwin_wealth internals).
        paths = np.array([wealth_path(X, w) for w in comps])  # (9, M)
        term_wealth = paths[:, -1]                            # (9,) terminal wealths

        L_comp[r] = np.log(term_wealth)
        # Mixture terminal wealth = (1/9) mean of the 9 terminal wealths
        # (equivalently the terminal value of the mean-of-processes mixture, since
        #  the mean commutes with taking the terminal index).
        L_mix[r] = np.log(term_wealth.mean())

    return L_comp, L_mix


def pct(a, q):
    return float(np.percentile(a, q))


def summarize(gamma, L_comp, L_mix, labels, emit):
    R_eff = L_comp.shape[0]
    emit("=" * 88)
    emit(f"gamma = {gamma:.2f}   (R = {R_eff} paired seeds, seed base {SEED_BASE}+r)")
    emit(f"  build={BUILD_KW}")
    emit("=" * 88)

    # Per-seed argmax component.
    argmax = np.argmax(L_comp, axis=1)                      # (R,)
    L_max = L_comp.max(axis=1)                              # (R,) max_j L_j
    L0 = L_comp[:, 0]                                       # equal-weights

    # ── [1] ARGMAX frequency ────────────────────────────────────────────────────
    emit("[1] Per-seed ARGMAX (component with highest terminal log-wealth)")
    frac_eq = float(np.mean(argmax == 0))
    frac_win = float(np.mean(argmax >= 1))
    emit(f"      argmax = component 0 (equal-weights) : {frac_eq:.4f}  "
         f"({int(np.sum(argmax == 0))}/{R_eff})")
    emit(f"      argmax in windows (components 1..8)   : {frac_win:.4f}  "
         f"({int(np.sum(argmax >= 1))}/{R_eff})")
    emit("      per-component argmax breakdown:")
    for j in range(N_COMP):
        cnt = int(np.sum(argmax == j))
        emit(f"        [{j}] {labels[j]:<28}: {cnt / R_eff:.4f}  ({cnt}/{R_eff})")

    # ── [2] SANDWICH check ──────────────────────────────────────────────────────
    # (max_j L_j) - log(9) <= L_mixture <= (max_j L_j)
    lo = L_max - np.log(N_COMP)
    hi = L_max
    tol = 1e-9
    holds = (L_mix >= lo - tol) & (L_mix <= hi + tol)
    gap = L_max - L_mix                                     # (max_j L_j) - L_mixture
    emit("[2] SANDWICH:  (max_j L_j) - log(9) <= L_mixture <= (max_j L_j)")
    emit(f"      fraction of seeds where bound holds : {np.mean(holds):.4f}  "
         f"({int(np.sum(holds))}/{R_eff})")
    emit(f"      gap = (max_j L_j) - L_mixture:")
    emit(f"        mean   = {gap.mean():.6f}")
    emit(f"        p10    = {pct(gap, 10):.6f}")
    emit(f"        median = {np.median(gap):.6f}")
    emit(f"        p90    = {pct(gap, 90):.6f}")
    emit(f"      (log(9) = {np.log(N_COMP):.6f} is the sandwich width upper bound on the gap)")

    # ── [3] MixWin vs EW via the max ────────────────────────────────────────────
    emit("[3] MixWin vs EW via the max  (means across all seeds)")
    emit(f"      mean(max_j L_j)                 : {L_max.mean():+.6f}")
    emit(f"      mean(L_0, equal-weights)        : {L0.mean():+.6f}")
    emit(f"      mean(L_mixture)                 : {L_mix.mean():+.6f}")
    beats = L_max > L0
    frac_beats = float(np.mean(beats))
    if np.any(beats):
        mean_margin_when_beats = float((L_max[beats] - L0[beats]).mean())
    else:
        mean_margin_when_beats = float('nan')
    emit(f"      frac(max_j L_j > L_0)           : {frac_beats:.4f}  "
         f"({int(np.sum(beats))}/{R_eff})")
    emit(f"      mean(max_j L_j - L_0 | beats)   : {mean_margin_when_beats:+.6f}")

    # ── [4] Mean terminal log-wealth of each component ──────────────────────────
    emit("[4] Mean terminal log-wealth per component (across all seeds)")
    means = L_comp.mean(axis=0)
    order = np.argsort(-means)   # descending for a quick read
    for j in range(N_COMP):
        tag = "  <- highest MEAN" if j == order[0] else ""
        emit(f"        [{j}] {labels[j]:<28}: mean={means[j]:+.6f}  "
             f"median={np.median(L_comp[:, j]):+.6f}  "
             f"sd={L_comp[:, j].std(ddof=1):.6f}{tag}")
    emit("")


def main():
    t0 = time.time()
    lines = []

    def emit(s):
        print(s)
        lines.append(s)

    labels = component_labels()

    emit("#" * 88)
    emit("MixWin PER-COMPONENT terminal log-wealth — Regime-1 uniform shock")
    emit(f"  Q={Q}, N={N}, BLOCK={BLOCK} (M={N // BLOCK} blocks), alpha={ALPHA}, "
         f"THRESH={THRESH}")
    emit(f"  N_COMP={N_COMP} (component 0 = equal-weights; 1..8 = sliding windows)")
    emit(f"  DGP: alt_dgp (uniform Marshall-Olkin shock across all strata)")
    emit(f"  mixture L = log((1/9) * sum_j terminal_wealth_j)")
    emit("#" * 88)
    emit("")

    r_used = {}
    for g in GAMMAS:
        tg = time.time()
        L_comp, L_mix = run_gamma(g, R)
        summarize(g, L_comp, L_mix, labels, emit)
        dt = time.time() - tg
        r_used[g] = (R, dt)
        emit(f"  (gamma={g:.2f} runtime: {dt:.1f}s, R={R})")
        emit("")

    emit("=" * 88)
    emit(f"TOTAL RUNTIME: {time.time() - t0:.1f}s   "
         f"(R={R} per gamma, {len(GAMMAS)} gammas)")
    emit("=" * 88)

    logdir = '/home/lyra/projects/evalue-sheaf/code/evalue/logs'
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, 'mixwin_components_regime1.log'), 'w') as f:
        f.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    main()
