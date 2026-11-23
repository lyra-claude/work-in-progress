"""
mixwin_perseed_regime1.py — PER-SEED continuous diagnostics for Eq vs MixWin
on the Regime-1 uniform-shock DGP (weak/unsaturated gamma band).

The weak-grid harness (reproduce_mixwin_regime1_weak.py) keeps ONLY the binary
Ville-reject indicator per seed and reports a reject-rate over R. This script
reuses the SAME DGP, SAME combiners, SAME BUILD_KW, SAME N/Q/BLOCK/THRESH/alpha,
and the SAME paired seed base (50000+r), but RETAINS the continuous per-seed
quantities for both arms:
    - terminal wealth M[-1]
    - terminal log-wealth log(M[-1])
    - running-max over the path
    - binary Ville reject indicator (ever crossed THRESH) + crossing index

PAIRED design: both arms see identical data per seed (same rng draw), exactly as
paired_run does in the weak-grid harness.

This script ONLY measures and prints numbers. It does not interpret them.
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/lyra/projects/evalue-sheaf/code/evalue')

# Reuse the EXISTING DGP + combiners + settings from the weak-grid harness.
from dgp import alt_dgp
from reproduce_mixwin_regime1_weak import (
    Q, BLOCK, N, ALPHA, THRESH, SEED_BASE, BUILD_KW,
    eq_wealth, mixwin_wealth,
)
from run_online_weights import build_increment_stream
from eprocess import ville_reject

# scipy.stats.wilcoxon — fall back to a manual signed-rank if unavailable.
_HAVE_SCIPY_WILCOXON = True
try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
except Exception:  # pragma: no cover
    _HAVE_SCIPY_WILCOXON = False


# ── Runtime settings ───────────────────────────────────────────────────────────
R = 600                     # paired seeds per gamma (drop to 400 if too slow)
GAMMAS = [0.05, 0.07]       # unsaturated band


# ── One paired replication, retaining continuous per-seed quantities ────────────
def paired_run_continuous(data, rng):
    """Both arms on identical data (same rng draw), exactly like paired_run."""
    X, b = build_increment_stream(data, rng, **BUILD_KW)

    M_eq = eq_wealth(X, b)
    M_mw = mixwin_wealth(X, b)

    rej_eq, cross_eq = ville_reject(M_eq, THRESH)
    rej_mw, cross_mw = ville_reject(M_mw, THRESH)

    return {
        'term_eq': float(M_eq[-1]),
        'term_mw': float(M_mw[-1]),
        'logterm_eq': float(np.log(M_eq[-1])),
        'logterm_mw': float(np.log(M_mw[-1])),
        'runmax_eq': float(M_eq.max()),
        'runmax_mw': float(M_mw.max()),
        'rej_eq': bool(rej_eq),
        'rej_mw': bool(rej_mw),
        'cross_eq': cross_eq,
        'cross_mw': cross_mw,
    }


def run_gamma(gamma, R):
    """Run R paired seeds at one gamma; return arrays of per-seed quantities."""
    term_eq = np.empty(R); term_mw = np.empty(R)
    log_eq = np.empty(R);  log_mw = np.empty(R)
    runmax_eq = np.empty(R); runmax_mw = np.empty(R)
    rej_eq = np.empty(R, dtype=bool); rej_mw = np.empty(R, dtype=bool)

    for r in range(R):
        rng = np.random.default_rng(SEED_BASE + r)
        data = alt_dgp(N, gamma, rng)
        res = paired_run_continuous(data, rng)
        term_eq[r] = res['term_eq']; term_mw[r] = res['term_mw']
        log_eq[r] = res['logterm_eq']; log_mw[r] = res['logterm_mw']
        runmax_eq[r] = res['runmax_eq']; runmax_mw[r] = res['runmax_mw']
        rej_eq[r] = res['rej_eq']; rej_mw[r] = res['rej_mw']

    return dict(term_eq=term_eq, term_mw=term_mw, log_eq=log_eq, log_mw=log_mw,
                runmax_eq=runmax_eq, runmax_mw=runmax_mw,
                rej_eq=rej_eq, rej_mw=rej_mw)


# ── Statistics helpers ──────────────────────────────────────────────────────────
def _manual_signed_rank_p(diffs):
    """Two-sided Wilcoxon signed-rank p (normal approx, zeros dropped, tie-corrected).
    Used only when scipy is unavailable."""
    d = diffs[diffs != 0.0]
    n = len(d)
    if n == 0:
        return float('nan'), 0
    ranks = _average_ranks(np.abs(d))
    W_plus = ranks[d > 0].sum()
    mean_W = n * (n + 1) / 4.0
    # tie correction for the variance
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = np.sum(counts**3 - counts)
    var_W = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var_W <= 0:
        return float('nan'), n
    z = (W_plus - mean_W) / np.sqrt(var_W)
    # two-sided normal approx
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2.0))
    return float(p), n


def _average_ranks(a):
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def cliffs_delta(x, y):
    """Cliff's delta for paired difference sign is not standard; here we use the
    per-seed paired form: delta = (#(x>y) - #(x<y)) / n over paired seeds."""
    gt = np.sum(x > y)
    lt = np.sum(x < y)
    n = len(x)
    return (gt - lt) / n


def pct(a, q):
    return float(np.percentile(a, q))


def summarize(name, gamma, D, lines):
    def emit(s):
        print(s); lines.append(s)

    emit("=" * 84)
    emit(f"gamma = {gamma:.2f}   (R = {len(D['rej_eq'])} paired seeds, "
         f"seed base {SEED_BASE}+r, build={BUILD_KW})")
    emit("=" * 84)

    # 1. reject rates
    rr_eq = float(D['rej_eq'].mean())
    rr_mw = float(D['rej_mw'].mean())
    emit("[1] Reject rate (fraction ever-crossing THRESH=%.1f)" % THRESH)
    emit(f"      Eq     : {rr_eq:.4f}  ({int(D['rej_eq'].sum())}/{len(D['rej_eq'])})")
    emit(f"      MixWin : {rr_mw:.4f}  ({int(D['rej_mw'].sum())}/{len(D['rej_mw'])})")

    # 2. terminal log-wealth summaries
    emit("[2] Terminal log-wealth  (mean, median, sd, p10, p90)")
    for arm, key in (("Eq", 'log_eq'), ("MixWin", 'log_mw')):
        a = D[key]
        emit(f"      {arm:<7}: mean={a.mean():+.4f}  median={np.median(a):+.4f}  "
             f"sd={a.std(ddof=1):.4f}  p10={pct(a,10):+.4f}  p90={pct(a,90):+.4f}")

    # 3. terminal wealth (linear)
    emit("[3] Terminal wealth, linear scale  (mean, median)")
    for arm, key in (("Eq", 'term_eq'), ("MixWin", 'term_mw')):
        a = D[key]
        emit(f"      {arm:<7}: mean={a.mean():.4f}  median={np.median(a):.4f}")

    # 4. paired comparison of terminal log-wealth (MixWin - Eq)
    diff = D['log_mw'] - D['log_eq']
    med_diff = float(np.median(diff))
    frac_mw_gt = float(np.mean(D['log_mw'] > D['log_eq']))
    delta = cliffs_delta(D['log_mw'], D['log_eq'])
    if _HAVE_SCIPY_WILCOXON:
        nz = diff[diff != 0.0]
        if len(nz) > 0:
            try:
                _, pval = _scipy_wilcoxon(nz)
            except Exception as e:
                pval = float('nan')
                emit(f"      (scipy wilcoxon raised: {e})")
        else:
            pval = float('nan')
        src = "scipy.stats.wilcoxon"
        n_used = int(len(nz))
    else:
        pval, n_used = _manual_signed_rank_p(diff)
        src = "manual signed-rank (normal approx) — scipy.stats.wilcoxon unavailable"

    emit("[4] Paired terminal log-wealth difference  (MixWin - Eq, per seed)")
    emit(f"      median paired diff : {med_diff:+.4f}")
    emit(f"      Wilcoxon p-value   : {pval:.3e}   [{src}; n_nonzero={n_used}]")
    emit(f"      Cliff's delta      : {delta:+.4f}   (paired sign form)")
    emit(f"      frac(MixWin > Eq)  : {frac_mw_gt:.4f}")

    # 5. 2x2 paired reject table
    both = int(np.sum(D['rej_eq'] & D['rej_mw']))
    only_eq = int(np.sum(D['rej_eq'] & ~D['rej_mw']))
    only_mw = int(np.sum(~D['rej_eq'] & D['rej_mw']))
    neither = int(np.sum(~D['rej_eq'] & ~D['rej_mw']))
    emit("[5] 2x2 paired reject table")
    emit(f"      both reject   : {both}")
    emit(f"      only Eq       : {only_eq}")
    emit(f"      only MixWin   : {only_mw}")
    emit(f"      neither       : {neither}")
    emit("")

    return rr_eq, rr_mw


def main():
    t0 = time.time()
    lines = []

    def emit(s):
        print(s); lines.append(s)

    emit("#" * 84)
    emit("MixWin vs Eq — PER-SEED continuous diagnostics on Regime-1 uniform shock")
    emit(f"  Q={Q}, N={N}, BLOCK={BLOCK} (M={N // BLOCK} blocks), alpha={ALPHA}, "
         f"THRESH={THRESH}")
    emit(f"  DGP: alt_dgp (uniform Marshall-Olkin shock across all strata)")
    emit(f"  combiners: eq_wealth / mixwin_wealth from reproduce_mixwin_regime1_weak")
    emit(f"  scipy.stats.wilcoxon available: {_HAVE_SCIPY_WILCOXON}")
    emit("#" * 84)
    emit("")

    sanity = {}
    for g in GAMMAS:
        tg = time.time()
        D = run_gamma(g, R)
        rr_eq, rr_mw = summarize("", g, D, lines)
        sanity[g] = (rr_eq, rr_mw)
        emit(f"  (gamma={g:.2f} runtime: {time.time() - tg:.1f}s)")
        emit("")

    # Sanity check vs known weak-grid values.
    expected = {0.05: (0.247, 0.394), 0.07: (0.773, 0.902)}
    emit("=" * 84)
    emit("SANITY CHECK vs known weak-grid reject rates (R=2000 reference)")
    emit("=" * 84)
    for g in GAMMAS:
        if g in expected:
            e_eq, e_mw = expected[g]
            o_eq, o_mw = sanity[g]
            emit(f"  gamma={g:.2f}: Eq observed={o_eq:.3f} expected~{e_eq:.3f} "
                 f"(|d|={abs(o_eq - e_eq):.3f}) | "
                 f"MixWin observed={o_mw:.3f} expected~{e_mw:.3f} "
                 f"(|d|={abs(o_mw - e_mw):.3f})")
    emit("")
    emit(f"TOTAL RUNTIME: {time.time() - t0:.1f}s   (R={R} per gamma, "
         f"{len(GAMMAS)} gammas)")
    emit("=" * 84)

    # Write results log.
    import os
    logdir = '/home/lyra/projects/evalue-sheaf/code/evalue/logs'
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, 'mixwin_perseed_regime1.log'), 'w') as f:
        f.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    main()
