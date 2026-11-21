"""
reproduce_mixwin_regime2.py — INDEPENDENT reproduction of Clio's "MixWin" result at
higher replication count (R >= 2000), so it can be stated as a paper result rather than
resting on her R=500 runs.

CONSTRUCTION (support-agnostic contiguous-window mixture):
  MixWin is a MIXTURE e-process: the equal-prior (1/9) average of the wealth processes
  of 9 component bettors.  Averaging e-processes with a fixed prior preserves validity
  (the average of e-processes is an e-process), so MixWin is valid by construction.

  The 9 components (Q=10 strata, 0-indexed 0..9):
    - 1 "Eq" component  : equal weights cap/Q across ALL strata  (= the Fixed bettor).
    - 8 "window" comps  : one per contiguous 3-stratum window {k,k+1,k+2}, bets cap/3
                          equally on its 3 strata, zero elsewhere.  For Q=10 there are
                          Q-2 = 8 windows: 1-indexed {1,2,3}..{8,9,10}, i.e. 0-indexed
                          {0,1,2}..{7,8,9}.

  Each component weight vector satisfies sum_s lambda_s = cap = 1/b, so every linear-bet
  factor 1 + <lambda, X_m> stays >= 0 (same feasibility as Fixed).  MixWin's wealth is the
  average of the 9 component wealth PROCESSES (not an average of weights).

CONTRASTS:
    Eq     = Fixed, equal weights everywhere (support-blind baseline; dilutes in Regime 2).
    Top3   = ORACLE bettor cap/3 on the TRUE support strata (0-indexed 7,8,9) — upper bound.
    MixWin = the mixture above (does NOT know the support).

DGPs (reused verbatim from the harness):
    static null       : dgp.null_dgp                       (Type-I check)
    Regime 2 alt      : alt_dgp_top_strata from run_equicorr_puzzle (shock only in the
                        top-3 difficulty strata, 0-indexed 7,8,9), at gamma=0.20, 0.35.

All bettors share the SAME per-seed increment stream X[m,s] (built by
build_increment_stream, within_block / fresh-pair, second_order delta — the SHIPPED
construction on this branch).  Threshold 1/alpha = 20, alpha = 0.05.

SEED BASE: 50000 + r, distinct from the existing harness seeds (1000+r, 42+...).
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/lyra/projects/evalue-sheaf/code/evalue')

from dgp import null_dgp
from eprocess import ville_reject, THRESHOLD
import run_online_weights as rw
from run_online_weights import build_increment_stream, wealth_path
from run_equicorr_puzzle import alt_dgp_top_strata

# ── Settings ────────────────────────────────────────────────────────────────────
Q      = 10
BLOCK  = 20
N      = 4000
R      = 2000          # >= 2000 (Clio used 500)
ALPHA  = 0.05
THRESH = 20.0          # 1/alpha
SEED_BASE = 50000      # distinct from existing harness seeds

# keep imported machinery's module globals consistent
rw.Q = Q
rw.BLOCK = BLOCK

# ── Shipped build arguments on this branch (fresh-pair, cyclic, second-order delta) ──
BUILD_KW = dict(pairing='within_block', slack='second_order', pairing_rule='cyclic')


# ── Component weight vectors (all sum to cap = 1/b) ──────────────────────────────
def eq_weights(b):
    cap = 1.0 / b
    return np.full(Q, cap / Q)

def window_weights(b, k0):
    """cap/3 on 0-indexed strata {k0, k0+1, k0+2}, 0 elsewhere."""
    cap = 1.0 / b
    w = np.zeros(Q)
    w[k0:k0 + 3] = cap / 3.0
    return w

def top3_weights(b, support=(7, 8, 9)):
    """ORACLE: cap/3 on the TRUE support strata (0-indexed 7,8,9)."""
    cap = 1.0 / b
    w = np.zeros(Q)
    for s in support:
        w[s] = cap / 3.0
    return w


def mixwin_components(b):
    """9 components: Eq + 8 windows {0,1,2}..{7,8,9} (0-indexed). Returns list of (Q,) vecs."""
    comps = [eq_weights(b)]
    n_windows = Q - 2            # = 8 for Q=10 ; = #strata - 2 in general
    for k0 in range(n_windows):  # k0 = 0..7  -> windows {0,1,2}..{7,8,9}
        comps.append(window_weights(b, k0))
    assert len(comps) == 9, f"expected 9 components for Q=10, got {len(comps)}"
    return comps


# ── Wealth processes ─────────────────────────────────────────────────────────────
def eq_wealth(X, b):
    return wealth_path(X, eq_weights(b))

def top3_wealth(X, b):
    return wealth_path(X, top3_weights(b))

def mixwin_wealth(X, b):
    """Mixture e-process: equal-prior (1/9) average of the 9 component wealth PROCESSES."""
    comps = mixwin_components(b)
    paths = np.array([wealth_path(X, w) for w in comps])   # (9, M)
    return paths.mean(axis=0)                              # (M,)


# ── One paired replication ───────────────────────────────────────────────────────
def paired_run(data, rng):
    X, b = build_increment_stream(data, rng, **BUILD_KW)
    rej_eq, _  = ville_reject(eq_wealth(X, b),     THRESH)
    rej_t3, _  = ville_reject(top3_wealth(X, b),   THRESH)
    rej_mw, _  = ville_reject(mixwin_wealth(X, b), THRESH)
    return rej_eq, rej_t3, rej_mw


def run_condition(make_data, label):
    n_eq = n_t3 = n_mw = 0
    for r in range(R):
        rng = np.random.default_rng(SEED_BASE + r)
        data = make_data(N, rng)
        e, t, m = paired_run(data, rng)
        n_eq += int(e); n_t3 += int(t); n_mw += int(m)
    return dict(label=label, Eq=n_eq / R, Top3=n_t3 / R, MixWin=n_mw / R)


def main():
    t0 = time.time()
    print("=" * 84)
    print("MixWin reproduction — support-agnostic contiguous-window mixture e-process")
    print(f"  Q={Q}, N={N}, BLOCK={BLOCK} (M={N // BLOCK} blocks), R={R}, alpha={ALPHA}, "
          f"thresh={THRESH}")
    print(f"  seed base {SEED_BASE}+r  |  build={BUILD_KW}")
    print(f"  MixWin = (1/9) mean of [Eq + 8 windows {{0,1,2}}..{{7,8,9}}] (0-indexed)")
    print(f"  Top3 (oracle) support = 0-indexed strata 7,8,9  (= Regime-2 shock strata)")
    print("=" * 84)

    results = []
    # static null -> Type-I
    results.append(run_condition(lambda Nn, rng: null_dgp(Nn, rng),
                                 "static null (Type-I)"))
    # Regime 2 alternative -> power
    for g in (0.20, 0.35):
        results.append(run_condition(lambda Nn, rng, gg=g: alt_dgp_top_strata(Nn, gg, rng),
                                     f"Regime 2  gamma={g:.2f} (power)"))

    print(f"\n  {'condition':>26} | {'Eq':>8} {'Top3':>8} {'MixWin':>8}")
    print("  " + "-" * 58)
    for row in results:
        print(f"  {row['label']:>26} | {row['Eq']:>8.3f} {row['Top3']:>8.3f} {row['MixWin']:>8.3f}")

    # validity check
    null_row = results[0]
    print("\n  VALIDITY (static null, must be <= alpha=0.05):")
    for name in ('Eq', 'Top3', 'MixWin'):
        ok = "OK" if null_row[name] <= ALPHA else "EXCEEDS alpha"
        print(f"    {name:>6}: Type-I = {null_row[name]:.3f}  [{ok}]")

    print(f"\n  TOTAL RUNTIME: {time.time() - t0:.1f}s   (R={R})")
    print("=" * 84)


if __name__ == '__main__':
    main()
