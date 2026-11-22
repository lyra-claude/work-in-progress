"""
reproduce_mixwin_regime1.py — MixWin experiment on REGIME 1 (homogeneous/uniform shock).

Based on reproduce_mixwin_regime2.py but switches the DGP to alt_dgp (uniform Marshall-Olkin
shock across ALL strata), which is the regime where equal weights are already near-Kelly.
Also adds Type-I checks on the latent-factor null and drift null (Part 2 of Clio's request).

THE QUESTION (Clio's peer review):
  "What does support-agnosticism COST where equal weights are already near-Kelly?"
  i.e. how much power does MixWin give up relative to equal weights in Regime 1?

CONSTRUCTION (identical to reproduce_mixwin_regime2.py):
  MixWin = (1/9) average of 9 component wealth processes:
    - 1 Eq component: equal weights cap/Q across ALL strata
    - 8 window components: contiguous 3-stratum windows {0,1,2}..{7,8,9} (0-indexed)

DGPs:
  static null         : dgp.null_dgp
  latent-factor null  : latent_factor_null_dgp (from run_equicorr_puzzle)
  drift null          : drift_null_dgp (from dgp)
  Regime 1 alt        : alt_dgp (uniform shock at several gamma values)

SEED BASE: 50000 + r (same as reproduce_mixwin_regime2.py for direct comparability)
R = 2000
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/lyra/projects/evalue-sheaf/code/evalue')

from dgp import null_dgp, alt_dgp, drift_null_dgp
from eprocess import ville_reject, THRESHOLD
import run_online_weights as rw
from run_online_weights import build_increment_stream, wealth_path
from run_equicorr_puzzle import latent_factor_null_dgp

# ── Settings ────────────────────────────────────────────────────────────────────
Q      = 10
BLOCK  = 20
N      = 4000
R      = 2000          # same as reproduce_mixwin_regime2.py
ALPHA  = 0.05
THRESH = 20.0          # 1/alpha
SEED_BASE = 50000      # same as reproduce_mixwin_regime2.py for direct comparability

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


def mixwin_components(b):
    """9 components: Eq + 8 windows {0,1,2}..{7,8,9} (0-indexed). Returns list of (Q,) vecs."""
    comps = [eq_weights(b)]
    n_windows = Q - 2            # = 8 for Q=10
    for k0 in range(n_windows):  # k0 = 0..7  -> windows {0,1,2}..{7,8,9}
        comps.append(window_weights(b, k0))
    assert len(comps) == 9, f"expected 9 components for Q=10, got {len(comps)}"
    return comps


# ── Wealth processes ─────────────────────────────────────────────────────────────
def eq_wealth(X, b):
    return wealth_path(X, eq_weights(b))

def mixwin_wealth(X, b):
    """Mixture e-process: equal-prior (1/9) average of the 9 component wealth PROCESSES."""
    comps = mixwin_components(b)
    paths = np.array([wealth_path(X, w) for w in comps])   # (9, M)
    return paths.mean(axis=0)                              # (M,)


# ── One paired replication ───────────────────────────────────────────────────────
def paired_run(data, rng):
    X, b = build_increment_stream(data, rng, **BUILD_KW)
    rej_eq, _  = ville_reject(eq_wealth(X, b),     THRESH)
    rej_mw, _  = ville_reject(mixwin_wealth(X, b), THRESH)
    return rej_eq, rej_mw


def run_condition(make_data, label):
    n_eq = n_mw = 0
    for r in range(R):
        rng = np.random.default_rng(SEED_BASE + r)
        data = make_data(N, rng)
        e, m = paired_run(data, rng)
        n_eq += int(e); n_mw += int(m)
    return dict(label=label, Eq=n_eq / R, MixWin=n_mw / R)


def main():
    t0 = time.time()
    print("=" * 84)
    print("MixWin on REGIME 1 — support-agnosticism cost where equal weights are near-Kelly")
    print(f"  Q={Q}, N={N}, BLOCK={BLOCK} (M={N // BLOCK} blocks), R={R}, alpha={ALPHA}, "
          f"thresh={THRESH}")
    print(f"  seed base {SEED_BASE}+r  |  build={BUILD_KW}")
    print(f"  MixWin = (1/9) mean of [Eq + 8 windows {{0,1,2}}..{{7,8,9}}] (0-indexed)")
    print(f"  DGP: alt_dgp = uniform Marshall-Olkin shock across ALL strata (Regime 1)")
    print("=" * 84)

    results = []

    # ── PART 1: Power on Regime 1 across gamma grid ──────────────────────────────
    gamma_grid = [0.15, 0.20, 0.25, 0.30, 0.35]
    for g in gamma_grid:
        results.append(run_condition(lambda Nn, rng, gg=g: alt_dgp(Nn, gg, rng),
                                     f"Regime 1  gamma={g:.2f} (power)"))

    print(f"\n  PART 1 — Power on Regime 1 (alt_dgp, uniform shock)")
    print(f"  {'condition':>26} | {'Eq':>8} {'MixWin':>8} {'gap (EW-MW)':>12}")
    print("  " + "-" * 62)
    for row in results:
        gap = row['Eq'] - row['MixWin']
        print(f"  {row['label']:>26} | {row['Eq']:>8.3f} {row['MixWin']:>8.3f} {gap:>+12.3f}")

    # ── PART 2: Type-I on static, latent, drift nulls ───────────────────────────
    print(f"\n  PART 2 — Type-I error completeness (all should be <= alpha={ALPHA})")
    null_results = []

    # static null (Type-I check)
    null_results.append(run_condition(lambda Nn, rng: null_dgp(Nn, rng),
                                      "static null"))
    # latent-factor null
    null_results.append(run_condition(lambda Nn, rng: latent_factor_null_dgp(Nn, rng),
                                      "latent-factor null"))
    # drift null
    null_results.append(run_condition(lambda Nn, rng: drift_null_dgp(Nn, rng),
                                      "drift null"))

    print(f"  {'null':>26} | {'Eq Type-I':>10} {'MixWin Type-I':>14} {'status':>12}")
    print("  " + "-" * 68)
    for row in null_results:
        eq_ok  = "OK" if row['Eq']     <= ALPHA else "EXCEEDS alpha"
        mw_ok  = "OK" if row['MixWin'] <= ALPHA else "EXCEEDS alpha"
        status = "OK" if (row['Eq'] <= ALPHA and row['MixWin'] <= ALPHA) else "EXCEEDS alpha"
        print(f"  {row['label']:>26} | {row['Eq']:>10.3f} {row['MixWin']:>14.3f}   {status}")

    elapsed = time.time() - t0
    print(f"\n  TOTAL RUNTIME: {elapsed:.1f}s   (R={R})")
    print("=" * 84)


if __name__ == '__main__':
    main()
