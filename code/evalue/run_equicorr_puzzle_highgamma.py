"""
run_equicorr_puzzle_highgamma.py — FOCUSED high-gamma sweep for Regimes 1 & 2 only.

The main run_equicorr_puzzle.py gamma grid tops out at gamma=0.15 (empirical phi_bar=0.31).
To reach phi_bar ~ 0.40 we extend the grid upward and re-run ONLY the two SIGNAL regimes
(R1 homogeneous, R2 heterogeneous). Regime 3 (validity) is unchanged and not re-run here.

This script does NOT modify run_equicorr_puzzle.py. It imports that module's already-reviewed
machinery and the DGP factories, swaps the module-level GAMMA_GRID to the high-gamma grid
(run_signal_regime reads GAMMA_GRID as a module global), and calls run_signal_regime for R1/R2.
Same N, R, Q, BLOCK, threshold, seeding and paired structure as the main script.
"""

import time
import numpy as np

import run_equicorr_puzzle as m
from run_equicorr_puzzle import alt_dgp, alt_dgp_top_strata

# High-gamma grid chosen to bracket phi_bar ~ 0.40 (and a bit beyond). The main grid's
# top point gamma=0.15 reached phi_bar=0.31; these should carry it through 0.40.
HIGH_GAMMA_GRID = [0.15, 0.20, 0.25, 0.30, 0.35]

# Keep N, R identical to the main script (N=4000, R=200).
assert m.N == 4000 and m.R == 200, f"unexpected N/R: N={m.N} R={m.R}"


def main():
    t0 = time.time()
    print("=" * 96)
    print("HIGH-GAMMA SWEEP — ONLINE (ONS) vs EQUAL (FIXED) weights, §5b combiner  [R1 & R2 only]")
    print(f"  Q={m.Q} strata, BLOCK={m.BLOCK} (M={m.N//m.BLOCK} blocks), threshold={m.THRESH}, alpha={m.ALPHA}")
    print(f"  N={m.N}, R={m.R}  |  paired within replication (same data for FIXED/ONLINE/ORACLE)")
    print(f"  high gamma grid: {HIGH_GAMMA_GRID}")
    print(f"  wilcoxon available: {m.HAVE_WILCOXON}")
    print("=" * 96)

    # Swap the module-level grid that run_signal_regime reads.
    m.GAMMA_GRID = HIGH_GAMMA_GRID

    # Regime 1: homogeneous (uniform shock) — alt_dgp
    r1_factory = lambda g: (lambda Nn, rng: alt_dgp(Nn, g, rng))
    m.run_signal_regime(
        r1_factory,
        "REGIME 1 — HOMOGENEOUS weak signal (uniform shock; puzzle home turf)  [HIGH GAMMA]",
    )

    # Regime 2: heterogeneous (top-30% strata shock)
    r2_factory = lambda g: (lambda Nn, rng: alt_dgp_top_strata(Nn, g, rng))
    m.run_signal_regime(
        r2_factory,
        "REGIME 2 — HETEROGENEOUS weak signal (shock in top-30% difficulty strata)  [HIGH GAMMA]",
    )

    elapsed = time.time() - t0
    print(f"\n{'='*96}")
    print(f"TOTAL RUNTIME: {elapsed:.1f}s   (N={m.N}, R={m.R})")
    print(f"{'='*96}")


if __name__ == '__main__':
    main()
