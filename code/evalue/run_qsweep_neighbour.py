"""
run_qsweep_neighbour.py — Starvation-fix comparison: within-block vs theta-neighbour.

Runs the SAME Q-sweep {5,10,20,40,80} and SAME config/seeds as run_qsweep.py, for BOTH
pairing modes:
  - 'within_block'   : the OLD (block, stratum) fresh-pair map that STARVES at large Q.
  - 'theta_neighbour': Clio's repair — sort each block by theta, pair adjacent -> BLOCK/2
                       pairs regardless of Q; per-pair Prop-1 slack; bet -> midpoint stratum.

Reports, per Q:  power_old, power_new, TypeI_old, TypeI_new   (headline = mixture power
at the ALT used by the starvation result, gamma=0.10).  Also the gamma=0.05 power and a
NULL Type-I check for the new method across all Q.

Config (identical to run_qsweep.py): N=2000, R=500, alpha=0.05, threshold=20, seed=42.
Writes results_qsweep_neighbour.json and a log.
"""

import json
import numpy as np

from dgp import null_dgp, alt_dgp
from eprocess import ville_reject, THRESHOLD
from run_qsweep import method_A_Q

N      = 2000
R      = 500
ALPHA  = 0.05
SEED   = 42
Q_GRID = [5, 10, 20, 40, 80]


def run_condition(dgp_fn, Q, pairing, R, N, seed):
    """Return (reject_fixed_rate, reject_mix_rate) over R Monte-Carlo runs."""
    rng = np.random.default_rng(seed)
    rej_fixed, rej_mix = [], []
    for _ in range(R):
        data = dgp_fn(N, rng)
        res = method_A_Q(data, rng, Q=Q, pairing=pairing)
        r_f, _ = ville_reject(res['M_fixed'])
        r_m, _ = ville_reject(res['M_mix'])
        rej_fixed.append(r_f)
        rej_mix.append(r_m)
    return float(np.mean(rej_fixed)), float(np.mean(rej_mix))


def main():
    dgp_null   = null_dgp
    dgp_alt005 = lambda N, rng: alt_dgp(N, 0.05, rng)
    dgp_alt010 = lambda N, rng: alt_dgp(N, 0.10, rng)

    rows = []
    print(f"\nSTARVATION-FIX COMPARISON  N={N} R={R} alpha={ALPHA} thr={THRESHOLD} seed={SEED}")
    print(f"Q grid: {Q_GRID}   (BLOCK=20 => theta-neighbour gives 10 pairs/block for all Q)\n")

    for Q in Q_GRID:
        row = {'Q': Q, 'items_per_cell_old': 20 / Q}
        for pairing, tag in [('within_block', 'old'), ('theta_neighbour', 'new')]:
            # SAME seeds as run_qsweep.py for each DGP condition
            t1_f, t1_m   = run_condition(dgp_null,   Q, pairing, R, N, SEED)
            p05_f, p05_m = run_condition(dgp_alt005, Q, pairing, R, N, SEED + 1)
            p10_f, p10_m = run_condition(dgp_alt010, Q, pairing, R, N, SEED + 2)
            row[f'typeI_fixed_{tag}'] = t1_f
            row[f'typeI_mix_{tag}']   = t1_m
            row[f'pw05_fixed_{tag}']  = p05_f
            row[f'pw05_mix_{tag}']    = p05_m
            row[f'pw10_fixed_{tag}']  = p10_f
            row[f'pw10_mix_{tag}']    = p10_m
        rows.append(row)
        print(f"  Q={Q:>3} (old cell={20/Q:.2f} items)  "
              f"Pw@.10 mix: old={row['pw10_mix_old']:.3f} new={row['pw10_mix_new']:.3f} | "
              f"TypeI mix: old={row['typeI_mix_old']:.4f} new={row['typeI_mix_new']:.4f}")

    # ── headline table (mixture, gamma=0.10) ──────────────────────────────────────
    print("\n" + "=" * 88)
    print("HEADLINE TABLE — mixture wealth, ALT gamma=0.10 (the starvation curve)")
    print("=" * 88)
    print(f"{'Q':>4}  {'power_old':>10}  {'power_new':>10}  {'TypeI_old':>10}  {'TypeI_new':>10}")
    print("-" * 88)
    for r in rows:
        v = "  <-- violation" if (r['typeI_mix_new'] > ALPHA or r['typeI_mix_old'] > ALPHA) else ""
        print(f"{r['Q']:>4}  {r['pw10_mix_old']:>10.4f}  {r['pw10_mix_new']:>10.4f}  "
              f"{r['typeI_mix_old']:>10.4f}  {r['typeI_mix_new']:>10.4f}{v}")

    print("\n" + "=" * 88)
    print("FIXED-lambda=0.5 TABLE — ALT gamma=0.10")
    print("=" * 88)
    print(f"{'Q':>4}  {'power_old':>10}  {'power_new':>10}  {'TypeI_old':>10}  {'TypeI_new':>10}")
    print("-" * 88)
    for r in rows:
        print(f"{r['Q']:>4}  {r['pw10_fixed_old']:>10.4f}  {r['pw10_fixed_new']:>10.4f}  "
              f"{r['typeI_fixed_old']:>10.4f}  {r['typeI_fixed_new']:>10.4f}")

    print("\n" + "=" * 88)
    print("NULL TYPE-I CHECK (new method) — both lambda modes, all Q (must be <= alpha)")
    print("=" * 88)
    print(f"{'Q':>4}  {'TypeI_fixed_new':>16}  {'TypeI_mix_new':>14}")
    print("-" * 88)
    worst = 0.0
    for r in rows:
        worst = max(worst, r['typeI_fixed_new'], r['typeI_mix_new'])
        print(f"{r['Q']:>4}  {r['typeI_fixed_new']:>16.4f}  {r['typeI_mix_new']:>14.4f}")
    print(f"\nWorst-case new-method Type-I across all Q/modes: {worst:.4f}  "
          f"({'OK' if worst <= ALPHA else 'VIOLATION'} vs alpha={ALPHA})")

    out = {
        'config': dict(N=N, R=R, alpha=ALPHA, seed=SEED, threshold=THRESHOLD,
                       Q_grid=Q_GRID, block=20, dgp='alt_dgp Marshall-Olkin atom'),
        'rows': rows,
        'worst_typeI_new': worst,
    }
    path = '/home/lyra/projects/evalue-sheaf/code/evalue/results_qsweep_neighbour.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON written to {path}")


if __name__ == '__main__':
    main()
