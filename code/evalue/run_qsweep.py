"""
run_qsweep.py — Stratification sweep for Method A.

Sweeps Q ∈ {5,10,20,40,80} strata (equal-width in theta ~ U(0,1)).
For each Q reports:
  - Type-I error on NULL DGP
  - Power on ALT gamma=0.05 and ALT gamma=0.10
  - Median detection time on ALT gamma=0.10 (among runs that fire)
  - Realized-bias diagnostic: eps_bar, delta=2*eps_bar**2, mean within-stratum bias, ratio

Fixed params: N=2000, R=500, alpha=0.05, threshold=20, seed=42.
Lambda modes: mixture over {0.1,...,0.9} AND fixed lambda=0.5.

Saves results_qsweep.csv and optionally power_vs_q.png.
"""

import csv
import sys
import numpy as np

from dgp import null_dgp, alt_dgp
from eprocess import (
    assign_strata, stratum_epsilon, make_within_block_pairing,
    make_theta_neighbour_block_pairing,
    accumulate_wealth, accumulate_wealth_mixture, ville_reject, THRESHOLD,
    BLOCK,
)

# ── Parameters ──────────────────────────────────────────────────────────────────
N       = 2000
R       = 500
ALPHA   = 0.05
SEED    = 42
Q_GRID  = [5, 10, 20, 40, 80]


# ── Method A with variable Q ─────────────────────────────────────────────────────

def method_A_Q(data: dict, rng: np.random.Generator, Q: int,
               pairing: str = 'within_block') -> dict:
    """
    Method A parameterised with a given number of strata Q and a pairing mode.

    pairing='within_block'  : (block, stratum) fresh-pair cyclic roll (the OLD map that
                              starves at large Q — expected items/cell = BLOCK/Q).
    pairing='theta_neighbour': theta-neighbour within-block pairing (Clio's repair):
                              sort each block by theta, pair adjacent items -> BLOCK/2
                              pairs regardless of Q; per-pair Prop-1 slack; bet assigned
                              to the midpoint-theta stratum.  Q enters ONLY the midpoint
                              bin assignment (diagnostic), not the pair count.

    Both use the SAME product-over-items wealth with per-item positivity clip
    lam <= 1/(1+2*eps**2); each paired item contributes one factor, sentinels contribute 1.
    Returns M_fixed (lambda=0.5) and M_mix.
    """
    W_i, W_j = data['W_i'], data['W_j']
    p_i, p_j = data['p_i'], data['p_j']
    theta     = data['theta']
    N_items   = len(W_i)
    block_ids = np.arange(N_items) // BLOCK

    if pairing == 'within_block':
        strata       = assign_strata(theta, Q=Q)
        eps_k        = stratum_epsilon(p_i, p_j, strata, Q=Q)
        t            = make_within_block_pairing(strata, block_ids, Q=Q)
        eps_per_item = eps_k[strata]
        delta        = 2.0 * eps_per_item ** 2   # QUADRATIC slack (1/2 R^2 = 2 eps^2)
    elif pairing == 'theta_neighbour':
        t, delta, eps_per_item, _ = make_theta_neighbour_block_pairing(
            theta, p_i, p_j, block_ids, Q=Q)
    else:
        raise ValueError(f"unknown pairing mode: {pairing!r}")

    U            = W_i * W_j
    V            = W_i[np.arange(N_items)] * W_j[t]
    # Sentinel items (t[s]==s) get delta=0 and eps=0 for theta_neighbour, and for
    # within_block they bet a self-pair where U-V=0; either way they contribute ~factor 1.

    M_fixed = accumulate_wealth(U, V, delta, lam=0.5, clip_lam=True,
                                eps_per_item=eps_per_item)
    M_mix   = accumulate_wealth_mixture(U, V, delta, clip_lam=True,
                                        eps_per_item=eps_per_item)

    return dict(M_fixed=M_fixed, M_mix=M_mix,
                eps_per_item=eps_per_item, delta_per_item=delta)


# ── Realized-bias measurement on NULL data ────────────────────────────────────────

def realized_bias_for_Q(Q: int, seed: int, R_diag: int = 200) -> dict:
    """
    On the NULL DGP, empirically measure the within-stratum bias
    = E[U - V] per stratum (should be ~0 under H0, but within-stratum
    variation in p(theta) means a single cross-pair can have nonzero mean).

    Returns eps_bar (mean eps over strata), delta=2*eps_bar**2 (quadratic slack),
    realized_bias (mean over strata of per-stratum |E[U-V]| on null data),
    and the ratio delta/realized_bias.
    """
    rng = np.random.default_rng(seed + 7777)

    # Collect per-stratum (U-V) means over R_diag runs
    # We want E[U_s - V_s | stratum k] estimated on null data.
    # For efficiency: run a single long dataset (N*R_diag items) and compute by strata.
    N_big = N * R_diag
    theta_all = rng.uniform(0.0, 1.0, size=N_big)

    # Use judge_probs from dgp without importing the full module
    from dgp import null_dgp as _null_dgp, judge_probs as _judge_probs
    p_i_all, p_j_all = _judge_probs(theta_all)
    W_i_all = (rng.uniform(size=N_big) < p_i_all).astype(int)
    W_j_all = (rng.uniform(size=N_big) < p_j_all).astype(int)

    strata_all = assign_strata(theta_all, Q=Q)
    # Build the SAME fresh-pair within-block partner map used by method_A_Q, so the
    # realized-bias diagnostic measures E[U - V] under the exact pairing the e-process
    # bets on (not an ad-hoc whole-stratum np.roll).
    block_ids_all = np.arange(N_big) // BLOCK
    t_all = make_within_block_pairing(strata_all, block_ids_all, Q=Q)
    has_partner = (t_all != np.arange(N_big))
    U_all = W_i_all * W_j_all
    V_all = W_i_all * W_j_all[t_all]

    eps_k   = np.zeros(Q)
    bias_k  = np.zeros(Q)

    for k in range(Q):
        mask = strata_all == k
        n_k  = mask.sum()
        if n_k < 2:
            eps_k[k]  = 0.0
            bias_k[k] = 0.0
            continue
        eps_k[k] = max(
            (p_i_all[mask].max() - p_i_all[mask].min()) / 2.0,
            (p_j_all[mask].max() - p_j_all[mask].min()) / 2.0,
        )
        # within-stratum cross-item bias, over items that have a fresh-pair partner
        cell = mask & has_partner
        if cell.sum() == 0:
            bias_k[k] = 0.0
            continue
        bias_k[k] = np.abs((U_all[cell] - V_all[cell]).mean())

    eps_bar       = float(eps_k.mean())
    delta_bar     = 2.0 * eps_bar ** 2    # QUADRATIC slack (1/2 R^2 = 2 eps^2)
    bias_bar      = float(bias_k.mean())
    ratio         = delta_bar / bias_bar if bias_bar > 1e-12 else float('inf')

    return dict(
        Q=Q,
        eps_bar=eps_bar,
        delta=delta_bar,
        realized_bias=bias_bar,
        ratio=ratio,
    )


# ── MC runner for one Q ───────────────────────────────────────────────────────────

def run_condition_Q(dgp_fn, Q: int, R: int, N: int, seed: int) -> dict:
    rng          = np.random.default_rng(seed)
    reject_fixed = []
    reject_mix   = []
    det_fixed    = []
    det_mix      = []

    for _ in range(R):
        data = dgp_fn(N, rng)
        res  = method_A_Q(data, rng, Q=Q)

        r_f, t_f = ville_reject(res['M_fixed'])
        r_m, t_m = ville_reject(res['M_mix'])

        reject_fixed.append(r_f)
        reject_mix.append(r_m)
        det_fixed.append(t_f + 1 if t_f is not None else None)
        det_mix.append(t_m + 1 if t_m is not None else None)

    return dict(
        reject_fixed=np.array(reject_fixed),
        reject_mix=np.array(reject_mix),
        det_fixed=det_fixed,
        det_mix=det_mix,
    )


def median_det(det_list):
    fired = [t for t in det_list if t is not None]
    if not fired:
        return None
    return int(np.median(fired))


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    print(f"\nQ-SWEEP: Method A only, N={N}, R={R}, alpha={ALPHA}, threshold={THRESHOLD}")
    print(f"Q grid: {Q_GRID}\n")

    dgp_null    = null_dgp
    dgp_alt005  = lambda N, rng: alt_dgp(N, 0.05, rng)
    dgp_alt010  = lambda N, rng: alt_dgp(N, 0.10, rng)

    sweep_rows  = []   # for the main table
    bias_rows   = []   # for the realized-bias table

    for Q in Q_GRID:
        items_per_stratum = N / Q
        print(f"  Q={Q:>3}  (N/Q = {items_per_stratum:.0f} items/stratum)")

        # --- Type-I ---
        res_null = run_condition_Q(dgp_null, Q=Q, R=R, N=N, seed=SEED)
        t1_fixed = float(res_null['reject_fixed'].mean())
        t1_mix   = float(res_null['reject_mix'].mean())
        flag_t1  = ""
        if t1_fixed > ALPHA or t1_mix > ALPHA:
            flag_t1 = "  *** TYPE-I VIOLATION ***"
        print(f"         Type-I: fixed={t1_fixed:.4f}  mix={t1_mix:.4f}{flag_t1}")

        # --- Power @ gamma=0.05 ---
        res_005 = run_condition_Q(dgp_alt005, Q=Q, R=R, N=N, seed=SEED + 1)
        pw05_fixed = float(res_005['reject_fixed'].mean())
        pw05_mix   = float(res_005['reject_mix'].mean())
        print(f"         Power@0.05: fixed={pw05_fixed:.4f}  mix={pw05_mix:.4f}")

        # --- Power @ gamma=0.10 ---
        res_010 = run_condition_Q(dgp_alt010, Q=Q, R=R, N=N, seed=SEED + 2)
        pw10_fixed = float(res_010['reject_fixed'].mean())
        pw10_mix   = float(res_010['reject_mix'].mean())
        md10_fixed = median_det(res_010['det_fixed'])
        md10_mix   = median_det(res_010['det_mix'])
        print(f"         Power@0.10: fixed={pw10_fixed:.4f}  mix={pw10_mix:.4f}")
        print(f"         MedDet@0.10: fixed={md10_fixed}  mix={md10_mix}")

        sweep_rows.append(dict(
            Q=Q,
            items_per_stratum=items_per_stratum,
            t1_fixed=t1_fixed,
            t1_mix=t1_mix,
            pw05_fixed=pw05_fixed,
            pw05_mix=pw05_mix,
            pw10_fixed=pw10_fixed,
            pw10_mix=pw10_mix,
            md10_fixed=md10_fixed,
            md10_mix=md10_mix,
        ))

        # --- Realized-bias diagnostic ---
        print(f"         Computing realized-bias diagnostic ...", flush=True)
        bd = realized_bias_for_Q(Q, seed=SEED)
        bias_rows.append(bd)
        print(f"         eps_bar={bd['eps_bar']:.5f}  delta={bd['delta']:.5f}  "
              f"realized_bias={bd['realized_bias']:.5f}  ratio={bd['ratio']:.2f}")
        print()

    # ── Print sweep table ─────────────────────────────────────────────────────────
    print()
    print("=" * 110)
    print("SWEEP TABLE — Method A (rows = Q, columns = mixture / fixed-lambda=0.5)")
    print(f"  N={N}, R={R}, alpha={ALPHA}, threshold={THRESHOLD}")
    print("=" * 110)
    hdr = (
        f"{'Q':>4}  {'N/Q':>5}  "
        f"{'TypeI(fix)':>10}  {'TypeI(mix)':>10}  "
        f"{'Pw@0.05(fix)':>12}  {'Pw@0.05(mix)':>12}  "
        f"{'Pw@0.10(fix)':>12}  {'Pw@0.10(mix)':>12}  "
        f"{'MedDet@0.10(fix)':>17}  {'MedDet@0.10(mix)':>17}"
    )
    print(hdr)
    print("-" * 110)
    for row in sweep_rows:
        md_f = str(row['md10_fixed']) if row['md10_fixed'] is not None else "—"
        md_m = str(row['md10_mix'])   if row['md10_mix']   is not None else "—"
        flag = ""
        if row['t1_fixed'] > ALPHA or row['t1_mix'] > ALPHA:
            flag = "  *** T1 VIOLATION ***"
        print(
            f"{row['Q']:>4}  {row['items_per_stratum']:>5.0f}  "
            f"{row['t1_fixed']:>10.4f}  {row['t1_mix']:>10.4f}  "
            f"{row['pw05_fixed']:>12.4f}  {row['pw05_mix']:>12.4f}  "
            f"{row['pw10_fixed']:>12.4f}  {row['pw10_mix']:>12.4f}  "
            f"{md_f:>17}  {md_m:>17}{flag}"
        )
    print()

    # ── Print bias table ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("REALIZED-BIAS TABLE — NULL DGP, per-stratum diagnostic")
    print("=" * 70)
    bh = f"{'Q':>4}  {'eps_bar':>8}  {'delta=2eps^2':>12}  {'real_bias':>10}  {'ratio':>7}"
    print(bh)
    print("-" * 70)
    for bd in bias_rows:
        ratio_str = f"{bd['ratio']:.2f}" if bd['ratio'] < 1e9 else "inf"
        print(
            f"{bd['Q']:>4}  {bd['eps_bar']:>8.5f}  {bd['delta']:>10.5f}  "
            f"{bd['realized_bias']:>10.5f}  {ratio_str:>7}"
        )
    print()

    # ── Find the knee ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("KNEE ANALYSIS")
    print("=" * 70)

    knee_010_mix   = None
    knee_010_fixed = None
    knee_005_mix   = None
    knee_005_fixed = None

    for row in sweep_rows:
        if knee_010_mix is None and row['t1_mix'] <= ALPHA and row['pw10_mix'] >= 0.80:
            knee_010_mix = row['Q']
        if knee_010_fixed is None and row['t1_fixed'] <= ALPHA and row['pw10_fixed'] >= 0.80:
            knee_010_fixed = row['Q']
        if knee_005_mix is None and row['t1_mix'] <= ALPHA and row['pw05_mix'] >= 0.80:
            knee_005_mix = row['Q']
        if knee_005_fixed is None and row['t1_fixed'] <= ALPHA and row['pw05_fixed'] >= 0.80:
            knee_005_fixed = row['Q']

    def _knee_str(knee, label, best_row_col, rows):
        if knee is not None:
            return f"  Smallest Q reaching Type-I<=0.05 AND Power>=0.80 at {label}: Q={knee}"
        else:
            best = max(rows, key=lambda r: r[best_row_col])
            return (f"  No Q in {Q_GRID} reaches Power>=0.80 at {label}. "
                    f"Best power achieved: {best[best_row_col]:.4f} at Q={best['Q']}")

    print(_knee_str(knee_010_mix,   "gamma=0.10 (mixture)",   "pw10_mix",   sweep_rows))
    print(_knee_str(knee_010_fixed, "gamma=0.10 (fixed lam)", "pw10_fixed", sweep_rows))
    print(_knee_str(knee_005_mix,   "gamma=0.05 (mixture)",   "pw05_mix",   sweep_rows))
    print(_knee_str(knee_005_fixed, "gamma=0.05 (fixed lam)", "pw05_fixed", sweep_rows))
    print()

    # Type-I sanity check
    print("TYPE-I SANITY (all Q, both lambda modes):")
    any_violation = False
    for row in sweep_rows:
        if row['t1_fixed'] > ALPHA or row['t1_mix'] > ALPHA:
            print(f"  *** ANOMALY at Q={row['Q']}: Type-I fixed={row['t1_fixed']:.4f} "
                  f"mix={row['t1_mix']:.4f}  (N/Q={row['items_per_stratum']:.0f})")
            any_violation = True
    if not any_violation:
        print("  All Q: Type-I controlled <= 0.05 for both lambda modes. Validity holds.")
    print()

    # ── Write CSV ────────────────────────────────────────────────────────────────
    csv_path = '/home/lyra/projects/evalue-sheaf/code/evalue/results_qsweep.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Q', 'items_per_stratum',
            'TypeI_fixed', 'TypeI_mix',
            'Power_005_fixed', 'Power_005_mix',
            'Power_010_fixed', 'Power_010_mix',
            'MedDet010_fixed', 'MedDet010_mix',
            'eps_bar', 'delta', 'realized_bias', 'ratio',
        ])
        for row, bd in zip(sweep_rows, bias_rows):
            writer.writerow([
                row['Q'],
                f"{row['items_per_stratum']:.0f}",
                f"{row['t1_fixed']:.4f}",
                f"{row['t1_mix']:.4f}",
                f"{row['pw05_fixed']:.4f}",
                f"{row['pw05_mix']:.4f}",
                f"{row['pw10_fixed']:.4f}",
                f"{row['pw10_mix']:.4f}",
                row['md10_fixed'] if row['md10_fixed'] is not None else '',
                row['md10_mix']   if row['md10_mix']   is not None else '',
                f"{bd['eps_bar']:.5f}",
                f"{bd['delta']:.5f}",
                f"{bd['realized_bias']:.5f}",
                f"{bd['ratio']:.3f}" if bd['ratio'] < 1e9 else 'inf',
            ])
    print(f"CSV written to {csv_path}")

    # ── Power-vs-Q PNG ───────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        Qs       = [r['Q'] for r in sweep_rows]
        pw10_m   = [r['pw10_mix']   for r in sweep_rows]
        pw10_f   = [r['pw10_fixed'] for r in sweep_rows]
        pw05_m   = [r['pw05_mix']   for r in sweep_rows]
        pw05_f   = [r['pw05_fixed'] for r in sweep_rows]
        t1_m     = [r['t1_mix']     for r in sweep_rows]
        t1_f     = [r['t1_fixed']   for r in sweep_rows]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.plot(Qs, pw10_m,   'o-', color='firebrick', label='Power@0.10 (mix)')
        ax.plot(Qs, pw10_f,   's--', color='firebrick', alpha=0.6, label='Power@0.10 (fixed λ=0.5)')
        ax.plot(Qs, pw05_m,   'o-', color='steelblue', label='Power@0.05 (mix)')
        ax.plot(Qs, pw05_f,   's--', color='steelblue', alpha=0.6, label='Power@0.05 (fixed λ=0.5)')
        ax.axhline(0.80, color='gray', ls=':', lw=1, label='power=0.80 target')
        ax.set_xlabel('Number of strata Q')
        ax.set_ylabel('Power')
        ax.set_title('Method A — Power vs Q')
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8)
        ax.set_xticks(Qs)

        ax2 = axes[1]
        ax2.plot(Qs, t1_m,  'o-', color='purple', label='Type-I (mix)')
        ax2.plot(Qs, t1_f,  's--', color='purple', alpha=0.6, label='Type-I (fixed λ=0.5)')
        ax2.axhline(ALPHA, color='k', ls='--', lw=1, label=f'α={ALPHA}')
        ax2.set_xlabel('Number of strata Q')
        ax2.set_ylabel('Type-I error')
        ax2.set_title('Method A — Type-I vs Q')
        ax2.set_ylim(-0.005, 0.08)
        ax2.legend(fontsize=8)
        ax2.set_xticks(Qs)

        fig.suptitle(f'Stratification sweep — Method A  (N={N}, R={R}, α={ALPHA})', fontsize=11)
        fig.tight_layout()
        png_path = '/home/lyra/projects/evalue-sheaf/code/evalue/power_vs_q.png'
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"Power-vs-Q PNG saved to {png_path}")
    except ImportError:
        print("matplotlib not available — skipping PNG")

    print("\nDone.")


if __name__ == '__main__':
    main()
