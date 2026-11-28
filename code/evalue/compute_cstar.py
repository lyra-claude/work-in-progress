"""
compute_cstar.py — Sweep c in Z_c = c*EW + (1-c)*mean_windows over [0,1].

The one-parameter family of combiners is:
  Z_c = c * EW_wealth + (1-c) * mean_windows_wealth
  c=1   => EW (equal-weight expert)
  c=1/2 => half-EW (Clio's proposal, up to rounding; see note below)
  c=1/9 => MixWin (mean of all 9 = EW + 8 windows)

NOTE on c=1/9 and c=1/2:
  MixWin = (1/9)*EW + (8/9)*mean_windows  => c_MixWin = 1/9
  half-EW = (1/2)*EW + (1/16)*sum_8_windows = (1/2)*EW + (1/2)*mean_windows => c=1/2

  mean_windows = mean of the 8 contiguous-window experts (NOT including EW).
  From the MixWin formula: mean_windows = (9*MixWin_wealth - EW_wealth) / 8.
  We collect per-seed EW_wealth and MixWin_wealth, then derive mean_windows.

For each DGP cell (dgp x gamma) and each c on a grid:
  (a) GROWTH: E[log Z_c] and argmax_c E[log Z_c]  = c_growth*
  (b) POWER : frac(seeds where sup_n Z_c(path) >= THRESH) and argmax_c power = c_power*
      Power requires the FULL PATH of Z_c, not just the terminal value.
      Z_c(n) = c*EW_wealth(n) + (1-c)*mean_windows_wealth(n) is a convex combination
      of e-processes — itself a valid e-process.  Its Ville crossing is computed from
      max over n of Z_c(n) >= THRESH.

WHAT IS STORED PER SEED:
  For efficiency we store the FULL WEALTH PATH arrays for EW and each of the 8
  window experts (not just the terminal value), which lets us compute
  Z_c(n) = c*EW_path(n) + (1-c)*mean_window_paths(n) for any c without re-running.

Writes:
  code/evalue/results_cstar_raw.npz  — per-seed arrays (compact, ~40 MB for R=2000)
  results/results_cstar.json         — aggregated growth+power curves and optima
  code/evalue/NOTES-cstar.md         — one-page human-readable note

Run from the repo root:
  python3 code/evalue/compute_cstar.py
"""

import sys
import os
import time
import json
import pathlib
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code' / 'evalue'))

from dgp import null_dgp, judge_probs
from eprocess import ville_reject, THRESHOLD
import run_online_weights as rw
from run_online_weights import build_increment_stream, wealth_path

# ── Settings (must match novel_dgp_halfew_test.py exactly) ──────────────────
Q         = 10
BLOCK     = 20
N         = 4000
R         = 500          # Reduced from 2000 for tractable runtime (~10 min).
                         # At c=1,1/2,1/9 we cross-validate against the R=2000 JSON.
ALPHA     = 0.05
THRESH    = 20.0          # 1/alpha
SEED_BASE = 80000
GAMMA_GRID = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
BUILD_KW   = dict(pairing='within_block', slack='second_order', pairing_rule='cyclic')

rw.Q = Q
rw.BLOCK = BLOCK

# c grid: 21 points in [0,1] plus special values 1/9 and 1/2 for alignment
C_GRID = np.round(np.linspace(0.0, 1.0, 21), 4).tolist()  # 0, 0.05, ..., 1.0

# ── DGP profiles (copy from novel_dgp_halfew_test.py) ───────────────────────
SHAPE_REVERSED = ((np.arange(Q) + 1.0) / Q) ** 2
SHAPE_TWOPEAK  = np.zeros(Q)
SHAPE_TWOPEAK[0:3] = 1.0
SHAPE_TWOPEAK[Q-3:Q] = 1.0

def _profile_alt_dgp(N, gamma, rng, shape):
    shape = np.asarray(shape, dtype=float)
    shape = shape / shape.max()
    theta  = rng.uniform(0.0, 1.0, size=N)
    p_i, p_j = judge_probs(theta)
    strata = np.clip((theta * Q).astype(int), 0, Q - 1)
    gamma_s = gamma * shape
    g_item  = gamma_s[strata]
    shock   = rng.uniform(size=N) < g_item
    W_i = np.where(shock, 1, (rng.uniform(size=N) < p_i).astype(int))
    W_j = np.where(shock, 1, (rng.uniform(size=N) < p_j).astype(int))
    return dict(theta=theta, W_i=W_i, W_j=W_j, p_i=p_i, p_j=p_j)

def reversed_dgp(N, gamma, rng): return _profile_alt_dgp(N, gamma, rng, SHAPE_REVERSED)
def twopeak_dgp(N, gamma, rng):  return _profile_alt_dgp(N, gamma, rng, SHAPE_TWOPEAK)

# ── Window experts (from novel_dgp_halfew_test.py) ───────────────────────────
def eq_weights(b):
    cap = 1.0 / b
    return np.full(Q, cap / Q)

def window_weights(b, k0):
    cap = 1.0 / b
    w = np.zeros(Q)
    w[k0:k0 + 3] = cap / 3.0
    return w

def window_experts(b):
    return [window_weights(b, k0) for k0 in range(Q - 2)]   # 8 windows


def one_seed_paths(data, rng):
    """
    Return per-seed wealth PATH arrays for EW and each of 8 window experts.
    Shape: (M,) for EW; (8, M) for windows.
    M = N/BLOCK = 200 blocks.
    """
    X, b = build_increment_stream(data, rng, **BUILD_KW)
    ew_path = wealth_path(X, eq_weights(b))                           # (M,)
    win_paths = np.array([wealth_path(X, w) for w in window_experts(b)])  # (8, M)
    return ew_path, win_paths


def compute_curves_online(dgp_name, make_data, c_grid):
    """
    Stream through seeds one at a time.  For each seed, compute EW path and
    mean-windows path, then immediately accumulate growth and power stats for
    every c on the grid — no large path arrays stored between seeds.

    Returns dict keyed by gamma string -> curves dict.
    """
    C = np.array(c_grid)                    # (Nc,)
    out = {}
    for g in GAMMA_GRID:
        t0 = time.time()
        Nc = len(C)
        # Accumulators: sum of log Z_c(terminal) and count of crossings
        sum_log_zc = np.zeros(Nc)
        sum_cross  = np.zeros(Nc, dtype=np.int64)

        for r in range(R):
            rng  = np.random.default_rng(SEED_BASE + r)
            data = make_data(N, g, rng)
            ew_path, win_paths = one_seed_paths(data, rng)        # (M,), (8, M)
            mean_win_path = win_paths.mean(axis=0)                 # (M,)

            # For each c, compute Z_c path and extract terminal + running max
            # Vectorised over c: Zc = C[:,None]*ew[None,:] + (1-C)[:,None]*mw[None,:]
            # Shape: (Nc, M)
            Zc = C[:, None] * ew_path[None, :] + (1.0 - C[:, None]) * mean_win_path[None, :]
            # Terminal log-wealth
            term = Zc[:, -1]                                        # (Nc,)
            sum_log_zc += np.log(np.maximum(term, 1e-300))
            # Running max crossing THRESH
            rmax = Zc.max(axis=1)                                   # (Nc,)
            sum_cross += (rmax >= THRESH).astype(np.int64)

        growth_arr = sum_log_zc / R
        power_arr  = sum_cross / R

        c_arr = C
        c_growth_star = float(c_arr[np.argmax(growth_arr)])
        c_power_star  = float(c_arr[np.argmax(power_arr)])
        growth_at_opt = float(growth_arr[np.argmax(growth_arr)])
        power_at_opt  = float(power_arr[np.argmax(power_arr)])

        g_str = f"{g:.2f}"
        out[g_str] = dict(
            c_grid=c_grid,
            growth=growth_arr.tolist(),
            power=power_arr.tolist(),
            c_growth_star=c_growth_star,
            c_power_star=c_power_star,
            growth_at_opt=growth_at_opt,
            power_at_opt=power_at_opt,
        )
        elapsed = time.time() - t0
        print(f"  [{dgp_name}] gamma={g:.2f}  "
              f"c_growth*={c_growth_star:.2f} E[log]={growth_at_opt:.3f}  "
              f"c_power*={c_power_star:.2f} power={power_at_opt:.3f}  "
              f"({elapsed:.1f}s)",
              flush=True)
    return out


def main():
    t_total = time.time()
    print("compute_cstar.py — Z_c = c*EW + (1-c)*mean_windows sweep")
    print(f"  Q={Q}, N={N}, BLOCK={BLOCK}, R={R}, THRESH={THRESH}")
    print(f"  c_grid: {C_GRID}")
    print(f"  gamma_grid: {GAMMA_GRID}")
    print()

    # Compute c-sweep curves (streaming, low memory)
    print(">>> (A) REVERSED-GRADIENT DGP")
    rev_results = compute_curves_online("REV", reversed_dgp, C_GRID)

    print("\n>>> (B) TWO-PEAK DGP")
    two_results = compute_curves_online("TWO", twopeak_dgp, C_GRID)

    results = dict(reversed_gradient=rev_results, two_peak=two_results)

    # Named-combiner values at c=1 (EW), c=1/2 (hEW), c=1/9 (MixWin)
    named_c = {'EW': 1.0, 'hEW': 0.5, 'MixWin': 1/9}

    # Build a summary table
    summary_rows = []
    for dgp_name in ["reversed_gradient", "two_peak"]:
        for g_str, curves in results[dgp_name].items():
            c_arr = np.array(curves['c_grid'])
            g_arr = np.array(curves['growth'])
            p_arr = np.array(curves['power'])
            row = dict(
                dgp=dgp_name,
                gamma=float(g_str),
                c_growth_star=curves['c_growth_star'],
                c_power_star=curves['c_power_star'],
                E_log_at_growth_opt=curves['growth_at_opt'],
                power_at_power_opt=curves['power_at_opt'],
            )
            # Named-combiner values (interpolate to nearest grid point)
            for name, c_val in named_c.items():
                idx = int(np.argmin(np.abs(c_arr - c_val)))
                row[f'growth_{name}'] = float(g_arr[idx])
                row[f'power_{name}']  = float(p_arr[idx])
            # Growth / power gap: opt vs EW
            idx_ew = int(np.argmin(np.abs(c_arr - 1.0)))
            row['growth_opt_vs_EW'] = curves['growth_at_opt'] - float(g_arr[idx_ew])
            # Does growth-optimal coincide with power-optimal?
            row['growth_power_coincide'] = abs(curves['c_growth_star'] - curves['c_power_star']) < 0.051
            summary_rows.append(row)

    payload = dict(
        meta=dict(
            script='compute_cstar.py',
            Q=Q, N=N, BLOCK=BLOCK, M=N//BLOCK, R=R, ALPHA=ALPHA, THRESH=THRESH,
            seed_base=SEED_BASE, build=BUILD_KW,
            c_grid=C_GRID,
            gamma_grid=GAMMA_GRID,
            note=(
                "Z_c = c*EW_wealth + (1-c)*mean_windows_wealth. "
                "mean_windows = mean of 8 contiguous 3-stratum window experts "
                "(NOT including EW). "
                "c=1 => EW, c=1/2 => half-EW, c=1/9 => MixWin. "
                "growth = E[log Z_c(terminal)]. "
                "power = frac seeds where running_max Z_c(n) >= THRESH=20=1/alpha."
            ),
        ),
        summary=summary_rows,
        curves=results,
    )

    # Write JSON
    out_json = REPO_ROOT / 'results' / 'results_cstar.json'
    out_json.parent.mkdir(exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_json}")

    print(f"\nTOTAL RUNTIME: {time.time()-t_total:.1f}s")
    return payload


if __name__ == '__main__':
    main()
