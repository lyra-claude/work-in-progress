"""
run_experiment.py — Monte-Carlo driver for the co-failure e-process simulation.

Measures Type-I error (H0), power (H1 at gamma=0.05 and 0.10), and median
detection time for three methods × two lambda modes (fixed λ=0.5 vs mixture).

Also runs the optional drift experiment comparing Method A vs C under H0 with
linearly drifting marginals.

Prints a formatted results table, writes results.csv, and saves one trajectory
PNG (null + alt Method A paths) if matplotlib is available.
"""

import sys
import csv
import numpy as np

from dgp import null_dgp, alt_dgp, drift_null_dgp
from eprocess import method_A, method_B, method_C, ville_reject, THRESHOLD

# ── Parameters ─────────────────────────────────────────────────────────────────

N   = 2000   # items per run
R   = 500    # Monte-Carlo runs per condition
ALPHA = 0.05
SEED  = 42


# ── Monte-Carlo runner ─────────────────────────────────────────────────────────

def run_condition(dgp_fn, method_fn, R: int, N: int, seed: int) -> dict:
    """
    Run R independent simulations of a given DGP × method combination.
    Returns dict with:
        reject_fixed, reject_mix  — bool arrays of length R (did Ville fire?)
        det_fixed, det_mix        — detection times (item index + 1) or None
    """
    rng = np.random.default_rng(seed)
    reject_fixed = []
    reject_mix   = []
    det_fixed    = []
    det_mix      = []

    for _ in range(R):
        data = dgp_fn(N, rng)
        res  = method_fn(data, rng) if 'rng' in method_fn.__code__.co_varnames else method_fn(data)

        r_f, t_f = ville_reject(res['M_fixed'])
        r_m, t_m = ville_reject(res['M_mix'])

        reject_fixed.append(r_f)
        reject_mix.append(r_m)
        det_fixed.append(t_f + 1 if t_f is not None else None)   # 1-based item count
        det_mix.append(t_m + 1 if t_m is not None else None)

    return dict(
        reject_fixed=np.array(reject_fixed),
        reject_mix=np.array(reject_mix),
        det_fixed=det_fixed,
        det_mix=det_mix,
    )


def median_detection(det_list: list) -> str:
    """Median detection time among runs that fired; '—' if none fired."""
    fired = [t for t in det_list if t is not None]
    if not fired:
        return "—"
    return str(int(np.median(fired)))


def run_method(dgp_fn, method_obj, R: int, N: int, seed: int) -> dict:
    """Dispatch: handle methods that take rng and those that don't."""
    rng = np.random.default_rng(seed)
    reject_fixed = []
    reject_mix   = []
    det_fixed    = []
    det_mix      = []

    for _ in range(R):
        data = dgp_fn(N, rng)
        if method_obj is method_C_wrap:
            res = method_C_wrap(data, rng)
        else:
            res = method_obj(data, rng)

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


def method_C_wrap(data, rng):
    """Wrap method_C to have the same signature as A and B."""
    return method_C(data)


# ── Save one representative trajectory ─────────────────────────────────────────

def save_trajectory_png(seed: int, N: int):
    """Save a PNG of one null and one alt (gamma=0.10) wealth trajectory for Method A."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping PNG")
        return

    rng = np.random.default_rng(seed + 9999)   # fresh seed, not the MC seed

    data_null = null_dgp(N, rng)
    res_null  = method_A(data_null, rng)

    data_alt  = alt_dgp(N, 0.10, rng)
    res_alt   = method_A(data_alt, rng)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(res_null['M_fixed'], color='steelblue', lw=1.0, label='H0 (null), fixed λ=0.5')
    ax.semilogy(res_alt['M_fixed'],  color='firebrick', lw=1.0, label='H1 (γ=0.10), fixed λ=0.5')
    ax.axhline(THRESHOLD, color='k', ls='--', lw=0.8, label=f'threshold = {THRESHOLD}')
    ax.set_xlabel('Item index')
    ax.set_ylabel('Wealth M_t  (log scale)')
    ax.set_title('Method A — e-process wealth trajectory')
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = 'trajectory.png'
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Trajectory PNG saved to {path}")


# ── Main experiment ─────────────────────────────────────────────────────────────

def main():
    print(f"\nRunning Monte-Carlo experiment: N={N}, R={R}, alpha={ALPHA}, threshold={THRESHOLD}\n")

    # Condition seeds: offset by condition index so each gets independent randomness
    seeds = {
        'null':     SEED,
        'alt_005':  SEED + 1,
        'alt_010':  SEED + 2,
        'drift':    SEED + 3,
    }

    methods = {
        'A (stratified)':    method_A,
        'B (unstratified)':  method_B,
        'C (naive plug-in)': method_C_wrap,
    }

    dgps = {
        'NULL':         null_dgp,
        'ALT γ=0.05':   lambda N, rng: alt_dgp(N, 0.05, rng),
        'ALT γ=0.10':   lambda N, rng: alt_dgp(N, 0.10, rng),
    }
    dgp_seeds = ['null', 'alt_005', 'alt_010']

    results = []

    for method_name, method_fn in methods.items():
        for (dgp_name, dgp_fn), seed_key in zip(dgps.items(), dgp_seeds):
            print(f"  {method_name} × {dgp_name}  (seed={seeds[seed_key]}) ...", flush=True)
            res = run_method(dgp_fn, method_fn, R, N, seeds[seed_key])

            type1_or_power_fixed = res['reject_fixed'].mean()
            type1_or_power_mix   = res['reject_mix'].mean()
            med_fixed = median_detection(res['det_fixed'])
            med_mix   = median_detection(res['det_mix'])

            results.append({
                'method':       method_name,
                'dgp':          dgp_name,
                'rate_fixed':   type1_or_power_fixed,
                'rate_mix':     type1_or_power_mix,
                'med_fixed':    med_fixed,
                'med_mix':      med_mix,
            })

    # ── Drift experiment ────────────────────────────────────────────────────────
    print(f"  Drift experiment (A vs C, H0 with drifting marginals) ...", flush=True)

    drift_results = {}
    for mname, mfn in [('A (stratified)', method_A), ('C (naive plug-in)', method_C_wrap)]:
        res = run_method(drift_null_dgp, mfn, R, N, seeds['drift'])
        drift_results[mname] = res

    # ── Print results table ─────────────────────────────────────────────────────

    print()
    print("=" * 100)
    print("RESULTS TABLE")
    print(f"  N={N} items, R={R} runs, alpha={ALPHA}, threshold={THRESHOLD}")
    print("=" * 100)

    header = (
        f"{'Method':<22} {'DGP':<14} "
        f"{'TypeI/Power (fixed λ)':<22} {'Med det (fixed)':<16} "
        f"{'TypeI/Power (mixture)':<22} {'Med det (mix)':<14}"
    )
    print(header)
    print("-" * 100)

    last_method = None
    for row in results:
        m = row['method']
        if m != last_method and last_method is not None:
            print()
        last_method = m
        label = "Type-I" if "NULL" in row['dgp'] else "Power "
        print(
            f"{m:<22} {row['dgp']:<14} "
            f"{label}  {row['rate_fixed']:.3f}          {row['med_fixed']:<16} "
            f"{label}  {row['rate_mix']:.3f}          {row['med_mix']}"
        )

    print()
    print("=" * 100)
    print("DRIFT EXPERIMENT — Type-I error under drifting marginals (H0)")
    print("-" * 100)
    print(f"  (Marginals ramp linearly: base_i 0.20→0.45, base_j 0.25→0.50 over N items)")
    print(f"  {'Method':<22}  {'Type-I fixed λ':<18}  {'Type-I mixture'}")
    for mname, res in drift_results.items():
        print(
            f"  {mname:<22}  {res['reject_fixed'].mean():.3f}              "
            f"  {res['reject_mix'].mean():.3f}"
        )
    print("=" * 100)

    # ── Write CSV ───────────────────────────────────────────────────────────────
    csv_path = 'results.csv'
    fieldnames = ['method', 'dgp', 'rate_fixed', 'rate_mix', 'med_det_fixed', 'med_det_mix']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                'method':        row['method'],
                'dgp':           row['dgp'],
                'rate_fixed':    f"{row['rate_fixed']:.4f}",
                'rate_mix':      f"{row['rate_mix']:.4f}",
                'med_det_fixed': row['med_fixed'],
                'med_det_mix':   row['med_mix'],
            })
        # drift rows
        for mname, res in drift_results.items():
            writer.writerow({
                'method':        mname + ' [DRIFT]',
                'dgp':           'NULL (drift)',
                'rate_fixed':    f"{res['reject_fixed'].mean():.4f}",
                'rate_mix':      f"{res['reject_mix'].mean():.4f}",
                'med_det_fixed': '—',
                'med_det_mix':   '—',
            })
    print(f"\nCSV written to {csv_path}")

    # ── Save trajectory PNG ─────────────────────────────────────────────────────
    save_trajectory_png(SEED, N)


if __name__ == '__main__':
    main()
