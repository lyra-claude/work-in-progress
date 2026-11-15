"""
run_equicorr_puzzle.py — ADVERSARIAL TEST of the ONLINE (ONS) weight-learner against
the plain EQUAL-weights (FIXED) baseline for the §5b joint linear bet, framed as the
econometrician's FORECAST COMBINATION PUZZLE.

THE PUZZLE
----------
A 50-year result in forecast combination: when you combine correlated, weak signals,
ESTIMATING optimal combination weights frequently loses to naive EQUAL weights
out-of-sample, because estimation noise swamps the (small) gain. The effect is most
brutal under EQUICORRELATION — where the optimal weights ARE equal, so any learner can
only hurt by chasing noise away from uniform.

We coded an Online-Newton-Step weight learner (ons_weights, ONLINE) for the stratified
co-failure combiner
    M_n = prod_m ( 1 + sum_s lambda_{s,m} X_{s,m} ),   lambda_s >= 0, sum_s lambda_s <= 1/b.
Before anyone publishes "adaptive weights", we check adversarially whether ONLINE actually
beats FIXED (equal 1/b per Q strata) in the regimes that matter. PAIRED comparison: within
each replication FIXED, ONLINE and ORACLE see the SAME increment stream (same data, same
past-only pairing), so differences are not Monte-Carlo noise between schemes.

THREE REGIMES
-------------
R1  HOMOGENEOUS weak signal — alt_dgp with a UNIFORM Marshall-Olkin shock gamma. All strata
    carry ~equal expected co-failure => the oracle weights are ~uniform => this is the
    puzzle's home turf: ONLINE can essentially only lose.
R2  HETEROGENEOUS weak signal — the shock is concentrated in the TOP 30% difficulty strata
    (bins 7,8,9 of Q=10), zero elsewhere. Now the oracle weights are genuinely non-uniform,
    so there is a real learning target ONLINE could exploit to beat FIXED.
R3  VALIDITY under an EQUICORRELATION NULL — a shared latent factor shifts ALL items'
    difficulty up/down per replication (conditional independence given difficulty is
    PRESERVED, so E[X_s] <= 0 — no true co-failure — but the strata increments co-move
    across the stream). Plus the existing drift_null. ONLINE's adaptivity must NOT inflate
    Type-I above alpha.

We reuse the already-reviewed machinery in run_online_weights.py (build_increment_stream,
lam_fixed, ons_weights, oracle_weights, wealth_path) and only add the two new DGP variants
needed for R2 and R3. numpy + scipy only.
"""

import csv
import sys
import time
import numpy as np

sys.path.insert(0, '/home/lyra/projects/evalue-sheaf/code/evalue')

from dgp import judge_probs, alt_dgp, drift_null_dgp
from eprocess import ville_reject, THRESHOLD
import run_online_weights as rw
from run_online_weights import (
    build_increment_stream,
    lam_fixed,
    ons_weights,
    oracle_weights,
    wealth_path,
)

try:
    from scipy.stats import wilcoxon
    HAVE_WILCOXON = True
except Exception:                                   # pragma: no cover
    HAVE_WILCOXON = False


# ── Settings ────────────────────────────────────────────────────────────────────
Q         = 10
BLOCK     = 20
THRESH    = 20.0
ALPHA     = 0.05
GAMMA_GRID = [0.02, 0.04, 0.06, 0.10, 0.15]

# First cut was N=2000,R=150 and ran in ~27s total, so we bump to the fuller grid.
N = 4000
R = 200

# Keep the imported machinery's module globals consistent with ours. Q and BLOCK are read
# as module globals INSIDE build_increment_stream / lam_fixed / ons_weights, so set them.
rw.Q = Q
rw.BLOCK = BLOCK


# ── New DGP variant for R2: shock ONLY in the top-30% difficulty strata ───────────
def alt_dgp_top_strata(N, gamma, rng, Q=Q, top_bins=3):
    """
    Identical to alt_dgp EXCEPT the Marshall-Olkin shock only fires for items whose
    difficulty theta lands in the top `top_bins` strata of Q equal-width bins
    (theta in top 30% for Q=10, top_bins=3). Items outside those bins get no shock
    (pure H0 conditional-independent draws). This makes the oracle weights genuinely
    non-uniform (mass should concentrate on the high-difficulty strata).
    """
    theta = rng.uniform(0.0, 1.0, size=N)
    p_i, p_j = judge_probs(theta)
    # stratum index 0..Q-1 by equal-width theta bins; top_bins highest are shock-eligible
    strata = np.clip((theta * Q).astype(int), 0, Q - 1)
    eligible = strata >= (Q - top_bins)
    shock = (rng.uniform(size=N) < gamma) & eligible
    W_i = np.where(shock, 1, (rng.uniform(size=N) < p_i).astype(int))
    W_j = np.where(shock, 1, (rng.uniform(size=N) < p_j).astype(int))
    return dict(theta=theta, W_i=W_i, W_j=W_j, p_i=p_i, p_j=p_j, shock=shock)


# ── New DGP variant for R3: equicorrelation NULL via a shared latent difficulty shift ──
def latent_factor_null_dgp(N, rng):
    """
    H0 (conditional independence given difficulty) with a per-replication SHARED LATENT
    factor z ~ U(-0.15, +0.15) that coherently shifts EVERY item's difficulty:
        theta_i = clip(base_theta_i + z, 0, 1).
    Judges then fail independently given theta (exactly as null_dgp). So E[X_s] <= 0 — no
    excess co-failure — but because z moves all items together, the per-stratum increments
    CO-MOVE across the stream (equicorrelation across strata). This is the adversarial
    validity stress for an ADAPTIVE weight learner: cross-stratum correlation with no signal.
    """
    base_theta = rng.uniform(0.0, 1.0, size=N)
    z = rng.uniform(-0.15, 0.15)
    theta = np.clip(base_theta + z, 0.0, 1.0)
    p_i, p_j = judge_probs(theta)
    W_i = (rng.uniform(size=N) < p_i).astype(int)
    W_j = (rng.uniform(size=N) < p_j).astype(int)
    return dict(theta=theta, W_i=W_i, W_j=W_j, p_i=p_i, p_j=p_j)


# ── Empirical pairwise error correlation phi_bar between the two judges ────────────
def phi_bar_pearson(data):
    """Pearson corr of (W_i, W_j) over all items; 0 if either is constant."""
    Wi = data['W_i'].astype(float)
    Wj = data['W_j'].astype(float)
    if Wi.std() < 1e-12 or Wj.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(Wi, Wj)[0, 1])


# ── One paired replication on a single data dict ──────────────────────────────────
def paired_run(data, rng):
    """
    Build the increment stream ONCE, then evaluate FIXED, ONLINE, ORACLE on that SAME
    stream. Returns per-scheme terminal log-wealth + (rejected, first_t) detection.
    """
    X, b = build_increment_stream(data, rng)

    lf = lam_fixed(b)
    lam_on_seq, _ = ons_weights(X, b)
    lo = oracle_weights(X, b)

    M_fx = wealth_path(X, lf)
    M_on = wealth_path(X, lam_on_seq)
    M_or = wealth_path(X, lo)

    def tlw(Mp):
        return float(np.log(max(Mp[-1], 1e-300)))

    rej_fx, t_fx = ville_reject(M_fx, THRESH)
    rej_on, t_on = ville_reject(M_on, THRESH)
    rej_or, t_or = ville_reject(M_or, THRESH)

    return dict(
        lw_fixed=tlw(M_fx), lw_online=tlw(M_on), lw_oracle=tlw(M_or),
        rej_fixed=rej_fx, t_fixed=t_fx,
        rej_online=rej_on, t_online=t_on,
        rej_oracle=rej_or, t_oracle=t_or,
    )


# ── Paired statistics (ONLINE vs FIXED), computed on PER-SEED differences ──────────
def paired_stats(online_vals, fixed_vals):
    """
    online_vals, fixed_vals : arrays of paired per-seed values (log-wealth OR detect time).
    Returns (median_diff, win_frac, p_value, n_used). Wilcoxon signed-rank (two-sided) on the
    differences; falls back to an exact binomial sign test if wilcoxon is unavailable or all
    differences are zero. win_frac = fraction of seeds with online > fixed.
    """
    d = np.asarray(online_vals, float) - np.asarray(fixed_vals, float)
    n = len(d)
    if n == 0:
        return (float('nan'), float('nan'), float('nan'), 0)
    med = float(np.median(d))
    win = float(np.mean(d > 0))
    nz = d[d != 0]
    if len(nz) == 0:
        return (med, win, 1.0, n)                   # perfectly tied
    if HAVE_WILCOXON:
        try:
            _, p = wilcoxon(d, zero_method='wilcox', alternative='two-sided')
            return (med, win, float(p), n)
        except Exception:
            pass
    # sign-test fallback (two-sided exact binomial on sign of nonzero diffs)
    from math import comb
    k = int(np.sum(nz > 0)); m = len(nz)
    tail = sum(comb(m, i) for i in range(0, min(k, m - k) + 1)) * 0.5 ** m
    return (med, win, float(min(1.0, 2.0 * tail)), n)


# ── Detection-time paired stats, restricted to seeds where BOTH detected ──────────
def detect_time_paired(rows):
    both = [(r['t_online'], r['t_fixed']) for r in rows
            if r['rej_online'] and r['rej_fixed']]
    if not both:
        return (float('nan'), float('nan'), float('nan'), 0)
    on = np.array([a for a, _ in both], float)
    fx = np.array([b for _, b in both], float)
    return paired_stats(on, fx)


# ── Run one regime (a DGP factory parameterized by gamma) over the gamma grid ─────
def run_signal_regime(dgp_factory, label):
    """
    dgp_factory(gamma) -> a function f(N, rng) producing one data dict.
    For each gamma, R paired replications, each seeded default_rng(1000+r) so FIXED,
    ONLINE, ORACLE share the SAME data within a replication.
    Returns (summary_rows, paired_rows) for printing / CSV.
    """
    summ_rows = []
    paired_rows = []
    print(f"\n{'='*96}\n{label}\n{'='*96}")
    header = (f"  {'gamma':>6} {'phi_bar':>8} | "
              f"{'FX_lw':>8} {'FX_pow':>7} {'FX_t50':>7} | "
              f"{'ON_lw':>8} {'ON_pow':>7} {'ON_t50':>7} | "
              f"{'OR_lw':>8} {'OR_pow':>7} {'OR_t50':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for gamma in GAMMA_GRID:
        make = dgp_factory(gamma)
        rows = []
        phis = []
        for r in range(R):
            rng = np.random.default_rng(1000 + r)
            data = make(N, rng)
            phis.append(phi_bar_pearson(data))
            rows.append(paired_run(data, rng))

        lw_fx = np.array([r['lw_fixed'] for r in rows])
        lw_on = np.array([r['lw_online'] for r in rows])
        lw_or = np.array([r['lw_oracle'] for r in rows])
        pow_fx = np.mean([r['rej_fixed'] for r in rows])
        pow_on = np.mean([r['rej_online'] for r in rows])
        pow_or = np.mean([r['rej_oracle'] for r in rows])

        def med_t(key):
            v = [r[key] for r in rows if r[key] is not None]
            return int(np.median(v)) if v else None

        t50_fx, t50_on, t50_or = med_t('t_fixed'), med_t('t_online'), med_t('t_oracle')
        phi_bar = float(np.mean(phis))

        print(f"  {gamma:>6.2f} {phi_bar:>8.3f} | "
              f"{lw_fx.mean():>+8.3f} {pow_fx:>7.3f} {str(t50_fx):>7} | "
              f"{lw_on.mean():>+8.3f} {pow_on:>7.3f} {str(t50_on):>7} | "
              f"{lw_or.mean():>+8.3f} {pow_or:>7.3f} {str(t50_or):>7}")

        # paired ONLINE vs FIXED on per-seed log-wealth
        med_d, win, pval, npar = paired_stats(lw_on, lw_fx)
        # paired detection time among seeds where BOTH detected
        dmed, dwin, dpval, dn = detect_time_paired(rows)

        summ_rows.append(dict(
            regime=label, gamma=gamma, phi_bar=phi_bar,
            fixed_lw=lw_fx.mean(), fixed_pow=pow_fx, fixed_t50=t50_fx,
            online_lw=lw_on.mean(), online_pow=pow_on, online_t50=t50_on,
            oracle_lw=lw_or.mean(), oracle_pow=pow_or, oracle_t50=t50_or,
        ))
        paired_rows.append(dict(
            regime=label, gamma=gamma,
            lw_median_diff=med_d, lw_win_frac=win, lw_wilcoxon_p=pval, n=npar,
            dt_median_diff=dmed, dt_win_frac=dwin, dt_wilcoxon_p=dpval, dt_n=dn,
        ))

    # paired stats block
    print(f"\n  PAIRED  ONLINE - FIXED  (per-seed; +median => ONLINE better):")
    print(f"    {'gamma':>6} | {'lw_med_diff':>12} {'lw_win':>7} {'lw_p':>9} | "
          f"{'dt_med_diff':>12} {'dt_win':>7} {'dt_p':>9} {'dt_n':>5}")
    for pr in paired_rows:
        dtmed = pr['dt_median_diff']
        dtmed_s = f"{dtmed:+.2f}" if dtmed == dtmed else "   nan"
        print(f"    {pr['gamma']:>6.2f} | "
              f"{pr['lw_median_diff']:>+12.4f} {pr['lw_win_frac']:>7.3f} {pr['lw_wilcoxon_p']:>9.2e} | "
              f"{dtmed_s:>12} {pr['dt_win_frac']:>7.3f} {pr['dt_wilcoxon_p']:>9.2e} {pr['dt_n']:>5}")

    return summ_rows, paired_rows


# ── Regime 3: validity under equicorrelation / drift nulls ────────────────────────
def run_validity_regime():
    print(f"\n{'='*96}\nREGIME 3 — VALIDITY under equicorrelation & drift NULLS "
          f"(Type-I must be <= {ALPHA})\n{'='*96}")
    nulls = [
        ("latent-factor equicorr null", latent_factor_null_dgp),
        ("drift_null",                  drift_null_dgp),
    ]
    rows = []
    print(f"  {'null':>30} | {'FIXED Type-I':>12} {'ONLINE Type-I':>13} {'ORACLE Type-I':>13}")
    print("  " + "-" * 74)
    for name, dgp_fn in nulls:
        rej_fx = rej_on = rej_or = 0
        for r in range(R):
            rng = np.random.default_rng(1000 + r)
            data = dgp_fn(N, rng)
            res = paired_run(data, rng)
            rej_fx += int(res['rej_fixed'])
            rej_on += int(res['rej_online'])
            rej_or += int(res['rej_oracle'])
        t_fx, t_on, t_or = rej_fx / R, rej_on / R, rej_or / R
        flag = "" if (t_fx <= ALPHA and t_on <= ALPHA) else "  <-- EXCEEDS alpha"
        print(f"  {name:>30} | {t_fx:>12.3f} {t_on:>13.3f} {t_or:>13.3f}{flag}")
        rows.append(dict(regime="R3-validity", null=name,
                         fixed_typeI=t_fx, online_typeI=t_on, oracle_typeI=t_or))
    return rows


def main():
    t0 = time.time()
    print("=" * 96)
    print("FORECAST-COMBINATION-PUZZLE TEST — ONLINE (ONS) vs EQUAL (FIXED) weights, §5b combiner")
    print(f"  Q={Q} strata, BLOCK={BLOCK} (M={N//BLOCK} blocks), threshold={THRESH}, alpha={ALPHA}")
    print(f"  N={N}, R={R}  |  paired within replication (same data for FIXED/ONLINE/ORACLE)")
    print(f"  wilcoxon available: {HAVE_WILCOXON}")
    print("=" * 96)

    # Regime 1: homogeneous (uniform shock) — alt_dgp
    r1_factory = lambda g: (lambda Nn, rng: alt_dgp(Nn, g, rng))
    s1, p1 = run_signal_regime(r1_factory, "REGIME 1 — HOMOGENEOUS weak signal (uniform shock; puzzle home turf)")

    # Regime 2: heterogeneous (top-30% strata shock)
    r2_factory = lambda g: (lambda Nn, rng: alt_dgp_top_strata(Nn, g, rng))
    s2, p2 = run_signal_regime(r2_factory, "REGIME 2 — HETEROGENEOUS weak signal (shock in top-30% difficulty strata)")

    # Regime 3: validity
    s3 = run_validity_regime()

    elapsed = time.time() - t0
    print(f"\n{'='*96}")
    print(f"TOTAL RUNTIME: {elapsed:.1f}s   (N={N}, R={R})")
    print(f"{'='*96}")

    # ── CSVs ──────────────────────────────────────────────────────────────────────
    d = '/home/lyra/projects/evalue-sheaf/code/evalue/'
    with open(d + 'results_equicorr_puzzle_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(s1[0].keys()))
        w.writeheader(); w.writerows(s1 + s2)
    with open(d + 'results_equicorr_puzzle_paired.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(p1[0].keys()))
        w.writeheader(); w.writerows(p1 + p2)
    with open(d + 'results_equicorr_puzzle_validity.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(s3[0].keys()))
        w.writeheader(); w.writerows(s3)
    print("CSVs written: results_equicorr_puzzle_{summary,paired,validity}.csv")
    print("Done.")


if __name__ == '__main__':
    main()
