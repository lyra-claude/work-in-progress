"""
novel_dgp_halfew_test.py — Stress-test Clio's half-EW combiner on TWO NOVEL DGPs
                           it was NOT tuned on.

CONTEXT
-------
We combine 9 "window" wealth processes into a single §5b joint-linear-bet e-process.
Three combiners are compared (all share the SAME per-block, per-stratum increment
stream X[m,s] built by run_online_weights.build_increment_stream, fresh-pair /
within_block / second_order / cyclic — the shipped Prop.3 build on this branch):

  EW      : equal weights — the EW *window expert* (cap/Q on every stratum).
            Its wealth process is wealth_path(X, eq_weights(b)).
  MixWin  : Bayes mixture (equal 1/9 prior) over the 9 experts
            [ EW + 8 contiguous 3-strata windows {0,1,2}..{7,8,9} ],
            combined as the (1/9) AVERAGE of the 9 component WEALTH PROCESSES
            (a predictable linear bet; NOT Cover's continuum).
  half-EW : Clio's proposal — weight 1/2 on the EW expert's wealth process and
            1/16 on each of the 8 non-trivial window experts' wealth processes.
            Mixture weights sum to 1/2 + 8*(1/16) = 1.  Claim: terminal log-wealth
            regret of half-EW vs EW is <= log 2 (0.693 nats) BY CONSTRUCTION, because
            half-EW >= (1/2)*EW pointwise so log(half-EW) >= log(EW) - log 2.

All combiners are mixtures of e-PROCESSES (averages of wealth paths with constant,
data-independent mixture weights), hence themselves valid e-processes: their Type-I
is bounded by alpha for ANY DGP as long as each component is a valid e-process under
H0 (guaranteed by the fresh-pair build).  So the Type-I check below is a consistency
check, not a tuning knob.

THE NOVEL DGPs
--------------
Both novel DGPs use a STRATUM-DEPENDENT Marshall-Olkin shock gamma_s chosen so that
the per-item excess co-failure covariance in stratum s,
    cov_s = gamma_s (1 - gamma_s) (1 - p_i_s)(1 - p_j_s),
follows a PRESCRIBED SHAPE profile w_s (up to a global scale set by `gamma`).  This
is the same covariance algebra as dgp._equicorr_stratum_shocks, but instead of a FLAT
profile we impose:

  (A) REVERSED-GRADIENT : w_s increases with stratum index  => signal concentrated in
      the HARD/high strata.  The previously-dominant easy window {0,1,2} is now the
      WORST expert; window {7,8,9} is best.  This is Regime 1 with the gradient flipped.

  (B) TWO-PEAK          : w_s peaks at BOTH extremes (low strata 0-2 AND high strata
      7-9) with a trough in the middle.  No single contiguous 3-stratum window captures
      the signal; the best strategy must BLEND the two ends (which only EW / MixWin /
      half-EW can do, not a single window).

SCALE CONVENTION.  As in dgp._equicorr_stratum_shocks, `gamma` is the shock prob in
the stratum with the LARGEST target covariance; every other stratum's gamma_s is scaled
DOWN to hold cov_s / w_s constant.  So `gamma` is directly comparable to the existing
alt_dgp gamma grid (it is the peak per-stratum shock).

NULL.  null_dgp (static H0, conditional independence given difficulty) — same null the
existing harness uses — serves as the Type-I check for all three combiners on top of
the novel-DGP machinery.

SETTINGS.  Q=10, N=4000, BLOCK=20 (M=200 blocks), R=2000, alpha=0.05, THRESH=20=1/alpha.
Fresh seed base 80000 (seeds 80000+r), non-overlapping with existing runs (50000, 90000).
gamma grid {0.04,0.05,0.06,0.07,0.08,0.10,0.12,0.15} spanning unsaturated to saturated.

REPORTS (per combiner, per gamma, PER-SEED where applicable):
  - terminal log-wealth: mean, median
  - paired vs EW: Wilcoxon signed-rank p, frac(seeds combiner>EW), mean paired diff (nats),
    Cliff's delta, and the WORST per-seed shortfall vs EW (min of combiner-EW; the
    empirical test of the <= log 2 regret claim).
  - power: frac(seeds) where sup_n M_n >= THRESH.
  - Type-I on null for each combiner.
"""

import sys
import time
import json
import numpy as np
from scipy.stats import wilcoxon, binomtest

sys.path.insert(0, '/home/lyra/projects/evalue-sheaf/code/evalue')

from dgp import null_dgp, judge_probs
from eprocess import ville_reject, THRESHOLD
import run_online_weights as rw
from run_online_weights import build_increment_stream, wealth_path

# ── Settings ────────────────────────────────────────────────────────────────────
Q         = 10
BLOCK     = 20
N         = 4000
R         = 2000
ALPHA     = 0.05
THRESH    = 20.0          # 1/alpha  (same convention as eprocess.THRESHOLD / existing harness)
SEED_BASE = 80000         # FRESH, non-overlapping with existing runs (50000, 90000, 42+)

# Grid spans unsaturated -> saturated for BOTH DGPs.  The two-peak DGP saturates by
# ~0.12-0.15; the reversed-gradient DGP concentrates signal in the HARD strata whose
# (1-p) is small, so it needs gamma up to ~0.30 to saturate (verified empirically).
# A single wide grid covers both honestly.
GAMMA_GRID = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

# keep imported machinery's module globals consistent with ours
rw.Q = Q
rw.BLOCK = BLOCK

# shipped fresh-pair build on this branch
BUILD_KW = dict(pairing='within_block', slack='second_order', pairing_rule='cyclic')


# ── DIRECT per-stratum shock profile.  gamma_s = gamma * shape_s, shape in [0,1]. ─
#    We use a DIRECT shock profile (not covariance-shaping) so that `gamma` is
#    literally the PEAK per-stratum shock probability — exactly the same meaning it
#    has in the existing alt_dgp grid (where gamma is the uniform shock prob).  This
#    also keeps every gamma_s bounded and feasible across the whole grid.  The
#    resulting per-stratum excess co-failure covariance is cov_s = gamma_s(1-gamma_s)
#    (1-p_i_s)(1-p_j_s); we PRINT the cov ordering at startup so the reader can see
#    exactly which window ends up best/worst (the signal structure is not assumed).
def _profile_alt_dgp(N, gamma, rng, shape, Q=Q):
    """Marshall-Olkin shock DGP with per-stratum shock gamma_s = gamma * shape_s."""
    shape = np.asarray(shape, dtype=float)
    shape = shape / shape.max()                    # peak shape = 1 => peak shock = gamma
    theta = rng.uniform(0.0, 1.0, size=N)
    p_i, p_j = judge_probs(theta)
    strata = np.clip((theta * Q).astype(int), 0, Q - 1)
    gamma_s = gamma * shape
    g_item = gamma_s[strata]
    shock = rng.uniform(size=N) < g_item
    W_i = np.where(shock, 1, (rng.uniform(size=N) < p_i).astype(int))
    W_j = np.where(shock, 1, (rng.uniform(size=N) < p_j).astype(int))
    return dict(theta=theta, W_i=W_i, W_j=W_j, p_i=p_i, p_j=p_j, shock=shock, gamma_s=gamma_s)


# shock-shape profiles over Q=10 strata (index 0 = easiest, 9 = hardest).
# (A) reversed gradient: shock RISES steeply with difficulty.  Because the per-stratum
#     covariance is gamma_s(1-gamma_s)(1-p_i)(1-p_j) and (1-p) FALLS with difficulty,
#     we use a quadratic rise so the shock gain overpowers the falling (1-p) and the
#     HARD window {7,8,9} carries the most co-failure covariance (opposite of Regime 1).
SHAPE_REVERSED = ((np.arange(Q) + 1.0) / Q) ** 2           # steep rise, peak at s=9
# (B) two-peak: shock peaks in the two END windows (strata 0-2 AND 7-9), ~0 in the middle.
SHAPE_TWOPEAK = np.zeros(Q)
SHAPE_TWOPEAK[0:3] = 1.0                                     # easy-end window {0,1,2}
SHAPE_TWOPEAK[Q-3:Q] = 1.0                                   # hard-end window {7,8,9}

def reversed_dgp(N, gamma, rng):
    return _profile_alt_dgp(N, gamma, rng, SHAPE_REVERSED)

def twopeak_dgp(N, gamma, rng):
    return _profile_alt_dgp(N, gamma, rng, SHAPE_TWOPEAK)


def _print_cov_profile():
    """Print the per-stratum excess co-failure covariance ordering (at gamma=0.10) so
    the reader can verify, not assume, which window is best/worst under each DGP."""
    centers = (np.arange(Q) + 0.5) / Q
    a, b = judge_probs(centers)
    r = (1.0 - a) * (1.0 - b)
    g = 0.10
    for name, shape in [("REVERSED", SHAPE_REVERSED), ("TWO-PEAK", SHAPE_TWOPEAK)]:
        sh = np.asarray(shape, float); sh = sh / sh.max()
        gs = g * sh
        cov = gs * (1.0 - gs) * r
        win_cov = [cov[k0:k0+3].sum() for k0 in range(Q - 2)]
        best = int(np.argmax(win_cov)); worst = int(np.argmin(win_cov))
        print(f"  [{name}] per-stratum cov (x1e4) @gamma=0.10: "
              f"{np.round(cov*1e4,2)}")
        print(f"            window co-failure cov (x1e4): "
              f"{[round(c*1e4,2) for c in win_cov]}")
        print(f"            BEST window = {{{best},{best+1},{best+2}}}  "
              f"WORST window = {{{worst},{worst+1},{worst+2}}}")


# ── Combiner weight vectors (window experts) ─────────────────────────────────────
def eq_weights(b):
    cap = 1.0 / b
    return np.full(Q, cap / Q)

def window_weights(b, k0):
    """cap/3 on 0-indexed strata {k0,k0+1,k0+2}, 0 elsewhere."""
    cap = 1.0 / b
    w = np.zeros(Q)
    w[k0:k0 + 3] = cap / 3.0
    return w

def window_experts(b):
    """8 contiguous 3-strata windows {0,1,2}..{7,8,9}."""
    return [window_weights(b, k0) for k0 in range(Q - 2)]   # 8 windows


# ── Combiner wealth processes (all averages of component WEALTH PROCESSES) ────────
def ew_wealth(X, b):
    """The EW expert's own wealth process."""
    return wealth_path(X, eq_weights(b))

def mixwin_wealth(X, b):
    """(1/9) average of [EW + 8 window] wealth processes."""
    comps = [eq_weights(b)] + window_experts(b)             # 9 experts
    assert len(comps) == 9
    paths = np.array([wealth_path(X, w) for w in comps])    # (9, M)
    return paths.mean(axis=0)

def halfew_wealth(X, b):
    """1/2 * EW-wealth + (1/16) * each of 8 window-wealths.  Weights sum to 1."""
    ew = wealth_path(X, eq_weights(b))                      # (M,)
    win_paths = np.array([wealth_path(X, w) for w in window_experts(b)])  # (8, M)
    weights_sum = 0.5 + 8 * (1.0 / 16.0)
    assert abs(weights_sum - 1.0) < 1e-12, weights_sum
    return 0.5 * ew + (1.0 / 16.0) * win_paths.sum(axis=0)


# ── Per-seed run: return terminal log-wealth and crossing bool for each combiner ──
def one_seed(data, rng):
    X, b = build_increment_stream(data, rng, **BUILD_KW)
    M_ew  = ew_wealth(X, b)
    M_mw  = mixwin_wealth(X, b)
    M_he  = halfew_wealth(X, b)

    # per-window-expert wealth paths (the 8 contiguous 3-strata windows {0,1,2}..{7,8,9}).
    # These are the SAME paths already summed inside mixwin_wealth / halfew_wealth;
    # computing them here is a pure function of X (no rng) so the EW/MixWin/half-EW
    # numbers reproduce bit-for-bit.
    win_paths = [wealth_path(X, w) for w in window_experts(b)]   # 8 paths

    def tlog(Mp):
        return float(np.log(max(Mp[-1], 1e-300)))

    def fired(Mp):
        return bool(ville_reject(Mp, THRESH)[0])

    return dict(
        lw_ew=tlog(M_ew), lw_mw=tlog(M_mw), lw_he=tlog(M_he),
        fired_ew=fired(M_ew), fired_mw=fired(M_mw), fired_he=fired(M_he),
        lw_win=[tlog(Mp) for Mp in win_paths],       # list of 8 terminal log-wealths
        fired_win=[fired(Mp) for Mp in win_paths],   # list of 8 crossing bools
    )


def paired_sign_test_power(fired_combiner, fired_ew):
    """Exact McNemar/sign test on PAIRED per-seed crossing outcomes.

    For each seed we have a pair (combiner fired?, EW fired?).  Discordant seeds
    are those where exactly one of the two fired.  Under H0 (no power difference)
    a discordant seed is equally likely to favour either arm, so the number of
    'combiner-only' crossings among the discordant seeds is Binomial(n_disc, 1/2).
    Returns (n_combiner_only, n_ew_only, two-sided p).  p=1.0 if no discordant seeds.
    """
    c = np.asarray(fired_combiner, bool)
    e = np.asarray(fired_ew, bool)
    comb_only = int(np.sum(c & ~e))
    ew_only   = int(np.sum(~c & e))
    n_disc = comb_only + ew_only
    if n_disc == 0:
        return comb_only, ew_only, 1.0
    p = float(binomtest(comb_only, n_disc, 0.5, alternative='two-sided').pvalue)
    return comb_only, ew_only, p


def cliffs_delta(a, b):
    """Cliff's delta for paired/unpaired samples a vs b: P(a>b)-P(a<b).
    Here we use the paired per-seed differences d=a-b directly for a *paired* effect:
    delta = [#(d>0) - #(d<0)] / n  (dominance of combiner over EW across seeds)."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    return (np.sum(d > 0) - np.sum(d < 0)) / n


def run_dgp(dgp_name, make_data):
    """Run all gammas for one novel DGP; return list of per-gamma result dicts."""
    out = []
    N_WIN = Q - 2  # 8 contiguous 3-strata windows
    for g in GAMMA_GRID:
        lw_ew = np.empty(R); lw_mw = np.empty(R); lw_he = np.empty(R)
        fr_ew = np.empty(R, bool); fr_mw = np.empty(R, bool); fr_he = np.empty(R, bool)
        lw_win = np.empty((R, N_WIN))          # per-seed per-window terminal log-wealth
        fr_win = np.empty((R, N_WIN), bool)    # per-seed per-window crossing bool
        for r in range(R):
            rng = np.random.default_rng(SEED_BASE + r)
            data = make_data(N, g, rng)
            res = one_seed(data, rng)
            lw_ew[r] = res['lw_ew']; lw_mw[r] = res['lw_mw']; lw_he[r] = res['lw_he']
            fr_ew[r] = res['fired_ew']; fr_mw[r] = res['fired_mw']; fr_he[r] = res['fired_he']
            lw_win[r] = res['lw_win']; fr_win[r] = res['fired_win']

        # per-window-expert aggregates (power = frac seeds crossing; growth = mean terminal log-wealth)
        win_power = fr_win.mean(axis=0)   # (8,)
        win_mean  = lw_win.mean(axis=0)   # (8,)
        win_labels = [f"{{{k0},{k0+1},{k0+2}}}" for k0 in range(N_WIN)]
        best_win_by_power  = int(np.argmax(win_power))
        best_win_by_growth = int(np.argmax(win_mean))

        d_mw = lw_mw - lw_ew
        d_he = lw_he - lw_ew

        def wilcox_p(d):
            if np.allclose(d, 0):
                return 1.0
            try:
                return float(wilcoxon(d, zero_method='wilcox', alternative='two-sided').pvalue)
            except ValueError:
                return 1.0

        # paired sign-test on per-seed crossing outcomes (power differences)
        mw_c_only, mw_e_only, mw_sign_p = paired_sign_test_power(fr_mw, fr_ew)
        he_c_only, he_e_only, he_sign_p = paired_sign_test_power(fr_he, fr_ew)

        row = dict(
            dgp=dgp_name, gamma=g,
            # terminal log-wealth
            mean_ew=float(lw_ew.mean()), med_ew=float(np.median(lw_ew)),
            mean_mw=float(lw_mw.mean()), med_mw=float(np.median(lw_mw)),
            mean_he=float(lw_he.mean()), med_he=float(np.median(lw_he)),
            # power (fraction of seeds crossing the alarm boundary)
            pow_ew=float(fr_ew.mean()), pow_mw=float(fr_mw.mean()), pow_he=float(fr_he.mean()),
            # power paired differences (points) + exact sign-test p
            mw_powdiff=float(fr_mw.mean() - fr_ew.mean()),
            mw_pow_comb_only=mw_c_only, mw_pow_ew_only=mw_e_only, mw_pow_sign_p=mw_sign_p,
            he_powdiff=float(fr_he.mean() - fr_ew.mean()),
            he_pow_comb_only=he_c_only, he_pow_ew_only=he_e_only, he_pow_sign_p=he_sign_p,
            # MixWin growth paired vs EW (mean diff, SE, Wilcoxon)
            mw_meandiff=float(d_mw.mean()), mw_se=float(d_mw.std(ddof=1) / np.sqrt(R)),
            mw_frac_gt=float((d_mw > 0).mean()),
            mw_wilcox_p=wilcox_p(d_mw), mw_cliff=float(cliffs_delta(lw_mw, lw_ew)),
            mw_worst=float(d_mw.min()),
            # half-EW growth paired vs EW (mean diff, SE, Wilcoxon)
            he_meandiff=float(d_he.mean()), he_se=float(d_he.std(ddof=1) / np.sqrt(R)),
            he_frac_gt=float((d_he > 0).mean()),
            he_wilcox_p=wilcox_p(d_he), he_cliff=float(cliffs_delta(lw_he, lw_ew)),
            he_worst=float(d_he.min()),
            # per-window-expert reporting (8 contiguous 3-strata windows {0,1,2}..{7,8,9})
            win_labels=win_labels,
            win_power=[float(x) for x in win_power],    # Ville-reject power per window
            win_mean_lw=[float(x) for x in win_mean],   # mean terminal log-wealth per window
            best_win_by_power=best_win_by_power,
            best_win_by_power_label=win_labels[best_win_by_power],
            best_win_power=float(win_power[best_win_by_power]),
            best_win_by_growth=best_win_by_growth,
            best_win_by_growth_label=win_labels[best_win_by_growth],
            best_win_growth=float(win_mean[best_win_by_growth]),
        )
        out.append(row)
        print(f"    [{dgp_name}] gamma={g:.2f} done  "
              f"pow(EW/MW/HE)={row['pow_ew']:.3f}/{row['pow_mw']:.3f}/{row['pow_he']:.3f}  "
              f"HE worst-vs-EW={row['he_worst']:+.3f} nats", flush=True)
        print(f"         best single window by power = {row['best_win_by_power_label']} "
              f"pow={row['best_win_power']:.3f} | by growth = {row['best_win_by_growth_label']} "
              f"lw={row['best_win_growth']:+.3f}  (vs MixWin pow={row['pow_mw']:.3f} "
              f"lw={row['mean_mw']:+.3f} ; half-EW pow={row['pow_he']:.3f} "
              f"lw={row['mean_he']:+.3f})", flush=True)
    return out


def run_null():
    """Type-I for all three combiners on the static null (consistency check)."""
    fr_ew = np.empty(R, bool); fr_mw = np.empty(R, bool); fr_he = np.empty(R, bool)
    for r in range(R):
        rng = np.random.default_rng(SEED_BASE + r)
        data = null_dgp(N, rng)
        res = one_seed(data, rng)
        fr_ew[r] = res['fired_ew']; fr_mw[r] = res['fired_mw']; fr_he[r] = res['fired_he']
    return dict(typeI_ew=fr_ew.mean(), typeI_mw=fr_mw.mean(), typeI_he=fr_he.mean())


def print_table(rows, title):
    print("\n" + "=" * 118)
    print(title)
    print("=" * 118)
    hdr = (f"{'gamma':>6} | {'power EW/MW/HE':>20} | "
           f"{'mean lw EW/MW/HE':>26} | "
           f"{'HE vs EW: mean/worst/frac>/p/cliff':>42}")
    print(hdr)
    print("-" * 118)
    for r in rows:
        print(f"{r['gamma']:>6.2f} | "
              f"{r['pow_ew']:>6.3f}/{r['pow_mw']:.3f}/{r['pow_he']:.3f}     | "
              f"{r['mean_ew']:>+7.3f}/{r['mean_mw']:+7.3f}/{r['mean_he']:+7.3f}  | "
              f"{r['he_meandiff']:>+6.3f}/{r['he_worst']:+6.3f}/{r['he_frac_gt']:.3f}/"
              f"{r['he_wilcox_p']:.1e}/{r['he_cliff']:+.3f}")
    print("-" * 118)
    print("  half-EW vs EW -- POWER paired differences (pts) + exact sign-test p:")
    for r in rows:
        print(f"    gamma={r['gamma']:.2f}  pow(EW/HE)={r['pow_ew']:.3f}/{r['pow_he']:.3f}  "
              f"HE-EW={r['he_powdiff']:+.3f}pt  "
              f"(HE-only={r['he_pow_comb_only']}, EW-only={r['he_pow_ew_only']}, "
              f"sign p={r['he_pow_sign_p']:.2e})")
    print("  half-EW vs EW -- GROWTH mean paired diff (nats) +/- SE, Wilcoxon p:")
    for r in rows:
        print(f"    gamma={r['gamma']:.2f}  HE-EW mean={r['he_meandiff']:+.3f} "
              f"+/- {r['he_se']:.3f}  wilcox_p={r['he_wilcox_p']:.1e}")
    print("-" * 118)
    print("  MixWin vs EW detail (power sign-test + growth):")
    for r in rows:
        print(f"    gamma={r['gamma']:.2f}  pow MW-EW={r['mw_powdiff']:+.3f}pt "
              f"(sign p={r['mw_pow_sign_p']:.2e})  "
              f"growth MW-EW mean={r['mw_meandiff']:+.3f}+/-{r['mw_se']:.3f}  "
              f"worst={r['mw_worst']:+.3f}  frac>={r['mw_frac_gt']:.3f}  "
              f"wilcox_p={r['mw_wilcox_p']:.1e}  cliff={r['mw_cliff']:+.3f}")


def main():
    t0 = time.time()
    print("=" * 118)
    print("half-EW STRESS TEST on TWO NOVEL DGPs (reversed-gradient, two-peak)")
    print(f"  Q={Q}, N={N}, BLOCK={BLOCK} (M={N//BLOCK} blocks), R={R}, "
          f"alpha={ALPHA}, THRESH={THRESH}=1/alpha")
    print(f"  seed base {SEED_BASE}+r (fresh)  |  build={BUILD_KW}")
    print(f"  Combiners: EW | MixWin (1/9 x9) | half-EW (1/2 EW + 1/16 x8 windows)")
    print(f"  gamma grid: {GAMMA_GRID}")
    print("-" * 118)
    print("  Signal structure (VERIFIED from covariance algebra, not assumed):")
    _print_cov_profile()
    print("=" * 118)

    print("\n>>> (A) REVERSED-GRADIENT DGP (signal in HARD strata; window {7,8,9} best, {0,1,2} worst)")
    rev_rows = run_dgp("REV", reversed_dgp)

    print("\n>>> (B) TWO-PEAK DGP (signal at both extremes; no single 3-window dominates)")
    tp_rows = run_dgp("TWO", twopeak_dgp)

    print("\n>>> NULL (static H0) — Type-I consistency check for all three combiners")
    null_res = run_null()

    print_table(rev_rows, "RESULTS — (A) REVERSED-GRADIENT DGP")
    print_table(tp_rows,  "RESULTS — (B) TWO-PEAK DGP")

    print("\n" + "=" * 60)
    print("TYPE-I on static null (target <= {:.2f}):".format(ALPHA))
    print(f"  EW={null_res['typeI_ew']:.4f}  "
          f"MixWin={null_res['typeI_mw']:.4f}  "
          f"half-EW={null_res['typeI_he']:.4f}")
    print("=" * 60)

    # overall worst-case half-EW shortfall vs EW across BOTH DGPs and ALL gammas
    all_worst = min([r['he_worst'] for r in rev_rows] + [r['he_worst'] for r in tp_rows])
    print(f"\nGLOBAL worst per-seed half-EW shortfall vs EW (both DGPs, all gammas): "
          f"{all_worst:+.4f} nats   (log 2 = {np.log(2):.4f})")
    print(f"TOTAL RUNTIME: {time.time()-t0:.1f}s   (R={R})")
    print("=" * 118)

    # ── Persist results (fixed descriptive name; no timestamp / no RNG in name) ──
    import os
    results_dir = '/home/lyra/projects/evalue-sheaf/results'
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'novel_dgp_halfew_results.json')
    payload = dict(
        meta=dict(
            script='novel_dgp_halfew_test.py',
            Q=Q, N=N, BLOCK=BLOCK, M=N // BLOCK, R=R, ALPHA=ALPHA, THRESH=THRESH,
            seed_base=SEED_BASE, build=BUILD_KW, gamma_grid=GAMMA_GRID,
            combiners='EW | MixWin(1/9 x9) | half-EW(1/2 EW + 1/16 x8 windows)',
            log2=float(np.log(2)),
        ),
        reversed_gradient=rev_rows,
        two_peak=tp_rows,
        null_typeI=null_res,
        global_worst_halfew_shortfall_nats=float(all_worst),
    )
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == '__main__':
    main()
