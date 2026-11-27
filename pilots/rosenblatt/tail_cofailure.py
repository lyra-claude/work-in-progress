"""
rosenblatt-pilot/tail_cofailure.py
==================================
A margins-free TAIL co-failure diagnostic for LLM-judge panels.

Motivation
----------
Second-moment statistics (pairwise Pearson/Spearman |corr| on the PIT streams,
as used in pilot.py / rolling_ranks.py) measure the AVERAGE co-movement of the
judges. They are known to be (near-)BLIND to TAIL co-failure: a copula can have
~0 global correlation yet still put excess joint mass in the upper corner, i.e.
all judges scoring an item extremely high (or low) at the same time. That joint
upper-tail event is exactly the "they all fail together" regime we care about.

The statistic
-------------
Work on the margins-free PIT streams U in [0,1]^K (K judges). For a tail level
alpha, define the JOINT UPPER-TAIL EXCEEDANCE RATE:

    p_hat(alpha) = (1/n) * sum_i  1{ U[i,j] > 1-alpha  for ALL judges j }

Under independent judges with uniform margins this equals alpha^K exactly.
We report:
    excess = p_hat - alpha^K           (0 under independence; >0 = tail co-failure)
    lift   = p_hat / alpha^K           (1 under independence)

Lower tail is the mirror image: 1{ U[i,j] < alpha for all j }, same null alpha^K.

This is a BATCH / EMPIRICAL diagnostic. It is NOT an anytime-valid e-process and
makes no sequential-validity claim.

Scenarios, seeding, and rank transforms are reused verbatim from rolling_ranks.py
so the numbers are directly comparable:
  A = genuine cross-judge dependence (one-factor Gaussian copula, rho 0->0.8)
  B = drift-only null (independent per-judge random walks, rho=0)
  D = shared-time-trend (common mu(t), independent errors -> NO genuine co-failure)

Same seeding convention: np.random.default_rng(20261123), per-scenario spawned
sub-streams. Multi-seed replication wraps the whole pipeline over several master
seeds to report mean +/- sd and judge tail-count noise.
"""

import numpy as np

from rolling_ranks import (
    N, K, SEED,
    scenario_dependence, scenario_drift_null, scenario_shared_trend,
    rank_pit_global, rank_pit_rolling,
)

# Tail levels to probe. alpha^K is the independence null for the joint event.
ALPHAS = [0.10, 0.05]
# Rolling window: 100 worked well in rolling_ranks.py (killed the D confound
# while preserving A). We use that single window for the tail statistic.
WINDOW = 100
# Number of master seeds for replication (mean +/- sd).
N_SEEDS = 10


# ---------------------------------------------------------------------------
# Tail co-failure statistic
# ---------------------------------------------------------------------------

def joint_tail_rate(U, alpha, upper=True):
    """
    Joint tail exceedance rate on a PIT array U (n x K), NaN-aware.

    upper=True : fraction of usable rows with  U[:,j] > 1-alpha  for ALL j.
    upper=False: fraction of usable rows with  U[:,j] <   alpha  for ALL j.

    Only rows with all-K non-NaN PITs are counted (rolling ranks have a causal
    warm-up region of NaNs). Returns (rate, n_usable_rows, n_joint_tail_events).
    """
    usable = ~np.isnan(U).any(axis=1)
    Uu = U[usable]
    n_use = Uu.shape[0]
    if n_use == 0:
        return np.nan, 0, 0
    if upper:
        hit = (Uu > (1.0 - alpha)).all(axis=1)
    else:
        hit = (Uu < alpha).all(axis=1)
    n_events = int(hit.sum())
    return n_events / n_use, n_use, n_events


def tail_stats(U, alpha, upper=True):
    """Return dict: joint rate, independence null alpha^K, excess, lift, counts."""
    rate, n_use, n_events = joint_tail_rate(U, alpha, upper=upper)
    null = alpha ** K
    excess = rate - null
    lift = rate / null if null > 0 else np.nan
    return {
        "alpha": alpha,
        "upper": upper,
        "rate": rate,
        "null": null,
        "excess": excess,
        "lift": lift,
        "n_usable": n_use,
        "n_events": n_events,
    }


# ---------------------------------------------------------------------------
# One full replicate (one master seed): build all scenarios, both transforms.
# ---------------------------------------------------------------------------

def run_one_seed(seed):
    """
    Returns nested dict:
      out[scenario][method][(alpha, 'upper'|'lower')] = tail_stats dict
    scenario in {A, B, D}; method in {global, rolling}.
    """
    master = np.random.default_rng(seed)
    rng_A, rng_B, rng_D = master.spawn(3)

    data = {
        "A": scenario_dependence(rng_A, N),
        "B": scenario_drift_null(rng_B, N),
        "D": scenario_shared_trend(rng_D, N),
    }

    out = {}
    for scn, X in data.items():
        Ug = rank_pit_global(X)
        Ur = rank_pit_rolling(X, WINDOW)
        out[scn] = {}
        for method, U in [("global", Ug), ("rolling", Ur)]:
            out[scn][method] = {}
            for alpha in ALPHAS:
                for upper in (True, False):
                    key = (alpha, "upper" if upper else "lower")
                    out[scn][method][key] = tail_stats(U, alpha, upper=upper)
    return out


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def aggregate(seeds):
    """
    Run all seeds, aggregate mean/sd of excess, lift, rate; also mean n_events
    and n_usable. Returns agg[scn][method][key] = dict of aggregated quantities.
    """
    per_seed = [run_one_seed(s) for s in seeds]

    scns = ["A", "B", "D"]
    methods = ["global", "rolling"]
    keys = [(a, t) for a in ALPHAS for t in ("upper", "lower")]

    agg = {}
    for scn in scns:
        agg[scn] = {}
        for method in methods:
            agg[scn][method] = {}
            for key in keys:
                rates = np.array([ps[scn][method][key]["rate"] for ps in per_seed])
                excesses = np.array([ps[scn][method][key]["excess"] for ps in per_seed])
                lifts = np.array([ps[scn][method][key]["lift"] for ps in per_seed])
                nev = np.array([ps[scn][method][key]["n_events"] for ps in per_seed])
                nus = np.array([ps[scn][method][key]["n_usable"] for ps in per_seed])
                null = per_seed[0][scn][method][key]["null"]
                agg[scn][method][key] = {
                    "null": null,
                    "rate_mean": float(np.nanmean(rates)),
                    "rate_sd": float(np.nanstd(rates, ddof=1)),
                    "excess_mean": float(np.nanmean(excesses)),
                    "excess_sd": float(np.nanstd(excesses, ddof=1)),
                    "excess_per_seed": excesses,   # kept for paired contrasts
                    "lift_mean": float(np.nanmean(lifts)),
                    "lift_sd": float(np.nanstd(lifts, ddof=1)),
                    "n_events_mean": float(np.mean(nev)),
                    "n_usable_mean": float(np.mean(nus)),
                }
    return agg, per_seed


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

SCN_DESC = {
    "A": "genuine dependence",
    "B": "drift-only null (independent)",
    "D": "shared trend (no genuine co-failure)",
}


# --- Verdict helpers (shared by console + markdown reporting) ---------------
def sig_pos(r):
    """Excess clearly positive: mean > 2 sd above 0 AND above the alpha^K null."""
    return r["excess_mean"] > max(2 * r["excess_sd"], 0) and r["excess_mean"] > r["null"]


def near_zero(r):
    """Excess consistent with the independence null (within 2 sd or below null)."""
    return abs(r["excess_mean"]) <= max(2 * r["excess_sd"], r["null"])


def paired_contrast(r, baseline):
    """
    PAIRED (same-master-seed) contrast of r's excess vs baseline's excess.

    Because every scenario in a replicate is driven by the SAME master seed
    (master.spawn(3) gives A/B/D common provenance), the shared sampling noise
    largely cancels in the per-seed difference, making this far more powerful
    than an unpaired pooled-sd test. Returns (mean_diff, t_stat, n_pos, n).
    """
    d = np.asarray(r["excess_per_seed"]) - np.asarray(baseline["excess_per_seed"])
    d = d[~np.isnan(d)]
    n = d.size
    md = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    t = md / (sd / np.sqrt(n)) if sd > 0 else float("inf")
    n_pos = int((d > 0).sum())
    return md, t, n_pos, n


def sig_above_paired(r, baseline):
    """r's excess paired-significantly exceeds baseline's (t > 2, two-sided ~5%)."""
    md, t, _, _ = paired_contrast(r, baseline)
    return (md > 0) and (t > 2.0)


def print_report(agg, seeds):
    print("=" * 92)
    print("TAIL CO-FAILURE DIAGNOSTIC (margins-free, batch/empirical)")
    print(f"N={N} items, K={K} judges, window={WINDOW}, seeds={list(seeds)} "
          f"({len(seeds)} replicates)")
    print("Null (independence): joint-tail rate = alpha^K; excess=rate-alpha^K; lift=rate/alpha^K")
    print("=" * 92)

    for tail in ("upper", "lower"):
        print(f"\n### {tail.upper()} tail")
        hdr = (f"{'Scen':4s} {'Method':8s} {'alpha':>6s} {'null a^K':>10s} "
               f"{'rate':>18s} {'excess':>18s} {'lift':>14s} {'~n_ev':>7s} {'n_use':>7s}")
        print(hdr)
        print("-" * len(hdr))
        for scn in ["A", "B", "D"]:
            for method in ["global", "rolling"]:
                for alpha in ALPHAS:
                    r = agg[scn][method][(alpha, tail)]
                    print(f"{scn:4s} {method:8s} {alpha:>6.2f} {r['null']:>10.5f} "
                          f"{r['rate_mean']:>9.5f}+/-{r['rate_sd']:<7.5f} "
                          f"{r['excess_mean']:>9.5f}+/-{r['excess_sd']:<7.5f} "
                          f"{r['lift_mean']:>7.2f}+/-{r['lift_sd']:<5.2f} "
                          f"{r['n_events_mean']:>7.1f} {r['n_usable_mean']:>7.0f}")
            print()

    # Success-criterion evaluation on the UPPER tail (symmetric for lower).
    print("=" * 92)
    print("SUCCESS CRITERION (upper tail):")
    print("  A: excess clearly POSITIVE under both transforms (sees real tail co-failure)")
    print("  B: excess ~0 (independent null)")
    print("  D GLOBAL: spurious POSITIVE excess? (shared trend fakes tail co-failure)")
    print("  D ROLLING: excess back to ~0 (causal ranks fix it)")
    print("-" * 92)
    for alpha in ALPHAS:
        A_g = agg["A"]["global"][(alpha, "upper")]
        A_r = agg["A"]["rolling"][(alpha, "upper")]
        B_r = agg["B"]["rolling"][(alpha, "upper")]
        B_g = agg["B"]["global"][(alpha, "upper")]
        D_g = agg["D"]["global"][(alpha, "upper")]
        D_r = agg["D"]["rolling"][(alpha, "upper")]
        print(f"alpha={alpha:.2f}:")
        print(f"  A global  excess={A_g['excess_mean']:+.5f} (lift {A_g['lift_mean']:.1f}) "
              f"-> {'POSITIVE' if sig_pos(A_g) else 'not sig'}")
        print(f"  A rolling excess={A_r['excess_mean']:+.5f} (lift {A_r['lift_mean']:.1f}) "
              f"-> {'POSITIVE' if sig_pos(A_r) else 'not sig'}")
        print(f"  B rolling excess={B_r['excess_mean']:+.5f} "
              f"-> {'~0' if near_zero(B_r) else 'NONZERO'}")
        # D-global spurious: PAIRED contrast of its excess vs the B-global
        # baseline (same master seed -> shared noise cancels; proper test).
        md, t, npos, ntot = paired_contrast(D_g, B_g)
        print(f"  D global  excess={D_g['excess_mean']:+.5f} (lift {D_g['lift_mean']:.1f}) "
              f"vs B global {B_g['excess_mean']:+.5f} "
              f"[paired diff={md:+.5f}, t={t:.2f}, {npos}/{ntot} seeds>0] "
              f"-> {'SPURIOUS POSITIVE' if sig_above_paired(D_g, B_g) else 'not sig vs B'}")
        print(f"  D rolling excess={D_r['excess_mean']:+.5f} (lift {D_r['lift_mean']:.1f}) "
              f"-> {'~0 (fixed)' if near_zero(D_r) else 'STILL NONZERO'}")
        print()


def build_md(agg, seeds):
    def cell(r):
        return (f"{r['rate_mean']:.5f} | {r['excess_mean']:+.5f}+/-{r['excess_sd']:.5f} "
                f"| {r['lift_mean']:.2f}")

    lines = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("# APPENDED: Tail Co-Failure Diagnostic")
    lines.append("")
    lines.append(f"**Setup:** N={N} items, K={K} judges, rolling window={WINDOW}, "
                 f"{len(seeds)} master seeds (base seed={SEED}, seeds {list(seeds)}).")
    lines.append("Same scenarios/seeding/transforms as `rolling_ranks.py` (imported directly).")
    lines.append("")
    lines.append("## What this adds")
    lines.append("")
    lines.append("The pairwise `|corr|` statistic measures *average* co-movement and is "
                 "(near-)blind to **tail** co-failure. This diagnostic targets the tail "
                 "directly: the **joint upper-tail exceedance rate** = fraction of items "
                 "where **all K judges' PITs exceed 1-alpha simultaneously**. Under "
                 "independent uniform margins this equals `alpha^K` exactly, so we report")
    lines.append("")
    lines.append("- **excess** = observed_rate - `alpha^K`  (0 under independence; >0 = tail co-failure)")
    lines.append("- **lift**   = observed_rate / `alpha^K`  (1 under independence)")
    lines.append("")
    lines.append("Lower tail is the mirror (`all judges < alpha`), same null. "
                 "**Batch / empirical diagnostic only -- NOT an anytime-valid e-process.**")
    lines.append("")

    for tail in ("upper", "lower"):
        lines.append(f"## {tail.capitalize()}-tail results (mean over {len(seeds)} seeds)")
        lines.append("")
        lines.append("| Scenario | Rank method | alpha | null alpha^K | joint-tail rate | "
                     "excess (mean+/-sd) | lift | ~n_events | n_usable |")
        lines.append("|----------|-------------|------:|------------:|---------------:|"
                     "------------------:|-----:|----------:|---------:|")
        for scn in ["A", "B", "D"]:
            for method in ["global", "rolling"]:
                for alpha in ALPHAS:
                    r = agg[scn][method][(alpha, tail)]
                    lines.append(
                        f"| {scn} ({SCN_DESC[scn]}) | {method} | {alpha:.2f} | "
                        f"{r['null']:.5f} | {r['rate_mean']:.5f} | "
                        f"{r['excess_mean']:+.5f}+/-{r['excess_sd']:.5f} | "
                        f"{r['lift_mean']:.2f} | {r['n_events_mean']:.1f} | "
                        f"{r['n_usable_mean']:.0f} |"
                    )
        lines.append("")

    # Verdict block (upper tail)
    lines.append("## Success criterion (upper tail)")
    lines.append("")
    lines.append("| alpha | A global | A rolling | B rolling | D global | D rolling | verdict |")
    lines.append("|------:|---------:|----------:|----------:|---------:|----------:|:-------:|")
    for alpha in ALPHAS:
        A_g = agg["A"]["global"][(alpha, "upper")]
        A_r = agg["A"]["rolling"][(alpha, "upper")]
        B_g = agg["B"]["global"][(alpha, "upper")]
        B_r = agg["B"]["rolling"][(alpha, "upper")]
        D_g = agg["D"]["global"][(alpha, "upper")]
        D_r = agg["D"]["rolling"][(alpha, "upper")]

        # D-global spurious = PAIRED (same-seed) contrast of its excess vs the
        # B-global baseline; D-rolling fixed = excess back within the null band.
        md_dg, t_dg, npos_dg, ntot_dg = paired_contrast(D_g, B_g)
        d_spurious = sig_above_paired(D_g, B_g)
        met = (sig_pos(A_g) and sig_pos(A_r) and near_zero(B_r)
               and d_spurious and near_zero(D_r))
        lines.append(
            f"| {alpha:.2f} | {A_g['excess_mean']:+.5f} ({'pos' if sig_pos(A_g) else 'ns'}) "
            f"| {A_r['excess_mean']:+.5f} ({'pos' if sig_pos(A_r) else 'ns'}) "
            f"| {B_r['excess_mean']:+.5f} ({'~0' if near_zero(B_r) else 'ne'}) "
            f"| {D_g['excess_mean']:+.5f} ({f'SPURIOUS+ (t={t_dg:.1f}, {npos_dg}/{ntot_dg})' if d_spurious else f'ns (t={t_dg:.1f})'}) "
            f"| {D_r['excess_mean']:+.5f} ({'fixed~0' if near_zero(D_r) else 'nonzero'}) "
            f"| {'**MET**' if met else 'partial'} |"
        )
    lines.append("")
    lines.append("Legend: pos=significant positive (excess>2sd and >null); ns=not significant; "
                 "~0=consistent with null; SPURIOUS+=D-global excess exceeds the B-global "
                 "baseline by a PAIRED same-seed contrast (t>2; shared trend fakes tail "
                 "co-failure); fixed~0=causal ranks return D's excess to the null band.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(f"- **Tail counts are small.** At alpha=0.05, K=3, the null is "
                 f"alpha^K=0.000125, i.e. expected ~{0.05**3 * N:.2f} joint events in "
                 f"N={N} items under independence. Even under dependence only a handful of "
                 f"items land in the joint tail, so alpha=0.05 is noisy (wide sd, limited "
                 f"power). alpha=0.10 (null {0.10**3:.4f}, ~{0.10**3*N:.1f} expected events) "
                 f"is the more reliable operating point here.")
    lines.append("- **Rolling ranks lose rows** to the causal warm-up, shrinking the usable "
                 "sample and further thinning the tail.")
    lines.append("- Multi-seed mean+/-sd (ddof=1) quantifies the finite-sample noise; it does "
                 "not confer any sequential/anytime validity.")
    lines.append("")
    lines.append("*Empirical tail-coincidence diagnostic only. No anytime-valid claim.*")
    lines.append("")

    path = "/home/lyra/projects/rosenblatt-pilot/results_rolling_ranks.md"
    with open(path, "a") as f:
        f.write("\n".join(lines))
    print(f"\nAppended tail-co-failure section to {path}")


if __name__ == "__main__":
    seeds = [SEED + i for i in range(N_SEEDS)]
    agg, per_seed = aggregate(seeds)
    print_report(agg, seeds)
    build_md(agg, seeds)
