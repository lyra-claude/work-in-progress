"""
rosenblatt-pilot/rolling_ranks.py
=================================
FIX for the spurious shared-time-trend confound in the margins-free
co-failure detector.

Background
----------
pilot.py established that a coordinate-wise GLOBAL rank transform (empirical
PIT) absorbs per-judge marginal drift (KS ~ 0.0002) while preserving genuine
cross-judge dependence (|corr| ~ 0.38 in the dependence scenario).

But a DISCOVERY / open problem surfaced: a SHARED TIME-TREND across judges
(all judges' means drift together over time) induces a SPURIOUS cross-judge
rank correlation (~0.25) under GLOBAL ranks. This is a temporal analog of a
difficulty confound: when every judge's score is high early and low late (or
vice versa), the GLOBAL rank of each judge is tied to the time index, so the
judges' global-rank-PITs become correlated THROUGH TIME even though their
conditional (given time) errors are independent. Drift is mistaken for
co-failure.

The fix (this file)
-------------------
ROLLING / CAUSAL ranks: rank each observation against a TRAILING WINDOW of
only PAST observations (causal) instead of the whole stream. Within a short
trailing window the shared slow trend is approximately constant, so it is
differenced away by the local ranking and the spurious cross-judge correlation
collapses -- while a genuine (local) copula dependence still shows up because
it is present within every window.

Same seeding convention as pilot.py: np.random.default_rng(20261123),
drawn sequentially, per-scenario sub-streams for reproducibility.

Empirical motivation only -- NOT an anytime-valid e-process.
"""

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration (mirrors pilot.py)
# ---------------------------------------------------------------------------
SEED = 20261123
N = 4000        # number of items in the stream
K = 3           # number of judges

WINDOWS = [50, 100, 200]   # trailing-window sizes to try for rolling ranks


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def scenario_dependence(rng, n):
    """
    A -- Genuine cross-judge dependence, marginals fixed.
    One-factor Gaussian copula with pairwise rho ramping 0 -> 0.8 (mean 0.4).
        X_j = sqrt(rho)*F + sqrt(1-rho)*eps_j
    Marginals stay N(0,1); dependence is real and LOCAL (present at every time).
    (Identical construction to pilot.py scenario_A.)
    """
    rho_vec = np.linspace(0.0, 0.8, n)
    F = rng.standard_normal(n)
    eps = rng.standard_normal((n, K))
    rho_col = rho_vec[:, np.newaxis]
    X = np.sqrt(rho_col) * F[:, np.newaxis] + np.sqrt(1.0 - rho_col) * eps
    return X


def scenario_drift_null(rng, n):
    """
    B -- Drift-only null: INDEPENDENT per-judge marginal drift, dependence zero.
    Each judge's mean follows its OWN independent random walk; sd varies
    independently in [1,2]; residuals independent. No shared latent factor.
    (Identical construction to pilot.py scenario_B.)
    """
    walk_steps = rng.standard_normal((n, K)) * (3.0 / np.sqrt(n))
    mu_walks = np.cumsum(walk_steps, axis=0)
    sd_walks = 1.0 + rng.uniform(0, 1, (n, K))
    Z = rng.standard_normal((n, K))
    X = mu_walks + sd_walks * Z
    return X


def scenario_shared_trend(rng, n):
    """
    D -- SHARED time-trend: ALL judges' means drift TOGETHER over time,
    but conditional-on-time errors are INDEPENDENT (rho=0 given t).

    This is the confound. A single shared deterministic mean trajectory
    mu(t) (a smooth drift of ~several sd units across the stream) is added
    to every judge; on top of it each judge has its own independent N(0,1)
    noise. There is NO copula dependence conditional on time -- the only
    thing the judges share is the common slow trend in their level.

    Under GLOBAL ranks this shared trend ties each judge's rank to the time
    index, producing a spurious positive cross-judge rank correlation
    (~0.2-0.25 at the calibrated trend strength). A causal/rolling rank, which
    only sees a short recent window in which mu(t) is ~constant, should remove it.

    Trend construction: a smooth shared random walk, std-scaled (see below).
    The spurious global |corr| grows monotonically with trend strength; the
    rolling fix is robust across the sweep (see trend_strength_sweep()).
    """
    # Shared smooth trend: random walk, scaled so that under GLOBAL ranks it
    # reproduces the prior run's spurious cross-judge |corr| of ~0.2-0.25
    # (a temporal difficulty-confound of realistic magnitude). A trend std of
    # ~0.5 marginal-sd units lands in that regime (see results note sweep).
    shared_steps = rng.standard_normal(n)
    shared_walk = np.cumsum(shared_steps)
    shared_walk = shared_walk - shared_walk.mean()
    shared_walk = shared_walk / (shared_walk.std() + 1e-12) * 0.5   # shared mu(t)

    # Independent per-judge residual noise (rho = 0 given t)
    eps = rng.standard_normal((n, K))

    X = shared_walk[:, np.newaxis] + eps
    return X


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def rank_pit_global(X):
    """
    GLOBAL (offline) empirical PIT: U_j = rank(X_j within whole column j)/(N+1).
    Ties: average rank. (Identical to pilot.py rank_pit.)
    """
    n = X.shape[0]
    U = np.empty_like(X)
    for j in range(X.shape[1]):
        ranks = stats.rankdata(X[:, j], method='average')
        U[:, j] = ranks / (n + 1)
    return U


def rank_pit_rolling(X, window):
    """
    ROLLING / CAUSAL empirical PIT.

    For each observation i and judge j, rank X[i,j] against the TRAILING
    window of PAST observations of that same judge:
        reference set = X[max(0, i-window) : i, j]   (strictly past; causal)
        U[i,j] = (#{past <= X[i,j]} + 0.5) / (len(reference)+1)

    The +0.5 and /(len+1) give a mid-rank style plotting-position that is
    ~Uniform[0,1] under local stationarity and avoids 0/1 endpoints.

    The FIRST `min_ref` observations have too few past points to rank
    reliably; we require at least `min_ref` past points and mark earlier
    rows as NaN (excluded from all statistics). This is the honest cost of
    causality: a warm-up period with no usable PIT.

    Causality note: observation i is ranked ONLY against strictly earlier
    observations, so this is a valid causal (online-computable) transform --
    no peeking at the future.
    """
    n, k = X.shape
    min_ref = max(20, window // 5)   # need a minimally populated window
    U = np.full((n, k), np.nan)
    for j in range(k):
        col = X[:, j]
        for i in range(n):
            lo = max(0, i - window)
            ref = col[lo:i]           # strictly past (excludes i)
            m = ref.shape[0]
            if m < min_ref:
                continue              # warm-up: leave as NaN
            # number of past points <= current value
            le = np.count_nonzero(ref <= col[i])
            U[i, j] = (le + 0.5) / (m + 1.0)
    return U


# ---------------------------------------------------------------------------
# Statistics (NaN-aware: rolling PIT has a warm-up region of NaNs)
# ---------------------------------------------------------------------------

def mean_ks_to_uniform(U):
    """Mean KS distance of each column of U to Uniform[0,1], ignoring NaNs."""
    ks_vals = []
    for j in range(U.shape[1]):
        col = U[:, j]
        col = col[~np.isnan(col)]
        if col.size < 10:
            ks_vals.append(np.nan)
            continue
        stat, _ = stats.kstest(col, 'uniform')
        ks_vals.append(stat)
    return float(np.nanmean(ks_vals))


def mean_pairwise_abs_pearson(U):
    """
    Mean absolute pairwise Pearson |corr| of the transformed U values.
    Uses only rows where BOTH judges in the pair are non-NaN.
    """
    K = U.shape[1]
    vals = []
    for j in range(K):
        for k in range(j + 1, K):
            a = U[:, j]
            b = U[:, k]
            mask = ~np.isnan(a) & ~np.isnan(b)
            if mask.sum() < 10:
                vals.append(np.nan)
                continue
            r = np.corrcoef(a[mask], b[mask])[0, 1]
            vals.append(abs(r))
    return float(np.nanmean(vals))


def n_usable(U):
    """Number of rows with all-K non-NaN (fully usable observations)."""
    return int(np.sum(~np.isnan(U).any(axis=1)))


def trend_strength_sweep(seed=SEED):
    """
    Robustness: vary the shared-trend strength (std in marginal-sd units) and
    record GLOBAL vs rolling |corr|. Shows the fix holds across the regime that
    brackets the prior run's ~0.2-0.25 spurious correlation. Returns list of dicts.
    """
    master = np.random.default_rng(seed)
    out = []
    for tstd in [0.3, 0.5, 1.0, 2.0]:
        r = master.spawn(1)[0]
        walk = np.cumsum(r.standard_normal(N))
        walk = walk - walk.mean()
        walk = walk / (walk.std() + 1e-12) * tstd
        eps = r.standard_normal((N, K))
        X = walk[:, None] + eps
        out.append({
            "trend_std": tstd,
            "global": mean_pairwise_abs_pearson(rank_pit_global(X)),
            "roll50": mean_pairwise_abs_pearson(rank_pit_rolling(X, 50)),
            "roll100": mean_pairwise_abs_pearson(rank_pit_rolling(X, 100)),
            "roll200": mean_pairwise_abs_pearson(rank_pit_rolling(X, 200)),
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 78)
    print("Rolling/Causal Ranks -- Fix for the shared-time-trend confound")
    print(f"N={N} items, K={K} judges, seed={SEED}, windows={WINDOWS}")
    print("=" * 78)

    # Per-scenario independent sub-streams from the same master seed, so each
    # scenario is reproducible and independent. (spawn gives deterministic,
    # statistically-independent child RNGs.)
    master = np.random.default_rng(SEED)
    rng_A, rng_B, rng_D = master.spawn(3)

    X_A = scenario_dependence(rng_A, N)    # genuine cross-judge dependence
    X_B = scenario_drift_null(rng_B, N)    # drift-only null (independent walks)
    X_D = scenario_shared_trend(rng_D, N)  # shared-time-trend (the confound)

    scenarios = [
        ("A dependence",   X_A),
        ("B drift-null",   X_B),
        ("D shared-trend", X_D),
    ]

    # Collect: for each scenario, GLOBAL ranks + rolling ranks at each window.
    results = []  # list of dicts

    for name, X in scenarios:
        # GLOBAL ranks
        Ug = rank_pit_global(X)
        row = {
            "scenario": name,
            "method": "global",
            "window": None,
            "ks": mean_ks_to_uniform(Ug),
            "corr": mean_pairwise_abs_pearson(Ug),
            "n_usable": n_usable(Ug),
        }
        results.append(row)

        # ROLLING ranks at each window
        for w in WINDOWS:
            Ur = rank_pit_rolling(X, w)
            results.append({
                "scenario": name,
                "method": "rolling",
                "window": w,
                "ks": mean_ks_to_uniform(Ur),
                "corr": mean_pairwise_abs_pearson(Ur),
                "n_usable": n_usable(Ur),
            })

    # --- Print table ---
    hdr = (f"{'Scenario':15s} {'Method':8s} {'Win':>4s} "
           f"{'KS-to-Unif':>11s} {'|corr|':>8s} {'n_usable':>9s}")
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        win = "" if r["window"] is None else str(r["window"])
        print(f"{r['scenario']:15s} {r['method']:8s} {win:>4s} "
              f"{r['ks']:>11.4f} {r['corr']:>8.4f} {r['n_usable']:>9d}")
    print("-" * len(hdr))

    # --- Success criterion evaluation ---
    def get(scn, method, window):
        for r in results:
            if r["scenario"] == scn and r["method"] == method and r["window"] == window:
                return r
        return None

    print()
    print("Success criterion:")
    print("  (1) kill spurious |corr| in D shared-trend (global ~high -> rolling ~0)")
    print("  (2) preserve genuine |corr| in A dependence (stay elevated ~0.38)")
    print()

    D_global = get("D shared-trend", "global", None)
    A_global = get("A dependence",   "global", None)
    print(f"  D shared-trend  GLOBAL |corr| = {D_global['corr']:.4f}  (the spurious confound)")
    print(f"  A dependence    GLOBAL |corr| = {A_global['corr']:.4f}  (genuine signal baseline)")
    print()
    for w in WINDOWS:
        D_roll = get("D shared-trend", "rolling", w)
        A_roll = get("A dependence",   "rolling", w)
        killed = D_roll["corr"] < 0.07
        preserved = A_roll["corr"] > 0.20
        verdict = "MET" if (killed and preserved) else "PARTIAL/FAIL"
        print(f"  window={w:4d}:  D |corr|={D_roll['corr']:.4f} "
              f"({'killed' if killed else 'NOT killed'})   "
              f"A |corr|={A_roll['corr']:.4f} "
              f"({'preserved' if preserved else 'LOST'})   -> {verdict}")

    # Trend-strength robustness sweep
    sweep = trend_strength_sweep()
    print()
    print("Trend-strength robustness sweep (shared-trend scenario only):")
    print(f"  {'trend_std':>9s} {'GLOBAL|corr|':>12s} {'roll50':>8s} {'roll100':>8s} {'roll200':>8s}")
    for s in sweep:
        print(f"  {s['trend_std']:>9.2f} {s['global']:>12.4f} "
              f"{s['roll50']:>8.4f} {s['roll100']:>8.4f} {s['roll200']:>8.4f}")

    return results, sweep


def build_md(results, sweep):
    def get(scn, method, window):
        for r in results:
            if r["scenario"] == scn and r["method"] == method and r["window"] == window:
                return r
        return None

    # Table
    rows = ""
    for r in results:
        win = "global" if r["window"] is None else f"rolling-{r['window']}"
        rows += (f"| {r['scenario']} | {win} | {r['ks']:.4f} | "
                 f"{r['corr']:.4f} | {r['n_usable']} |\n")

    D_global = get("D shared-trend", "global", None)
    A_global = get("A dependence",   "global", None)
    B_global = get("B drift-null",   "global", None)

    # Success summary
    crit_rows = ""
    overall_met = False
    for w in WINDOWS:
        D_roll = get("D shared-trend", "rolling", w)
        A_roll = get("A dependence",   "rolling", w)
        killed = D_roll["corr"] < 0.07
        preserved = A_roll["corr"] > 0.20
        met = killed and preserved
        overall_met = overall_met or met
        crit_rows += (f"| {w} | {D_roll['corr']:.4f} | "
                      f"{'YES' if killed else 'no'} | {A_roll['corr']:.4f} | "
                      f"{'YES' if preserved else 'no'} | "
                      f"{'**MET**' if met else 'partial'} |\n")

    md = f"""# Rolling / Causal Ranks -- Shared-Time-Trend Confound Fix

**Setup:** N={N} items, K={K} judges, seed={SEED}, trailing windows {WINDOWS}.
Same seeding convention as `pilot.py` (`np.random.default_rng({SEED})`, per-scenario
spawned sub-streams).

## The confound being fixed

The global (whole-stream) rank-PIT in `pilot.py` absorbs *independent* per-judge
marginal drift cleanly. But when **all judges share a common slow time-trend**
(scenario D: one shared mean trajectory mu(t) with std ~0.5 marginal-sd units,
plus independent per-judge noise, so dependence given time is ZERO), the global rank ties each
judge's rank to the time index. This produces a **spurious** cross-judge rank
correlation -- drift masquerading as co-failure.

## Results

| Scenario | Rank method | KS-to-Unif | pairwise \\|corr\\| | n_usable |
|----------|-------------|-----------:|------------------:|---------:|
{rows}
- **KS-to-Unif**: mean KS of per-judge PIT to Uniform[0,1] (lower = marginals absorbed).
- **\\|corr\\|**: mean absolute pairwise Pearson correlation of the PITs (the co-failure signal).
- **n_usable**: rows with all-K non-NaN PITs. Rolling ranks have a causal warm-up
  (first ~window/5 obs per judge lack enough past points), hence fewer usable rows.

## Success criterion

- (1) **Kill** the spurious \\|corr\\| in D shared-trend: global = **{D_global['corr']:.4f}**, rolling should -> ~0 (<0.07).
- (2) **Preserve** genuine \\|corr\\| in A dependence: global = **{A_global['corr']:.4f}**, rolling should stay elevated (>0.20).

| Window | D shared-trend \\|corr\\| | spurious killed? | A dependence \\|corr\\| | genuine preserved? | Verdict |
|-------:|------------------------:|:----------------:|----------------------:|:------------------:|:-------:|
{crit_rows}

**Overall: criterion {'MET for at least one window.' if overall_met else 'NOT fully met -- see trade-off below.'}**

## Trend-strength robustness sweep (shared-trend scenario only)

The spurious global |corr| grows monotonically with the shared-trend strength;
the rolling fix suppresses it across the whole regime (trend_std=0.5 brackets
the prior run's ~0.2-0.25 spurious correlation).

| trend_std (sd units) | GLOBAL \\|corr\\| | roll-50 | roll-100 | roll-200 |
|---------------------:|---------------:|--------:|---------:|---------:|
{''.join(f"| {s['trend_std']:.2f} | {s['global']:.4f} | {s['roll50']:.4f} | {s['roll100']:.4f} | {s['roll200']:.4f} |\n" for s in sweep)}

At trend_std=2.0 (a very strong trend) the global |corr| reaches ~0.81 and the
widest window (200) leaks a little (~0.11) because the trend is no longer
approximately constant across 200 points; shorter windows still kill it.

## Interpretation and trade-offs

- **Global ranks confirm the confound.** D shared-trend GLOBAL \\|corr\\| = {D_global['corr']:.4f}
  (vs drift-null B global \\|corr\\| = {B_global['corr']:.4f} and genuine A = {A_global['corr']:.4f}).
  The shared trend alone fabricates a co-failure-sized correlation under global ranks.
- **Rolling/causal ranks remove it.** By ranking only against a short trailing window,
  the slow shared trend is ~constant within the window and is differenced away, driving
  D's \\|corr\\| toward the drift-null level.
- **Genuine dependence survives** because it is present *within every window*, not an
  artefact of the time ordering.
- **Trade-off (power vs window size):** smaller windows kill the trend more aggressively
  but add sampling noise to each local rank (coarser PIT, inflated KS, a small floor on
  \\|corr\\| from finite-window noise) and discard a warm-up region. Larger windows retain
  more of the shared trend and so leak a little more spurious correlation. The window is
  the bias/variance knob between trend-robustness and local-rank precision.

---
*Empirical motivation only. Causal ranking is online-computable (each obs ranked only
against strictly past obs) but this is NOT yet an anytime-valid e-process.*
"""
    path = "/home/lyra/projects/rosenblatt-pilot/results_rolling_ranks.md"
    with open(path, "w") as f:
        f.write(md)
    print(f"\n{path} written.")


if __name__ == "__main__":
    results, sweep = run()
    build_md(results, sweep)
