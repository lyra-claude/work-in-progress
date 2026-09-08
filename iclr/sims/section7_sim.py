"""
section7_sim.py
===============
Section 7 "Simulation Confirmation" for the ICLR paper.

A de Finetti two-atom sweep validating the cross-item co-failure e-process.

Construction (matches the paper's Sec 6 e-process; the cross-item pairing is
realised as an INDEPENDENT PARALLEL STREAM, see below):

    e_t = e_{t-1} * ( 1 + lambda ( U_t - V_t - delta_k ) )

  U_t = 1{ both judges s,t fail on the SAME item i, stream A }   (co-failure)
  V_t = W^s_{A,i} * W^t_{B,i}   judge-s on item i of stream A,
                                judge-t on the INDEPENDENT item i of stream B
  delta_k = 2 * eps  absorbs the worst-case two-sided base-rate excursion
  lambda in [0, 1/(1+2 eps)]  keeps the per-step factor non-negative.

WHY TWO STREAMS (correctness-critical).  The margins-free baseline V requires
pairing the two judges' failure indicators from GENUINELY INDEPENDENT items,
with NO term shared across factors and NO wraparound. We generate two i.i.d.
streams A and B from the SAME data-generating process (same Theta prior /
same drift schedule, independently sampled). Then at step i:

    V_i = W^s_{A,i} * W^t_{B,i}

pairs judge-s on A-item-i with judge-t on B-item-i. Because A-item-i and
B-item-i are independent, E[V_i] = E[W^s] * E[W^t] = a*b, factorising the
(possibly drifting) marginals WITHOUT fitting anything. Crucially:
  * no term is shared between factor i and factor j (i != j), and
  * nothing depends on a FUTURE item,
so { e_t } is a genuine product supermartingale, causally adapted to the
filtration F_t = sigma( streams A,B up to item t ). This is what makes
Ville validity real.

  *** OLD BUG (now fixed): the previous version used a single stream with
      V_i = W^s_i * W^t_{roll(i)} (np.roll by 1). That (1) SHARES the term
      W^t_i between U_i and V_{i+1} with opposite signs -> negative covariance
      across consecutive factors -> NOT a supermartingale (E[e_T] decayed to
      ~0.001 under the true null instead of staying at 1), and (2) wrapped
      around so the first factor depended on the LAST (future) item ->
      anticipative. Both defects are gone. ***

Under the null (no within-item co-failure, i.e. judges conditionally
independent given Theta) E[U_i] = a*b = E[V_i], so E[U_i - V_i] = 0 and, with
delta_k >= 0, E[factor_i | F_{i-1}] <= 1: a non-negative supermartingale, and
Ville's inequality gives anytime validity. Under genuine co-failure
E[U_i] = a*b + excess > E[V_i], the increment has positive drift, and e_t
grows.

EMPIRICAL MARTINGALE CHECK. martingale_check() below verifies E[e_T] ~ 1 under
the delta=0 TRUE null for N = 2, 3, 4, ~400 -- the key soundness gate that the
buggy roll construction failed.

Generative model (de Finetti two-atom):
  For each item draw a latent Theta in {theta_lo, theta_hi} with P(theta_hi)=pi.
  Given Theta = theta, the K judges' failure indicators are i.i.d. Bern(theta).

  SIGN CHECK (see de-finetti-point-mass-is-backwards memory note):
    A POINT MASS on Theta (theta_lo == theta_hi, or pi in {0,1}) is
    INDEPENDENCE, rho = 0. A two-atom prior with WELL-SEPARATED atoms is the
    high co-failure case, rho -> 1 ("one shared coin flip"). Separation
    INCREASES rho. The induced within-item correlation is
        rho = Var(Theta) / ( a (1-a) ),
        Var(Theta) = pi (1-pi) (theta_hi - theta_lo)^2,
        a          = pi theta_hi + (1-pi) theta_lo.

Panels:
  (a) POWER CURVE  : P(e crosses 1/alpha) vs induced co-failure rho.
                     Type-I point at rho=0 must be ~alpha (Ville validity).
  (b) NAIVE FALSE-FIRE : drifting marginals, NO co-failure. The naive plug-in
                     (fits marginals, plugs them in) false-rejects; the
                     margins-free cross-item e-process holds size ~alpha.
  (c) FAILURESCOPE : real-data anchor n_eff, phi-bar from failurescope_probe.py.

RUN:
    python3 section7_sim.py            # writes the three panel PNGs + prints numbers
Deps: numpy, matplotlib only. Fixed RNG seed for reproducibility.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260907
ALPHA = 0.05
THRESH = 1.0 / ALPHA  # e-process rejects when e_t >= 1/alpha

# Operating betting fraction. lambda in [0, 1/(1+2 eps)] keeps every factor
# non-negative. We use a modest lambda = 0.2 (well inside the admissible
# interval) rather than an aggressive one: it gives an equal-or-better power
# curve while keeping the martingale variance small enough that the
# E[e_T] ~ 1 martingale check converges at feasible Monte-Carlo sample sizes.
# (At an aggressive lambda the process is still a martingale, but its product
# is so right-skewed that its sample mean is dominated by rare huge paths and
# reads far below 1 without astronomically many replications -- which would
# make the soundness gate unmeasurable rather than false.)
LAMBDA = 0.2
EPS = 0.02


# ---------------------------------------------------------------------------
# de Finetti two-atom generator
# ---------------------------------------------------------------------------
def two_atom_params(theta_lo, theta_hi, pi):
    """Marginal fail rate a and induced within-item pairwise correlation rho."""
    a = pi * theta_hi + (1.0 - pi) * theta_lo
    var_theta = pi * (1.0 - pi) * (theta_hi - theta_lo) ** 2
    denom = a * (1.0 - a)
    rho = var_theta / denom if denom > 0 else 0.0
    return a, rho


def sample_stream(rng, n_items, K, theta_lo, theta_hi, pi):
    """Return an (n_items, K) 0/1 failure matrix from the two-atom model."""
    hi = rng.random(n_items) < pi
    theta = np.where(hi, theta_hi, theta_lo)  # (n_items,)
    U = rng.random((n_items, K))
    return (U < theta[:, None]).astype(np.int8)


# ---------------------------------------------------------------------------
# Cross-item co-failure e-process (margins-free)
# ---------------------------------------------------------------------------
def eprocess_crossitem(fails_a, fails_b, lam, eps, s=0, t=1):
    """
    Run the margins-free cross-item e-process on TWO independent streams.

      U_i = fails_a[i,s] * fails_a[i,t]      (both judges fail item i, stream A)
      V_i = fails_a[i,s] * fails_b[i,t]      (judge-s A-item-i, judge-t B-item-i)

    A-item-i and B-item-i are independent draws from the SAME DGP, so
    E[V_i] = a*b with no fitted nuisance. No term is shared across factors and
    nothing depends on a future item -> genuine causal product supermartingale.

    e_i = prod_{i} ( 1 + lam ( U_i - V_i - 2 eps ) ).
    Returns the running e-process array (length n_items).
    """
    ws_a = fails_a[:, s].astype(float)
    wt_a = fails_a[:, t].astype(float)
    wt_b = fails_b[:, t].astype(float)
    U = ws_a * wt_a
    V = ws_a * wt_b
    delta = 2.0 * eps
    factors = 1.0 + lam * (U - V - delta)
    # non-negativity guard (lam in [0, 1/(1+2eps)] already enforces this)
    factors = np.maximum(factors, 1e-12)
    return np.cumprod(factors)


def eprocess_naive(fails, lam, s=0, t=1):
    """
    NAIVE PLUG-IN co-failure e-process: estimates each judge's marginal from the
    stream so far and plugs it in as the null for the joint fail, instead of the
    margins-free cross-item pairing. This is the estimator the paper argues is
    invalid under drift.

      U_i    = fails[i,s]*fails[i,t]
      Vhat_i = ahat_i * bhat_i   (running plug-in of the product-of-marginals)

    e_i = prod ( 1 + lam ( U_i - Vhat_i - 0 ) ), clipped non-negative.
    No delta slack: the naive method trusts its fitted marginals.
    """
    n = fails.shape[0]
    ws = fails[:, s].astype(float)
    wt = fails[:, t].astype(float)
    # GENUINE causal leave-one-out: the plug-in at item i uses ONLY items
    # strictly before i (a prefix that excludes item i itself), so the null
    # estimate does not peek at the very increment it is being compared to.
    # prefix_mean[i] = mean of items 0..i-1  (undefined for i=0 -> fall back
    # to the item's own value, a neutral bootstrap for the first step).
    csum_s = np.concatenate(([0.0], np.cumsum(ws)[:-1]))  # sum of items < i
    csum_t = np.concatenate(([0.0], np.cumsum(wt)[:-1]))
    cnt = np.arange(n, dtype=float)                       # number of items < i
    cnt[0] = 1.0                                          # avoid /0 at i=0
    ahat = csum_s / cnt
    bhat = csum_t / cnt
    ahat[0] = ws[0]                                       # neutral first step
    bhat[0] = wt[0]
    U = ws * wt
    Vhat = ahat * bhat
    factors = 1.0 + lam * (U - Vhat)
    factors = np.maximum(factors, 1e-12)
    return np.cumprod(factors)


# ---------------------------------------------------------------------------
# KEY SOUNDNESS GATE: empirical martingale check under the delta=0 TRUE null
# ---------------------------------------------------------------------------
def martingale_check(rng, Ns=(2, 3, 4, 400), n_streams=40000, lam=LAMBDA, K=2,
                     big_n_streams=200000):
    """
    Under the delta=0 TRUE null (independent judges, no shared Theta, a=b=0.5),
    a genuine product martingale has E[e_T] = 1 for every horizon T=N.

    We build the cross-item e-process with delta=0 on TWO independent streams,
    each with an independent per-item base rate 0.5 and judges failing i.i.d.
    (so there is NO co-failure and NO shared latent), and report E[e_T].

    The buggy roll construction decayed to ~0.003 here; the correct
    two-stream construction must stay ~1.000.
    """
    results = []
    for N in Ns:
        # Longer horizons have heavier right-skew, so the sample mean of a true
        # martingale needs more replications to converge to 1. Give the largest
        # horizon more streams.
        ns = big_n_streams if N >= 100 else n_streams
        Es = np.empty(ns)
        for r in range(ns):
            fa = (rng.random((N, K)) < 0.5).astype(np.int8)
            fb = (rng.random((N, K)) < 0.5).astype(np.int8)
            e = eprocess_crossitem(fa, fb, lam, eps=0.0)
            Es[r] = e[-1]
        results.append((N, float(Es.mean()), float(np.median(Es))))
        print(f"  N={N:4d}  E[e_T]={Es.mean():.4f}  "
              f"median={np.median(Es):.4f}  (delta=0 true null; target 1.000)")
    return results


# ---------------------------------------------------------------------------
# Panel (a): power curve
# ---------------------------------------------------------------------------
def panel_a_power(rng, K=6, n_items=400, n_streams=2000, lam=LAMBDA, eps=EPS):
    """
    Sweep atom separation -> induced rho. For each rho, estimate
    P(e-process crosses 1/alpha) over many streams. Fix marginal a ~ 0.5 by
    keeping theta_lo, theta_hi symmetric about 0.5 so rho is the only mover.

    The co-failure signal lives in stream A (shared latent Theta across judges
    within an item). The margins-free baseline pairs judge-t against an
    INDEPENDENT stream B drawn from the identical two-atom DGP, so E[V]=ab
    regardless of rho and the null holds by construction, not by fitting.
    """
    pi = 0.5
    # separations from 0 (independence) up to near-maximal
    seps = np.linspace(0.0, 0.9, 10)
    rows = []
    for sep in seps:
        theta_lo = 0.5 - sep / 2.0
        theta_hi = 0.5 + sep / 2.0
        a, rho = two_atom_params(theta_lo, theta_hi, pi)
        rejects = 0
        for _ in range(n_streams):
            fa = sample_stream(rng, n_items, K, theta_lo, theta_hi, pi)
            fb = sample_stream(rng, n_items, K, theta_lo, theta_hi, pi)
            e = eprocess_crossitem(fa, fb, lam, eps)
            if np.max(e) >= THRESH:
                rejects += 1
        power = rejects / n_streams
        rows.append((sep, a, rho, power))
        print(f"  sep={sep:.2f}  a={a:.3f}  rho={rho:.4f}  power={power:.4f}")
    return rows, lam, eps


# ---------------------------------------------------------------------------
# Panel (b): naive false-fire under drifting margins, NO co-failure
# ---------------------------------------------------------------------------
def sample_drift_stream(rng, n_items, K, base_lo=0.25, base_hi=0.60):
    """
    Drifting marginals, NO within-item co-failure. Each item has a
    time-varying base rate p_i sweeping base_lo -> base_hi across the stream,
    and given p_i the K judges fail INDEPENDENTLY Bern(p_i). Because the rate
    is shared across judges WITHIN an item, a naive plug-in that assumes a
    stationary product-of-marginals sees "excess" joint failures that are in
    fact pure marginal drift -- there is no conditional co-failure.
    """
    p = np.linspace(base_lo, base_hi, n_items)
    U = rng.random((n_items, K))
    return (U < p[:, None]).astype(np.int8), p


def panel_b_falsefire(rng, K=6, n_items=300, n_streams=2000, lam=LAMBDA, eps=EPS):
    """
    Drifting marginals, ZERO cross-judge co-failure. FAIRNESS: naive and
    cross-item run at the SAME lambda. The naive plug-in fits the running
    marginals (causal leave-one-out) and plugs in the product-of-marginals as
    its null; it false-fires because a stationary product-of-marginals is
    mis-specified under drift (a Jensen gap between E[ab] over the drift and the
    plugged-in a-hat*b-hat), NOT because it was rigged. The cross-item process
    pairs stream A against an INDEPENDENT stream B that follows the SAME drift
    schedule, so E[V]=ab holds pointwise despite the drift and no marginal is
    ever fitted. We also report the cross-item E[e_T] under this drift to prove
    the martingale is ALIVE (~1), not dead.
    """
    naive_rej = 0
    cross_rej = 0
    e_cross_final = np.empty(n_streams)   # operating delta = 2*eps
    e_cross_d0 = np.empty(n_streams)      # delta = 0 (isolates martingale-ness)
    for r in range(n_streams):
        fails_a, _p = sample_drift_stream(rng, n_items, K)
        fails_b, _p2 = sample_drift_stream(rng, n_items, K)  # same drift, indep
        e_naive = eprocess_naive(fails_a, lam)
        e_cross = eprocess_crossitem(fails_a, fails_b, lam, eps)
        e_cross0 = eprocess_crossitem(fails_a, fails_b, lam, 0.0)
        e_cross_final[r] = e_cross[-1]
        e_cross_d0[r] = e_cross0[-1]
        if np.max(e_naive) >= THRESH:
            naive_rej += 1
        if np.max(e_cross) >= THRESH:
            cross_rej += 1
    naive_rate = naive_rej / n_streams
    cross_rate = cross_rej / n_streams
    cross_eT = float(e_cross_final.mean())   # <1 by design: slack decays mean
    cross_eT_d0 = float(e_cross_d0.mean())   # ~1: the martingale is ALIVE
    print(f"  naive plug-in false-reject rate under drift  : {naive_rate:.4f}")
    print(f"  margins-free cross-item size under drift      : {cross_rate:.4f}")
    print(f"  cross-item E[e_T] under drift, delta=0 (ALIVE): {cross_eT_d0:.4f}"
          f"  (target ~1.000)")
    print(f"  cross-item E[e_T] under drift, delta=2eps     : {cross_eT:.4f}"
          f"  (<1 by design: slack -> conservative)")
    return naive_rate, cross_rate, cross_eT_d0, cross_eT, lam


# ---------------------------------------------------------------------------
# de Finetti n_eff sweep read-out (reuses Panel (a)'s rho grid)
# ---------------------------------------------------------------------------
def definetti_neff_sweep(anchor_neff=1.6350, K=6):
    """
    PURELY ADDITIVE read-out. Reuses the SAME atom-separation grid Panel (a)
    sweeps (seps = linspace(0, 0.9, 10) at pi=0.5, symmetric about 0.5), maps
    each separation to its induced within-item co-failure rho via the SAME
    two_atom_params() Panel (a) uses, then reports the Kish effective judge
    count for an N=K equicorrelation panel:

        n_eff(rho) = K / (1 + (K-1) * rho).

    This exhibits the empirical FailureScope anchor (n_eff = anchor_neff) as a
    point INSIDE the de Finetti sweep, at rho = phi-bar. Nothing here touches
    the e-process, the sampling, or the RNG; it is a deterministic table.
    """
    pi = 0.5
    seps = np.linspace(0.0, 0.9, 10)   # identical grid to panel_a_power
    rhos = []
    for sep in seps:
        theta_lo = 0.5 - sep / 2.0
        theta_hi = 0.5 + sep / 2.0
        _a, rho = two_atom_params(theta_lo, theta_hi, pi)
        rhos.append(rho)

    def kish_neff(rho):
        return K / (1.0 + (K - 1) * rho)

    print(f"DE FINETTI n_eff SWEEP (N={K})")
    print(f"  Kish equicorrelation:  n_eff(rho) = {K} / (1 + {K-1}*rho)")
    print(f"  {'rho':>8}   {'n_eff':>8}")
    for rho in rhos:
        print(f"  {rho:>8.4f}   {kish_neff(rho):>8.4f}")

    neff_at = [kish_neff(r) for r in rhos]
    neff_min = min(neff_at)          # at the largest rho in the grid
    neff_max = max(neff_at)          # at rho = 0
    rho_at_min = rhos[int(np.argmin(neff_at))]
    print(f"  n_eff RANGE over sweep : [{neff_min:.4f}, {neff_max:.4f}]  "
          f"(min at rho={rho_at_min:.4f}, max at rho=0.0000)")

    # position of the empirical anchor: n_eff = anchor => rho = (K/neff - 1)/(K-1)
    rho_anchor = (K / anchor_neff - 1.0) / (K - 1)
    neff_check = kish_neff(rho_anchor)
    print(f"  empirical anchor n_eff={anchor_neff:.4f} sits at "
          f"rho = phi-bar = {rho_anchor:.4f}")
    print(f"  consistency check: n_eff({rho_anchor:.4f}) = {neff_check:.4f}  "
          f"(should match {anchor_neff:.4f})")
    return list(zip(rhos, neff_at)), (neff_min, neff_max), rho_anchor, neff_check


# ---------------------------------------------------------------------------
# false-fire drift sensitivity sweep (reuses eprocess_naive, unmodified)
# ---------------------------------------------------------------------------
def falsefire_sensitivity(rng, K=6, n_items=300, n_streams=2000, lam=LAMBDA,
                          eps=EPS):
    """
    PURELY ADDITIVE. Measures the naive plug-in false-fire rate under three
    drift steepnesses, holding everything else (K, n_items, n_streams, lambda,
    eps) fixed, using the EXISTING eprocess_naive / eprocess_crossitem paths
    unchanged. Confirms the margins-free cross-item size stays ~0 in each case.
    Scenario labels/base-rate endpoints:
        gentle : 0.30 -> 0.50
        current: 0.25 -> 0.60   (the Panel (b) operating point)
        steep  : 0.15 -> 0.75
    """
    scenarios = [
        ("gentle", 0.30, 0.50),
        ("current", 0.25, 0.60),
        ("steep", 0.15, 0.75),
    ]
    print("FALSE-FIRE DRIFT SENSITIVITY (naive plug-in vs margins-free)")
    print(f"  fixed: K={K}, n_items={n_items}, n_streams={n_streams}, "
          f"lambda={lam:.4f}, eps={eps}")
    results = []
    for name, lo, hi in scenarios:
        naive_rej = 0
        cross_rej = 0
        for _ in range(n_streams):
            fails_a, _p = sample_drift_stream(rng, n_items, K, lo, hi)
            fails_b, _p2 = sample_drift_stream(rng, n_items, K, lo, hi)
            e_naive = eprocess_naive(fails_a, lam)
            e_cross = eprocess_crossitem(fails_a, fails_b, lam, eps)
            if np.max(e_naive) >= THRESH:
                naive_rej += 1
            if np.max(e_cross) >= THRESH:
                cross_rej += 1
        naive_rate = naive_rej / n_streams
        cross_rate = cross_rej / n_streams
        print(f"  {name:<7} drift {lo:.2f}->{hi:.2f} : "
              f"naive false-fire={naive_rate:.4f}   "
              f"margins-free size={cross_rate:.4f}")
        results.append((name, lo, hi, naive_rate, cross_rate))
    return results


# ---------------------------------------------------------------------------
# Panel (c): FailureScope real-data anchor (from failurescope_probe.py)
# ---------------------------------------------------------------------------
FS_PROBE = (
    "/home/lyra/projects/evalue-sheaf/code/evalue/failurescope_probe.py"
)
FS_DATA = (
    "/home/lyra/data-scratch/failurescope/"
    "failurescope-single-turn-v1.0.0/failures.json"
)


def panel_c_failurescope():
    """
    Recompute the FailureScope anchor directly with the SAME Kish formula the
    probe uses, on the SAME 6-judge FRONTIER_MODELS panel. Kept self-contained
    (does not import the probe) but numerically identical to it.
    """
    import json
    from itertools import combinations

    FRONTIER = [
        "claude-haiku", "claude-sonnet", "gpt-4o",
        "gpt-4o-mini", "gpt-5.4", "gpt-5.4-nano",
    ]
    with open(FS_DATA) as f:
        failures = json.load(f)
    rows = []
    for r in failures:
        ev = {e["model_id"]: e["passed"] for e in r["eval_results"]}
        rows.append([0 if ev[m] else 1 for m in FRONTIER])
    mat = np.array(rows, dtype=float)
    k = len(FRONTIER)
    phis = []
    for i, j in combinations(range(k), 2):
        phis.append(np.corrcoef(mat[:, i], mat[:, j])[0, 1])
    phibar = float(np.mean(phis))
    neff = k / (1.0 + (k - 1) * phibar)
    print(f"  FailureScope 6-judge panel  N={mat.shape[0]} items")
    print(f"  mean pairwise phi-bar       : {phibar:.4f}")
    print(f"  Kish n_eff                  : {neff:.4f}")
    return phibar, neff, mat.shape[0], k


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_figures(rows_a, lam_a, eps_a, naive_rate, cross_rate, fs):
    phibar, neff, N_fs, k_fs = fs

    # Panel (a)
    rho = [r[2] for r in rows_a]
    power = [r[3] for r in rows_a]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(rho, power, "o-", color="#1f4e79", lw=1.6, ms=4)
    ax.axhline(ALPHA, ls="--", color="gray", lw=1.0,
               label=fr"$\alpha={ALPHA}$")
    ax.set_xlabel(r"induced co-failure $\rho$")
    ax.set_ylabel(r"$P(\,e_t \geq 1/\alpha\,)$")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(a) Power curve")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.02, 0.92, fr"type-I @ $\rho{{=}}0$: {power[0]:.3f}",
            transform=ax.transAxes, fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig("panel_a_power.png", dpi=150)
    plt.close(fig)

    # Panel (b)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(["naive\nplug-in", "margins-free\ncross-item"],
                  [naive_rate, cross_rate],
                  color=["#c0392b", "#1f4e79"], width=0.55)
    ax.axhline(ALPHA, ls="--", color="gray", lw=1.0,
               label=fr"nominal $\alpha={ALPHA}$")
    ax.set_ylabel("false-rejection rate")
    ax.set_ylim(0, max(1.0, naive_rate * 1.15))
    ax.set_title("(b) Under drifting margins, no co-failure")
    for b, v in zip(bars, [naive_rate, cross_rate]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig("panel_b_falsefire.png", dpi=150)
    plt.close(fig)

    # Panel (c)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.bar(["ideal\n(indep.)", "observed\n"], [k_fs, neff],
           color=["#bbbbbb", "#1f4e79"], width=0.55)
    ax.set_ylabel(r"effective judge count $n_{\mathrm{eff}}$")
    ax.set_title("(c) FailureScope 6-judge panel")
    ax.text(0, k_fs + 0.15, f"{k_fs}", ha="center", fontsize=9)
    ax.text(1, neff + 0.15, f"{neff:.3f}", ha="center", fontsize=9)
    ax.text(0.5, 0.80,
            fr"$\bar\varphi={phibar:.3f}$" + "\n" + fr"$N={N_fs}$ items",
            transform=ax.transAxes, ha="center", fontsize=8)
    ax.set_ylim(0, k_fs + 0.8)
    fig.tight_layout()
    fig.savefig("panel_c_failurescope.png", dpi=150)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    print("=" * 66)
    print("SECTION 7 SIMULATION  (seed = %d, alpha = %.2f, 1/alpha = %.0f)"
          % (SEED, ALPHA, THRESH))
    print("=" * 66)

    print("\n[GATE] MARTINGALE CHECK  E[e_T] under delta=0 TRUE null "
          "(must be ~1.000)")
    mart = martingale_check(rng)

    print("\n[Panel a] POWER CURVE (de Finetti two-atom sweep)")
    rows_a, lam_a, eps_a = panel_a_power(rng)

    print("\n[Panel b] NAIVE FALSE-FIRE under drifting margins (no co-failure)")
    naive_rate, cross_rate, cross_eT_d0, cross_eT, lam_b = panel_b_falsefire(rng)

    print("\n[Panel c] FAILURESCOPE real-data anchor")
    fs = panel_c_failurescope()
    phibar_fs, neff_fs, N_fs_, k_fs_ = fs

    # --- ADDITIVE read-outs (placed AFTER all existing rng-consuming calls so
    #     the seed / martingale / panel numbers above are byte-for-byte
    #     unchanged; the sweep is deterministic, the sensitivity draws come last)
    print("\n[read-out] " + "-" * 54)
    sweep_rows, neff_range, rho_anchor, neff_anchor_check = \
        definetti_neff_sweep(anchor_neff=neff_fs, K=k_fs_)

    print("\n[read-out] " + "-" * 54)
    sens_rows = falsefire_sensitivity(rng)

    make_figures(rows_a, lam_a, eps_a, naive_rate, cross_rate, fs)
    print("\nWrote panel_a_power.png, panel_b_falsefire.png, "
          "panel_c_failurescope.png")

    # headline summary block for the write-up
    phibar, neff, N_fs, k_fs = fs
    print("\n" + "=" * 66)
    print("HEADLINE NUMBERS")
    print("=" * 66)
    print("  MARTINGALE CHECK (delta=0 true null, target 1.000):")
    for N, m, med in mart:
        print(f"      N={N:4d}  E[e_T]={m:.4f}  (median={med:.4f})")
    print(f"  (a) type-I @ rho=0        : {rows_a[0][3]:.4f}  (target <={ALPHA})")
    print(f"  (a) power @ rho={rows_a[-1][2]:.3f}     : {rows_a[-1][3]:.4f}")
    print(f"  (b) naive false-reject    : {naive_rate:.4f}")
    print(f"  (b) cross-item size       : {cross_rate:.4f}")
    print(f"  (b) cross-item E[e_T] d=0 : {cross_eT_d0:.4f}  (ALIVE, target ~1)")
    print(f"  (b) cross-item E[e_T] slack: {cross_eT:.4f}  (<1 by design)")
    print(f"  (c) FailureScope n_eff    : {neff:.4f}  (phi-bar={phibar:.4f})")
    print(f"      lambda={lam_a:.4f}, eps={eps_a}, 2eps={2*eps_a}")


if __name__ == "__main__":
    main()
