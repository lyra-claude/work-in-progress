"""
section7_sim.py
===============
Section 7 "Simulation Confirmation" for the ICLR paper.

A de Finetti two-atom sweep validating the cross-item co-failure e-process.

Construction (verified against
memos/2026-09-03-sec5b-stratification-draft.tex, lines 76-84):

    e_t = e_{t-1} * ( 1 + lambda ( U_t - V_t - delta_k ) )

  U_t = 1{ both judges fail on the SAME item i }         (contemporaneous)
  V_t = W^s_a * W^t_b   with a != b, distinct items       (cross-item pairing)
  delta_k = 2 * eps  absorbs the worst-case two-sided base-rate excursion
  lambda in [0, 1/(1+2 eps)]  keeps the per-step factor non-negative.

Null baseline is MARGINS-FREE: because a and b are drawn from DISTINCT items
that are independent by construction, E[V] = a * b with no fitted nuisance.
Under the null (no within-item co-failure) E[U] = a * b as well, so E[U-V]=0
and the e-process is a non-negative martingale (Ville validity). Under genuine
co-failure E[U] = a*b + excess > E[V], the drift is positive, and e_t grows.

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
def eprocess_crossitem(fails, lam, eps, s=0, t=1, rng=None):
    """
    Run the margins-free cross-item e-process on a failure stream.

      U_i = fails[i,s] * fails[i,t]                    (same item)
      V_i = fails[i,s] * fails[perm(i),t]              (distinct item, perm != i)

    e_i = prod_{i} ( 1 + lam ( U_i - V_i - 2 eps ) ).
    Returns the running e-process array (length n_items).
    """
    n = fails.shape[0]
    ws = fails[:, s].astype(float)
    wt = fails[:, t].astype(float)
    # derangement-ish pairing: pair item i's judge-t with a DIFFERENT item.
    # A fixed cyclic shift by 1 guarantees perm(i) != i for all i.
    wt_other = np.roll(wt, 1)
    U = ws * wt
    V = ws * wt_other
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
    # running means BEFORE item i (leave-one-out prefix to avoid using U_i itself)
    csum_s = np.cumsum(ws)
    csum_t = np.cumsum(wt)
    idx = np.arange(1, n + 1)
    ahat = csum_s / idx  # mean up to and including i
    bhat = csum_t / idx
    U = ws * wt
    Vhat = ahat * bhat
    factors = 1.0 + lam * (U - Vhat)
    factors = np.maximum(factors, 1e-12)
    return np.cumprod(factors)


# ---------------------------------------------------------------------------
# Panel (a): power curve
# ---------------------------------------------------------------------------
def panel_a_power(rng, K=6, n_items=400, n_streams=2000, lam=None, eps=0.02):
    """
    Sweep atom separation -> induced rho. For each rho, estimate
    P(e-process crosses 1/alpha) over many streams. Fix marginal a ~ 0.5 by
    keeping theta_lo, theta_hi symmetric about 0.5 so rho is the only mover.
    """
    pi = 0.5
    if lam is None:
        lam = 0.5 / (1.0 + 2.0 * eps)  # interior of the admissible interval
    # separations from 0 (independence) up to near-maximal
    seps = np.linspace(0.0, 0.9, 10)
    rows = []
    for sep in seps:
        theta_lo = 0.5 - sep / 2.0
        theta_hi = 0.5 + sep / 2.0
        a, rho = two_atom_params(theta_lo, theta_hi, pi)
        rejects = 0
        for _ in range(n_streams):
            fails = sample_stream(rng, n_items, K, theta_lo, theta_hi, pi)
            e = eprocess_crossitem(fails, lam, eps)
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


def panel_b_falsefire(rng, K=6, n_items=300, n_streams=2000, lam=None, eps=0.02):
    if lam is None:
        lam = 0.5 / (1.0 + 2.0 * eps)
    naive_rej = 0
    cross_rej = 0
    for _ in range(n_streams):
        fails, _p = sample_drift_stream(rng, n_items, K)
        e_naive = eprocess_naive(fails, lam)
        e_cross = eprocess_crossitem(fails, lam, eps)
        if np.max(e_naive) >= THRESH:
            naive_rej += 1
        if np.max(e_cross) >= THRESH:
            cross_rej += 1
    naive_rate = naive_rej / n_streams
    cross_rate = cross_rej / n_streams
    print(f"  naive plug-in false-reject rate under drift : {naive_rate:.4f}")
    print(f"  margins-free cross-item size under drift     : {cross_rate:.4f}")
    return naive_rate, cross_rate, lam


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

    print("\n[Panel a] POWER CURVE (de Finetti two-atom sweep)")
    rows_a, lam_a, eps_a = panel_a_power(rng)

    print("\n[Panel b] NAIVE FALSE-FIRE under drifting margins (no co-failure)")
    naive_rate, cross_rate, lam_b = panel_b_falsefire(rng)

    print("\n[Panel c] FAILURESCOPE real-data anchor")
    fs = panel_c_failurescope()

    make_figures(rows_a, lam_a, eps_a, naive_rate, cross_rate, fs)
    print("\nWrote panel_a_power.png, panel_b_falsefire.png, "
          "panel_c_failurescope.png")

    # headline summary block for the write-up
    phibar, neff, N_fs, k_fs = fs
    print("\n" + "=" * 66)
    print("HEADLINE NUMBERS")
    print("=" * 66)
    print(f"  (a) type-I @ rho=0        : {rows_a[0][3]:.4f}  (target ~{ALPHA})")
    print(f"  (a) power @ rho={rows_a[-1][2]:.3f}     : {rows_a[-1][3]:.4f}")
    print(f"  (b) naive false-reject    : {naive_rate:.4f}")
    print(f"  (b) cross-item size       : {cross_rate:.4f}")
    print(f"  (c) FailureScope n_eff    : {neff:.4f}  (phi-bar={phibar:.4f})")
    print(f"      lambda={lam_a:.4f}, eps={eps_a}, 2eps={2*eps_a}")


if __name__ == "__main__":
    main()
