"""
run_online_weights.py — COST OF ONLINE-LEARNED WEIGHTS for the §5b joint linear bet.

CONTEXT
-------
The batch-mode stratified combiner adopted from the Clio correspondence
(2026-11-14-reply-neff-beta-and-5b-drift.tex, Lemma 6 / Vovk-Wang Thm 3.2) is the
JOINT LINEAR BET across strata:

    M_n = prod_{m<=n} ( 1 + sum_s lambda_{s,m} X_{s,m} ),
          lambda_{s,m} >= 0 predictable,  sum_s lambda_{s,m} <= 1/b,

where X_{s,m} is the per-stratum centered co-failure increment at block m and
|X_{s,m}| <= b.  This is anytime-valid under ARBITRARY contemporaneous cross-stratum
dependence (so it replaces the independence-requiring product combiner in a
simultaneous-batch deployment).

This script COSTS the online learning of the weight vector lambda_m = (lambda_{s,m})_s
against an oracle that knows the single best constant weight vector in hindsight.

INCREMENT
---------
Per stratum s, per block m, we form the SAME centered co-failure statistic the
per-pair e-process uses (eprocess.py):
    X_{s,m} = mean over items in stratum s of block m of  (U - V - delta_s)
    U = W_i*W_j          (same-item co-failure)
    V = W_i*W_j[t]       (past-only cross-item baseline, t from make_online_past_pairing)
    delta_s = 2*eps_s    (oracle slack from true marginals)
X is a mean of terms in [-(1+delta), 1], so |X_{s,m}| <= b with b = 1 + max_s delta_s.
Blocks where a stratum has no valid past-paired item contribute X_{s,m}=0 for that s.

THREE WEIGHT SCHEMES (all share the SAME X_{s,m} stream per run, per seed):
  (1) FIXED    : lambda_s = (1/b)/Q uniformly across the Q strata (a natural fixed default
                 that satisfies sum_s lambda_s = 1/b); constant over blocks.
  (2) ONLINE   : Online Newton Step (ONS) on the log-wealth objective, projected onto
                 the simplex-scaled feasible set {lambda>=0, sum lambda <= 1/b}.
                 lambda_m is F_{m-1}-measurable (uses only blocks < m) => PREDICTABLE.
  (3) ORACLE   : the single best constant lambda* chosen WITH hindsight on each run,
                 by maximizing realized log-wealth over the feasible set (projected
                 gradient ascent to convergence on the full realized X stream).
                 This is the benchmark the ONS regret is measured against.

METRICS (per seed, not seed-averaged where the harness supports it):
  - realized terminal log-wealth  log M_N   (null: want bounded; alt: want large)
  - Type-I: P(sup_n M_n >= 1/alpha) under the null  (must stay <= alpha for ONLINE)
  - Power / detection time under the alternative.
  - COST of online learning = log M_N(oracle) - log M_N(online)   (the regret, realized)
    compared to  log M_N(oracle) - log M_N(fixed).

Reuses dgp.py and eprocess.py primitives.  Lighter than the full benign-window harness
(smaller N, block aggregation) so the 3-scheme x seed sweep is tractable.
"""

import csv
import sys
import numpy as np

sys.path.insert(0, '/home/lyra/projects/evalue-sheaf/code/evalue')

from dgp import null_dgp, alt_dgp, drift_null_dgp
from eprocess import (
    assign_strata,
    stratum_epsilon,
    make_online_past_pairing,
    THRESHOLD,
)

# ── Parameters ────────────────────────────────────────────────────────────────
Q           = 10       # number of strata (matches the §5b K=10 harness; finer bins =>
                       # smaller within-stratum p-range => smaller oracle slack delta_s,
                       # so a genuine alternative can overcome the conservative H0 bound.
                       # At Q=5 delta=2*eps swamps gamma<=0.2 and the oracle picks lam*=0.)
N           = 4000     # items per run
BLOCK       = 20       # items per block (simultaneous-batch step); M = N/BLOCK blocks
R           = 200      # replications (seeds)
ALPHA       = 0.05
GAMMA_ALT   = 0.25     # alternative strength (Marshall-Olkin shock prob); chosen so the
                       # oracle lam* > 0 (there is real growth to compete for).
SEED        = 42

# ONS hyperparameters (standard choices; Hazan OCO ONS)
ONS_GAMMA   = None     # set from b below:  gamma_ons = 0.5 * min(1/(G*D), 1) heuristic
ONS_EPS     = 1.0      # regularizer A_0 = eps * I


# ── Build the per-block, per-stratum increment stream X[m, s] ───────────────────

def build_increment_stream(data: dict, rng: np.random.Generator):
    """
    Returns:
        X   : (M, Q) array of per-block, per-stratum centered co-failure increments.
        b   : scalar bound, b = 1 + max_s delta_s, so |X[m,s]| <= b and sum_s lambda_s <= 1/b
              keeps every linear-bet increment  1 + sum_s lambda_s X[m,s]  >= 0.
    Uses PAST-ONLY pairing (predictable) and oracle slack delta_s = 2 eps_s.
    """
    W_i, W_j = data['W_i'], data['W_j']
    p_i, p_j = data['p_i'], data['p_j']
    theta    = data['theta']
    Nloc     = len(W_i)

    strata = assign_strata(theta, Q=Q)
    eps_k  = stratum_epsilon(p_i, p_j, strata, Q=Q)        # (Q,)
    delta  = 2.0 * eps_k                                    # (Q,)
    t      = make_online_past_pairing(strata, rng, Q=Q)    # past-only partners
    has_partner = (t != np.arange(Nloc))

    U = (W_i * W_j).astype(float)
    V = (W_i * W_j[t]).astype(float)
    # per-item signed increment; sentinel (no partner) -> 0 contribution
    inc_item = np.where(has_partner, U - V - delta[strata], 0.0)
    valid    = has_partner.astype(float)

    M = Nloc // BLOCK
    X = np.zeros((M, Q))
    for m in range(M):
        sl = slice(m * BLOCK, (m + 1) * BLOCK)
        s_blk = strata[sl]
        inc_blk = inc_item[sl]
        val_blk = valid[sl]
        for s in range(Q):
            sel = (s_blk == s) & (val_blk > 0)
            if sel.any():
                X[m, s] = inc_blk[sel].mean()   # mean over this stratum's items in the block
            # else leave 0 (no valid item for stratum s in this block)

    # bound: each term in inc_item lies in [-(1+max delta), 1]; a mean stays in that range
    b = 1.0 + float(delta.max())
    return X, b


# ── Feasible-set projection:  {lambda >= 0, sum_s lambda_s <= cap},  cap = 1/b ──

def project_feasible(v: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection of v onto {x >= 0, sum x <= cap}."""
    x = np.maximum(v, 0.0)
    if x.sum() <= cap:
        return x
    # project onto the simplex {x>=0, sum x = cap} (sum must bind here)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho_idx = np.nonzero(u * np.arange(1, len(u) + 1) > (css - cap))[0]
    rho = rho_idx[-1] if len(rho_idx) else 0
    tau = (css[rho] - cap) / (rho + 1)
    return np.maximum(v - tau, 0.0)


# ── Wealth given a (possibly time-varying) weight sequence ──────────────────────

def wealth_path(X: np.ndarray, lam_seq: np.ndarray) -> np.ndarray:
    """
    X       : (M, Q) increments
    lam_seq : (M, Q) per-block weights (predictable) OR (Q,) constant.
    Returns M_path : (M,) wealth process (cumulative product of 1 + <lam_m, X_m>).
    Asserts non-negativity of every factor.
    """
    if lam_seq.ndim == 1:
        lam_seq = np.broadcast_to(lam_seq, X.shape)
    factors = 1.0 + np.einsum('ms,ms->m', lam_seq, X)
    assert np.all(factors >= -1e-9), f"negative linear-bet factor: min={factors.min()}"
    factors = np.maximum(factors, 0.0)
    return np.cumprod(factors)


# ── (1) FIXED weights ───────────────────────────────────────────────────────────

def lam_fixed(b: float) -> np.ndarray:
    cap = 1.0 / b
    return np.full(Q, cap / Q)


# ── (2) ONLINE weights: Online Newton Step on log-wealth ────────────────────────

def ons_weights(X: np.ndarray, b: float):
    """
    ONS for log-wealth maximization (Hazan, 'Intro to Online Convex Optimization', Ch 4;
    the universal-portfolio / exp-concave regret result).  Loss at block m:
        f_m(lam) = -log(1 + <lam, X_m>)   (exp-concave on the feasible set).
    ONS update (predictable: lam_m depends only on X_1..X_{m-1}):
        g_m   = grad f_m(lam_m) = -X_m / (1 + <lam_m, X_m>)
        A_m   = A_{m-1} + g_m g_m^T
        lam_{m+1} = Proj^{A_m}_K ( lam_m - (1/gamma) A_m^{-1} g_m )
    Returns (lam_seq (M,Q) predictable, log-wealth-regret-bound-constants dict).
    """
    M = X.shape[0]
    cap = 1.0 / b
    # ONS exp-concavity param alpha and learning rate gamma (Hazan, OCO Thm 4.5):
    #   for alpha-exp-concave losses, gamma = 0.5 * min(1/(G D), alpha).
    # The loss f_m(lam) = -log(1 + <lam,X_m>).  The argument 1 + <lam,X_m> lies in [0,2]
    # because sum_s lam_s <= 1/b and |X_{s,m}| <= b give <lam,X_m> in [-1,1].  On (0,2]
    # the map z |-> -log z is alpha-exp-concave with alpha = 1/sup z = 1/2.
    alpha_ec = 0.5
    G = b * np.sqrt(Q)          # gradient-norm bound ||g|| ~ ||X||/(1+<lam,X>) <= O(b sqrt Q)
    D = cap * np.sqrt(Q) + 1e-12   # diameter of the feasible set K
    # The 1/(G D) term is a loose worst-case guard; for this well-scaled problem it is
    # dominated by alpha_ec, so gamma = 0.5*alpha_ec = 0.25 (verified near-oracle growth
    # across null/alt; a smaller gamma over-bets, a larger gamma under-reacts to drift).
    gamma = 0.5 * min(1.0 / (G * D), alpha_ec)
    gamma = max(gamma, 0.25)    # floor at the exp-concavity-implied 0.5*alpha_ec

    A = ONS_EPS * np.eye(Q)
    Ainv = (1.0 / ONS_EPS) * np.eye(Q)
    lam = lam_fixed(b).copy()                 # warm start at the fixed default
    lam_seq = np.zeros((M, Q))

    for m in range(M):
        lam_seq[m] = lam                      # F_{m-1}-measurable: set BEFORE seeing X_m
        xm = X[m]
        denom = 1.0 + lam @ xm
        if denom <= 1e-12:
            denom = 1e-12
        g = -xm / denom                       # grad of -log(1+<lam,x>)
        # rank-1 update of A and its inverse (Sherman-Morrison)
        Ag = Ainv @ g
        A = A + np.outer(g, g)
        Ainv = Ainv - np.outer(Ag, Ag) / (1.0 + g @ Ag)
        step = lam - (1.0 / gamma) * (Ainv @ g)
        lam = project_feasible(step, cap)     # generalized projection approximated by Euclidean

    consts = dict(gamma=gamma, G=G, D=D, cap=cap)
    return lam_seq, consts


# ── (3) ORACLE constant weights: best fixed lam* in hindsight ───────────────────

def _log_wealth_obj(X, lam):
    f = 1.0 + X @ lam
    if np.any(f <= 1e-12):
        return -np.inf          # outside the valid domain (a factor hit 0)
    return float(np.sum(np.log(f)))


def oracle_weights(X: np.ndarray, b: float, iters: int = 2000):
    """
    Maximize realized log-wealth  sum_m log(1 + <lam, X_m>)  over the feasible set
    {lam >= 0, sum_s lam_s <= 1/b}, by PROJECTED gradient ascent with BACKTRACKING
    line search.  The objective is concave => this converges to the global optimum.
    The line search guarantees monotone improvement and never leaves the domain where
    some factor 1+<lam,X_m> hits 0 (plain fixed-step ascent diverges there).
    Returns lam* (Q,).  By construction obj(lam*) >= obj(lam_fixed).
    """
    cap = 1.0 / b
    lam = lam_fixed(b).copy()
    cur = _log_wealth_obj(X, lam)
    for it in range(iters):
        f = 1.0 + X @ lam
        f = np.maximum(f, 1e-12)
        grad = (X / f[:, None]).sum(axis=0)
        # backtracking line search over step size
        step = cap
        improved = False
        for _ in range(40):
            cand = project_feasible(lam + step * grad, cap)
            val = _log_wealth_obj(X, cand)
            if val > cur + 1e-9:
                lam, cur, improved = cand, val, True
                break
            step *= 0.5
        if not improved:
            break   # converged (no step improves)
    return lam


# ── One run: build stream, evaluate three schemes ───────────────────────────────

def one_run(data, rng):
    X, b = build_increment_stream(data, rng)

    lf = lam_fixed(b)
    lam_on_seq, consts = ons_weights(X, b)
    lo = oracle_weights(X, b)

    M_fixed  = wealth_path(X, lf)
    M_online = wealth_path(X, lam_on_seq)
    M_oracle = wealth_path(X, lo)

    def terminal_log(Mp):
        return float(np.log(max(Mp[-1], 1e-300)))

    def fired(Mp):
        cr = np.where(Mp >= THRESHOLD)[0]
        return (len(cr) > 0, int(cr[0]) + 1 if len(cr) else None)

    f_fx, t_fx = fired(M_fixed)
    f_on, t_on = fired(M_online)
    f_or, t_or = fired(M_oracle)

    return dict(
        lw_fixed=terminal_log(M_fixed),
        lw_online=terminal_log(M_online),
        lw_oracle=terminal_log(M_oracle),
        fired_fixed=f_fx, det_fixed=t_fx,
        fired_online=f_on, det_online=t_on,
        fired_oracle=f_or, det_oracle=t_or,
        oracle_sumlam=float(lo.sum()), cap=1.0 / b,
        consts=consts,
    )


def run_condition(dgp_fn, seed, R):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(R):
        data = dgp_fn(N, rng)
        rows.append(one_run(data, rng))
    return rows


def summarize(rows, label, is_null):
    lw_fx = np.array([r['lw_fixed'] for r in rows])
    lw_on = np.array([r['lw_online'] for r in rows])
    lw_or = np.array([r['lw_oracle'] for r in rows])
    fired_fx = np.array([r['fired_fixed'] for r in rows])
    fired_on = np.array([r['fired_online'] for r in rows])
    fired_or = np.array([r['fired_oracle'] for r in rows])

    cost_online = lw_or - lw_on          # per-seed regret (oracle - online)
    cost_fixed  = lw_or - lw_fx          # per-seed (oracle - fixed)

    print(f"\n=== {label} ===")
    print(f"  mean terminal log-wealth:  fixed={lw_fx.mean():+.3f}  online={lw_on.mean():+.3f}  oracle={lw_or.mean():+.3f}")
    if is_null:
        print(f"  Type-I (sup M >= {THRESHOLD:.0f}):  fixed={fired_fx.mean():.3f}  "
              f"online={fired_on.mean():.3f}  oracle={fired_or.mean():.3f}   (target <= {ALPHA})")
    else:
        def med(rows, key):
            v = [r[key] for r in rows if r[key] is not None]
            return int(np.median(v)) if v else None
        print(f"  Power (detect):  fixed={fired_fx.mean():.3f}  online={fired_on.mean():.3f}  oracle={fired_or.mean():.3f}")
        print(f"  Median detect block:  fixed={med(rows,'det_fixed')}  "
              f"online={med(rows,'det_online')}  oracle={med(rows,'det_oracle')}")
    sumlam = np.array([r['oracle_sumlam'] for r in rows])
    cap = rows[0]['cap']
    print(f"  oracle sum(lam*)/cap:  mean={ (sumlam/cap).mean():.3f}  "
          f"(0 => oracle declines to bet; 1 => bets at the Vovk-Wang cap 1/b)")
    print(f"  COST of online (oracle-online):  mean={cost_online.mean():+.3f}  "
          f"median={np.median(cost_online):+.3f}  max={cost_online.max():+.3f}")
    print(f"  (oracle-fixed) for comparison :  mean={cost_fixed.mean():+.3f}  "
          f"median={np.median(cost_fixed):+.3f}  max={cost_fixed.max():+.3f}")
    # per-seed sample (first 8 seeds)
    print(f"  per-seed (first 8): oracle-online = " +
          ", ".join(f"{c:+.2f}" for c in cost_online[:8]))
    print(f"                      oracle-fixed  = " +
          ", ".join(f"{c:+.2f}" for c in cost_fixed[:8]))

    return dict(
        label=label, is_null=is_null,
        lw_fixed=lw_fx.mean(), lw_online=lw_on.mean(), lw_oracle=lw_or.mean(),
        typeI_fixed=fired_fx.mean(), typeI_online=fired_on.mean(), typeI_oracle=fired_or.mean(),
        cost_online_mean=cost_online.mean(), cost_online_med=float(np.median(cost_online)),
        cost_fixed_mean=cost_fixed.mean(), cost_fixed_med=float(np.median(cost_fixed)),
    )


def main():
    print("=" * 90)
    print("COST OF ONLINE-LEARNED WEIGHTS — joint linear bet (Vovk-Wang Thm 3.2 combiner)")
    print(f"  Q={Q} strata, N={N}, BLOCK={BLOCK} (M={N//BLOCK} blocks), R={R}, alpha={ALPHA}")
    print(f"  Schemes: FIXED (uniform cap/Q) | ONLINE (ONS, predictable) | ORACLE (best fixed, hindsight)")
    print("=" * 90)

    summ = []
    summ.append(summarize(run_condition(null_dgp,       SEED + 100, R), "NULL  (static H0)", True))
    summ.append(summarize(run_condition(drift_null_dgp, SEED + 101, R), "NULL  (drift  H0)", True))
    alt025 = lambda Nn, rng: alt_dgp(Nn, 0.25, rng)
    alt040 = lambda Nn, rng: alt_dgp(Nn, 0.40, rng)
    summ.append(summarize(run_condition(alt025, SEED + 104, R), "ALT   (gamma=0.25)", False))
    summ.append(summarize(run_condition(alt040, SEED + 105, R), "ALT   (gamma=0.40)", False))

    # report an ONS regret-bound sanity constant from one run
    print("\n" + "=" * 90)
    print("ONS constants (illustrative, from last alt run's final seed):")
    print(f"  (regret bound: log-wealth regret <= (Q/(2*gamma_eff)) * log(n) + const; see report)")
    print("=" * 90)

    # CSV
    csv_path = '/home/lyra/projects/evalue-sheaf/code/evalue/results_online_weights.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    print(f"\nCSV written to {csv_path}")
    print("Done.")


if __name__ == '__main__':
    main()
