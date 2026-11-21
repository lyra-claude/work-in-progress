"""
census_prop1_independent.py

INDEPENDENT peer-review census of Proposition 1 (fresh-pair cyclic drift bound).
Authored from scratch for adversarial verification. Does NOT import or reuse any
existing census/slack code. All quantities derived from the math in the proposition.

Proposition 1 (as stated for review):
  Per-cell increment
      X = (1/n) sum_r [ W_i(s_r) * ( W_j(s_r) - W_j(s_{sigma_r}) ) - delta ]
  where sigma is the cyclic shift r -> (r+1) mod n.  Under H0 the conditional mean is
      E[X | F_{m-1}] = (1/n) sum_r (p_i(s_r) - c)(p_j(s_r) - p_j(s_{sigma_r})) - delta
  for EVERY real c (cyclic-sum identity makes c free).  With
      delta = (1/2) R_i R_j,   R_i = max_r p_i - min_r p_i,  R_j likewise,
  the drift term T := (1/n) sum_r (p_i - c)(p_j(s_r) - p_j(s_{sigma_r})) satisfies
      T <= (1/2) R_i R_j      (so E[X|.] <= 0),
  and this bound is TIGHT (attained at n=2 comonotone), i.e. no smaller cell-uniform
  multiple alpha of R_i R_j is valid.

We compute T EXACTLY from the true p_i, p_j of each cell (no Monte-Carlo sampling of W).
"""

import math
import random


# ---------------------------------------------------------------------------
# Core exact computations (derived directly from the definitions above)
# ---------------------------------------------------------------------------

def cyclic_next(seq):
    """sigma: r -> r+1 mod n.  Return [seq[1], seq[2], ..., seq[0]]."""
    n = len(seq)
    return [seq[(r + 1) % n] for r in range(n)]


def drift_term(p_i, p_j, c):
    """
    T = (1/n) sum_r (p_i[r] - c) * (p_j[r] - p_j[sigma_r]).
    This is the EXACT conditional-mean drift (first term of E[X|F_{m-1}]).
    """
    n = len(p_i)
    pj_next = cyclic_next(p_j)
    s = 0.0
    for r in range(n):
        s += (p_i[r] - c) * (p_j[r] - pj_next[r])
    return s / n


def ranges(p_i, p_j):
    R_i = max(p_i) - min(p_i)
    R_j = max(p_j) - min(p_j)
    return R_i, R_j


def mid_i(p_i):
    return 0.5 * (max(p_i) + min(p_i))


def cond_mean_X(p_i, p_j, delta, c):
    """E[X | F_{m-1}] = T - delta."""
    return drift_term(p_i, p_j, c) - delta


# ---- leave-one-out (LOO) variant ----
def loo_next(p_j):
    """
    s_{sigma_r} replaced by leave-one-out mean of the OTHER items' p_j:
        q_r = (sum_k p_j[k] - p_j[r]) / (n-1).
    """
    n = len(p_j)
    total = sum(p_j)
    return [(total - p_j[r]) / (n - 1) for r in range(n)]


def drift_term_loo(p_i, p_j, c):
    """T_LOO = (1/n) sum_r (p_i[r] - c)(p_j[r] - q_r) with q_r the LOO mean."""
    n = len(p_i)
    q = loo_next(p_j)
    s = 0.0
    for r in range(n):
        s += (p_i[r] - c) * (p_j[r] - q[r])
    return s / n


# ---------------------------------------------------------------------------
# Random cell generator (independent RNG; distinct seed base)
# ---------------------------------------------------------------------------

def random_cell(rng, n):
    p_i = [rng.uniform(1e-4, 1 - 1e-4) for _ in range(n)]
    p_j = [rng.uniform(1e-4, 1 - 1e-4) for _ in range(n)]
    return p_i, p_j


# ---------------------------------------------------------------------------
# Census battery
# ---------------------------------------------------------------------------

def run_census(n_per_n=200000, seed_base=90000000):
    ns = [2, 3, 4, 5, 6]

    # (a) validity: max_cell [ T - (1/2) R_i R_j ]  under delta2
    worst_valid = -1e18
    worst_valid_cell = None

    # (c) lesion: max_cell E[X|.] under half-slack delta2/2
    worst_lesion = -1e18
    worst_lesion_cell = None

    # (d) tightness: max_cell T / (R_i R_j)
    tightest_alpha = -1e18
    tightest_cell = None

    # (e) LOO: max_cell E[X|.] under delta_LOO = (n/(n-1)) R_i R_j / 4
    worst_loo = -1e18
    worst_loo_cell = None

    for n in ns:
        rng = random.Random(seed_base + n)
        for _ in range(n_per_n):
            p_i, p_j = random_cell(rng, n)
            R_i, R_j = ranges(p_i, p_j)
            RR = R_i * R_j
            c = mid_i(p_i)
            T = drift_term(p_i, p_j, c)

            delta2 = 0.5 * RR
            # (a)
            v = T - delta2
            if v > worst_valid:
                worst_valid = v
                worst_valid_cell = (n, list(p_i), list(p_j), T, RR)

            # (c) lesion
            exc = cond_mean_X(p_i, p_j, 0.5 * delta2, c)
            if exc > worst_lesion:
                worst_lesion = exc
                worst_lesion_cell = (n, list(p_i), list(p_j), T, RR)

            # (d) tightness ratio (guard against tiny RR)
            if RR > 1e-9:
                a = T / RR
                if a > tightest_alpha:
                    tightest_alpha = a
                    tightest_cell = (n, list(p_i), list(p_j), T, RR)

            # (e) LOO
            if n >= 2:
                T_loo = drift_term_loo(p_i, p_j, c)
                delta_loo = (n / (n - 1)) * RR / 4.0
                exc_loo = T_loo - delta_loo
                if exc_loo > worst_loo:
                    worst_loo = exc_loo
                    worst_loo_cell = (n, list(p_i), list(p_j), T_loo, RR)

    return dict(
        worst_valid=worst_valid, worst_valid_cell=worst_valid_cell,
        worst_lesion=worst_lesion, worst_lesion_cell=worst_lesion_cell,
        tightest_alpha=tightest_alpha, tightest_cell=tightest_cell,
        worst_loo=worst_loo, worst_loo_cell=worst_loo_cell,
    )


# ---------------------------------------------------------------------------
# (b) c-invariance check
# ---------------------------------------------------------------------------

def check_c_invariance(seed=90001234, trials=2000):
    rng = random.Random(seed)
    max_dev = 0.0
    for _ in range(trials):
        n = rng.choice([2, 3, 4, 5, 6])
        p_i, p_j = random_cell(rng, n)
        cs = [0.0, 1.0, mid_i(p_i), rng.uniform(-5, 5)]
        vals = [drift_term(p_i, p_j, c) for c in cs]
        dev = max(vals) - min(vals)
        max_dev = max(max_dev, dev)
    return max_dev


# ---------------------------------------------------------------------------
# (d) explicit extremal constructions
# ---------------------------------------------------------------------------

def construct_n2_comonotone(a=0.1, b=0.9):
    """n=2, p_i=(a,b), p_j=(a,b).  Expect T/(R_i R_j) = 1/2 exactly."""
    p_i = [a, b]
    p_j = [a, b]
    R_i, R_j = ranges(p_i, p_j)
    c = mid_i(p_i)
    T = drift_term(p_i, p_j, c)
    return T, R_i * R_j, (T / (R_i * R_j) if R_i * R_j > 0 else float('nan'))


def construct_n2_anticomonotone(a=0.1, b=0.9):
    """n=2, p_i=(a,b), p_j=(b,a) (anti-comonotone)."""
    p_i = [a, b]
    p_j = [b, a]
    R_i, R_j = ranges(p_i, p_j)
    c = mid_i(p_i)
    T = drift_term(p_i, p_j, c)
    return T, R_i * R_j, (T / (R_i * R_j) if R_i * R_j > 0 else float('nan'))


def loo_n2_reduces():
    """At n=2 the LOO 'other mean' is just the single other item => reduces to cyclic.
       delta_LOO = (2/1) R_iR_j/4 = (1/2) R_iR_j.  Verify equality of drift terms."""
    rng = random.Random(90009999)
    max_diff = 0.0
    for _ in range(5000):
        p_i, p_j = random_cell(rng, 2)
        c = mid_i(p_i)
        d = abs(drift_term(p_i, p_j, c) - drift_term_loo(p_i, p_j, c))
        max_diff = max(max_diff, d)
    return max_diff


# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("INDEPENDENT CENSUS — Proposition 1 (fresh-pair cyclic drift bound)")
    print("seed base 90000000+n ; pure-python RNG ; exact (no Monte-Carlo)")
    print("=" * 70)

    res = run_census()

    print("\n(a) VALIDITY under delta2 = (1/2) R_i R_j")
    print(f"    max_cell [ T - (1/2) R_i R_j ] = {res['worst_valid']:.3e}")
    print(f"    (must be <= ~1e-12 for Prop 1 to hold)")
    nc = res['worst_valid_cell']
    print(f"    worst cell: n={nc[0]} T={nc[3]:.6f} R_iR_j={nc[4]:.6f}")
    if res['worst_valid'] > 1e-9:
        print("    *** POSITIVE — PROP 1 VIOLATED ***")
        print(f"    p_i={nc[1]}\n    p_j={nc[2]}")

    print("\n(b) c-INVARIANCE (cyclic-sum identity)")
    dev = check_c_invariance()
    print(f"    max spread of T over c in {{0,1,mid,random}} = {dev:.3e}")
    print("    (must be ~1e-12 if c is truly free)")

    print("\n(c) LESION — half slack delta2/2 = (1/4) R_i R_j")
    print(f"    max_cell E[X|.] = {res['worst_lesion']:.3e}")
    lc = res['worst_lesion_cell']
    print(f"    worst cell: n={lc[0]}")
    print("    (should be POSITIVE => census has power to catch invalid slack)")

    print("\n(d) TIGHTNESS — empirical tightest alpha = max_cell T/(R_i R_j)")
    print(f"    alpha_emp = {res['tightest_alpha']:.6f}   (target 0.5)")
    tc = res['tightest_cell']
    print(f"    tightest cell: n={tc[0]}")
    T2, RR2, ratio2 = construct_n2_comonotone()
    print(f"    n=2 comonotone p_i=p_j=(0.1,0.9): T={T2:.6f} RR={RR2:.6f} "
          f"ratio={ratio2:.12f}")
    Ta, RRa, ratioa = construct_n2_anticomonotone()
    print(f"    n=2 ANTIcomonotone p_i=(0.1,0.9) p_j=(0.9,0.1): T={Ta:.6f} "
          f"ratio={ratioa:.12f}")

    print("\n(e) LOO variant — delta_LOO = (n/(n-1)) R_i R_j / 4")
    print(f"    max_cell E[X|.]_LOO = {res['worst_loo']:.3e}")
    ec = res['worst_loo_cell']
    print(f"    worst cell: n={ec[0]}  (must be <= ~0 if LOO slack valid)")
    d2 = loo_n2_reduces()
    print(f"    n=2 LOO vs cyclic drift max |diff| = {d2:.3e} "
          f"(LOO reduces to cyclic at n=2; delta_LOO=(1/2)R_iR_j there)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
