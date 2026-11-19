"""
dgp.py — Data-generating processes for the co-failure e-process simulation.

Two LLM judges i and j score N items. Each item has a latent difficulty theta ~ U(0,1).
Judge error probabilities are difficulty-driven and shared across judges, creating
marginal correlation via shared difficulty but (under H0) NO excess co-failure.
"""

import numpy as np


def judge_probs(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (p_i, p_j) — per-item marginal error probs for judges i and j.
    Both rise with difficulty theta, creating shared-difficulty marginal correlation.
    """
    p_i = np.clip(0.30 + 0.50 * (theta - 0.5), 0.02, 0.98)
    p_j = np.clip(0.35 + 0.50 * (theta - 0.5), 0.02, 0.98)
    return p_i, p_j


def null_dgp(N: int, rng: np.random.Generator) -> dict:
    """
    H0 — conditional independence given difficulty.
    Judges share difficulty but fail independently given theta.
    Returns dict with keys: theta, W_i, W_j, p_i, p_j.
    """
    theta = rng.uniform(0.0, 1.0, size=N)
    p_i, p_j = judge_probs(theta)
    W_i = (rng.uniform(size=N) < p_i).astype(int)
    W_j = (rng.uniform(size=N) < p_j).astype(int)
    return dict(theta=theta, W_i=W_i, W_j=W_j, p_i=p_i, p_j=p_j)


def alt_dgp(N: int, gamma: float, rng: np.random.Generator) -> dict:
    """
    H1 — genuine excess co-failure via a common shock (Marshall-Olkin atom).
    With probability gamma each item is a "shock": both judges fail together
    regardless of difficulty. Otherwise generated as H0.
    gamma in {0.05, 0.10} per the spec.
    Returns dict with keys: theta, W_i, W_j, p_i, p_j, shock (bool mask).
    """
    theta = rng.uniform(0.0, 1.0, size=N)
    p_i, p_j = judge_probs(theta)

    shock = rng.uniform(size=N) < gamma
    W_i = np.where(shock, 1, (rng.uniform(size=N) < p_i).astype(int))
    W_j = np.where(shock, 1, (rng.uniform(size=N) < p_j).astype(int))
    return dict(theta=theta, W_i=W_i, W_j=W_j, p_i=p_i, p_j=p_j, shock=shock)


def _equicorr_stratum_shocks(gamma: float, Q: int = 10):
    """
    Derive per-stratum shock probabilities gamma_s that make the per-item EXCESS
    co-failure covariance  gamma_s*(1-gamma_s)*(1-p_i_s)*(1-p_j_s)  CONSTANT across strata.

    Derivation.  Under a Marshall-Olkin shock of probability g in a stratum with mean
    marginals (a, b), the covariance of the two failure indicators is
        Cov = [g + (1-g) a b] - [g+(1-g)a][g+(1-g)b]
            = g (1-g) (1-a)(1-b).
    (Algebra: expand and collect; the cross terms cancel to leave the product form.)
    So the per-stratum "signal" the stratified combiner sees is  g_s (1-g_s) r_s  with
        r_s = (1 - a_s)(1 - b_s).
    To make this CONSTANT = C across strata we solve  g_s (1-g_s) = C / r_s  for g_s,
        g_s = (1 - sqrt(1 - 4 C / r_s)) / 2      (smaller root, g_s in [0, 0.5]).
    Feasible iff C <= r_s / 4 for every s, i.e. C <= min_s r_s / 4.

    We parametrize the overall strength by `gamma` = the shock probability in the
    HARDEST stratum (smallest r_s), so  C = gamma (1-gamma) * min_s r_s.  Easier strata
    (larger 1-p => larger r_s) then get SMALLER g_s to hold the covariance fixed.

    Returns (gamma_s array of length Q, C, r_s array).  Uses the TRUE stratum-mean
    marginals from judge_probs on the bin centers.
    """
    # stratum bin centers in theta (equal-width bins on [0,1])
    centers = (np.arange(Q) + 0.5) / Q
    a, b = judge_probs(centers)              # mean marginals per stratum
    r = (1.0 - a) * (1.0 - b)                # (Q,)
    C = gamma * (1.0 - gamma) * float(r.min())
    K = C / r                                # (Q,)
    disc = 1.0 - 4.0 * K
    if np.any(disc < 0):
        raise ValueError(f"equicorr infeasible at gamma={gamma}: C={C} exceeds r_s/4 for some stratum")
    gamma_s = (1.0 - np.sqrt(disc)) / 2.0    # (Q,)
    return gamma_s, C, r


def alt_dgp_equicorr_true(N: int, gamma: float, rng: np.random.Generator, Q: int = 10) -> dict:
    """
    GENUINELY-EQUICORRELATED alternative (Task A).

    Unlike alt_dgp (uniform shock gamma => per-stratum covariance g(1-g)(1-p_i)(1-p_j)
    that varies ~4x across strata because p rises with difficulty), here the shock
    probability is STRATUM-DEPENDENT, gamma_s, scaled so that the per-item excess
    co-failure covariance  gamma_s*(1-gamma_s)*(1-p_i)*(1-p_j)  is CONSTANT across strata.
    See _equicorr_stratum_shocks for the derivation.  `gamma` sets the overall strength
    (= shock prob in the hardest stratum).  Result: the per-stratum Kelly/oracle target
    is ~uniform => this is the TRUE forecast-combination-puzzle home turf.

    Returns dict with keys: theta, W_i, W_j, p_i, p_j, shock, gamma_s.
    """
    theta = rng.uniform(0.0, 1.0, size=N)
    p_i, p_j = judge_probs(theta)
    gamma_s, _, _ = _equicorr_stratum_shocks(gamma, Q=Q)
    strata = np.clip((theta * Q).astype(int), 0, Q - 1)
    g_item = gamma_s[strata]                                  # per-item shock prob
    shock = rng.uniform(size=N) < g_item
    W_i = np.where(shock, 1, (rng.uniform(size=N) < p_i).astype(int))
    W_j = np.where(shock, 1, (rng.uniform(size=N) < p_j).astype(int))
    return dict(theta=theta, W_i=W_i, W_j=W_j, p_i=p_i, p_j=p_j, shock=shock, gamma_s=gamma_s)


def drift_null_dgp(N: int, rng: np.random.Generator) -> dict:
    """
    Drift-null variant — marginals ramp linearly over the stream:
      base_i: 0.20 -> 0.45 across N items
      base_j: 0.25 -> 0.50 across N items
    Still H0 (conditional independence given difficulty), but global sample means
    are a biased estimate of the true marginals at any given time.
    Returns dict with keys: theta, W_i, W_j, p_i, p_j.
    """
    theta = rng.uniform(0.0, 1.0, size=N)
    t_frac = np.linspace(0.0, 1.0, N)
    base_i = 0.20 + 0.25 * t_frac   # 0.20 -> 0.45
    base_j = 0.25 + 0.25 * t_frac   # 0.25 -> 0.50
    p_i = np.clip(base_i + 0.50 * (theta - 0.5), 0.02, 0.98)
    p_j = np.clip(base_j + 0.50 * (theta - 0.5), 0.02, 0.98)
    W_i = (rng.uniform(size=N) < p_i).astype(int)
    W_j = (rng.uniform(size=N) < p_j).astype(int)
    return dict(theta=theta, W_i=W_i, W_j=W_j, p_i=p_i, p_j=p_j)
