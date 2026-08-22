"""
eprocess.py — Three e-process methods for testing co-failure between a judge pair.

All three test whether judges i and j co-fail MORE than their shared per-item
difficulty explains. They differ in how they form the baseline term V_s.

Key idea:
  U_s = W_i^s * W_j^s          (same-item co-failure)
  V_s = W_i^s * W_j^{t(s)}     (cross-item product: i on item s, j on a DIFFERENT item t)

Because s and t are different items with independent difficulties:
  E[V_s] = E[W_i^s] * E[W_j^t] = a * b   (exactly, regardless of item difficulty distribution)

E-value increment (affine bet): e_s = 1 + lambda * (U_s - V_s - delta_k)
Wealth: M_t = prod_{s<=t} e_s,  M_0 = 1
Reject H0 when M_t >= threshold (= 1/alpha = 20 at alpha=0.05).
"""

import numpy as np


# ── Stratification helpers ─────────────────────────────────────────────────────

Q = 5   # number of strata

def assign_strata(theta: np.ndarray, Q: int = Q) -> np.ndarray:
    """
    Assign each item to a stratum by equal-width binning of theta in [0,1].
    theta ~ Uniform(0,1) so bins [0,0.2), [0.2,0.4), ... are equal-probability.
    Returns integer array of stratum indices in {0,...,Q-1}.
    """
    bins = np.linspace(0.0, 1.0, Q + 1)
    # clip to handle theta=1.0 exactly
    strata = np.searchsorted(bins[1:], theta, side='left')
    return np.clip(strata, 0, Q - 1)


def stratum_epsilon(p_i: np.ndarray, p_j: np.ndarray, strata: np.ndarray, Q: int = Q) -> np.ndarray:
    """
    Compute epsilon per stratum = max over the two judges of
      (max_s p(theta^s) - min_s p(theta^s)) / 2  for items s in that stratum.

    Returns array of length Q: epsilon_k for k = 0..Q-1.
    Uses TRUE p(theta) — in deployment this would be estimated.
    """
    eps = np.zeros(Q)
    for k in range(Q):
        mask = strata == k
        if mask.sum() == 0:
            eps[k] = 0.0
            continue
        range_i = (p_i[mask].max() - p_i[mask].min()) / 2.0
        range_j = (p_j[mask].max() - p_j[mask].min()) / 2.0
        eps[k] = max(range_i, range_j)
    return eps


def make_derangement_within_strata(strata: np.ndarray, rng: np.random.Generator, Q: int = Q) -> np.ndarray:
    """
    Build t(s): a function from item index to a DIFFERENT item in the SAME stratum.
    Within each stratum, produce a permutation with no fixed points (derangement).
    For strata of size 1 (degenerate), map to itself — delta_k will absorb any bias.
    Returns array t of length N.
    """
    N = len(strata)
    t = np.arange(N, dtype=int)
    for k in range(Q):
        idx = np.where(strata == k)[0]
        if len(idx) <= 1:
            continue   # can't derange a singleton; t[idx] stays as-is
        # shuffle until derangement (expected 2 iterations)
        perm = idx.copy()
        while True:
            rng.shuffle(perm)
            if not np.any(perm == idx):
                break
        t[idx] = perm
    return t


def make_derangement_global(N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build t(s): a derangement over the WHOLE stream (Method B ablation).
    Returns array t of length N with t[s] != s for all s.
    """
    t = np.arange(N, dtype=int)
    while True:
        rng.shuffle(t)
        if not np.any(t == np.arange(N)):
            break
    return t


# ── Core wealth accumulation ───────────────────────────────────────────────────

LAMBDA_GRID = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def accumulate_wealth(
    U: np.ndarray,
    V: np.ndarray,
    delta: np.ndarray,
    lam: float,
    clip_lam: bool = False,
    eps_per_item: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the wealth process M_t = prod_{s<=t} (1 + lam*(U_s - V_s - delta_s)).

    Args:
        U:            same-item co-failure indicator (N,)
        V:            cross-item product (N,)
        delta:        per-item conservative slack delta_k = 2*epsilon_k (N,)
        lam:          fixed betting fraction
        clip_lam:     if True, clip lam per item to [0, 1/(1+2*eps)] to keep e_s >= 0
        eps_per_item: required when clip_lam=True; per-item epsilon

    Returns:
        M: wealth array of shape (N,), M[t] = product over s=0..t
    """
    if clip_lam and eps_per_item is not None:
        lam_max = 1.0 / (1.0 + 2.0 * eps_per_item)
        lam_eff = np.minimum(lam, lam_max)
    else:
        lam_eff = lam

    increments = 1.0 + lam_eff * (U - V - delta)
    # Safety assertion: increments must be non-negative for the e-process to be a
    # valid supermartingale (E[increment] <= 1 requires increment >= 0 almost surely
    # for binary {0,1} outcomes with lam clipped to [0, 1/(1+2*eps)]).
    # We assert loudly here so that any future extension to non-binary outcomes or
    # misconfigured lambda fails immediately rather than being silently masked.
    # The np.maximum below only handles tiny float-rounding negatives (~1e-12 scale),
    # NOT real negative mass — if this assert fires, fix the DGP or lambda bound.
    assert np.all(increments >= -1e-12), (
        "negative increment: martingale property broken — "
        "the np.maximum clamp masks this for non-binary outcomes; "
        "do not clamp, fix the DGP/lambda bound"
    )
    increments = np.maximum(increments, 0.0)   # zero out float-rounding dust only
    M = np.cumprod(increments)
    return M


def accumulate_wealth_mixture(
    U: np.ndarray,
    V: np.ndarray,
    delta: np.ndarray,
    clip_lam: bool = False,
    eps_per_item: np.ndarray | None = None,
) -> np.ndarray:
    """
    Robbins-style mixture: M_t^mix = mean over lambda grid of prod_{s<=t}(1+lam*(U-V-delta)).
    Returns M of shape (N,).
    """
    wealths = []
    for lam in LAMBDA_GRID:
        W = accumulate_wealth(U, V, delta, lam, clip_lam=clip_lam, eps_per_item=eps_per_item)
        wealths.append(W)
    return np.mean(np.stack(wealths, axis=0), axis=0)


# ── The three methods ──────────────────────────────────────────────────────────

def method_A(data: dict, rng: np.random.Generator) -> dict:
    """
    Method A — STRATIFIED cross-item e-process (the proposed method).
    t(s) is a derangement WITHIN the same stratum.
    delta_k = 2 * epsilon_k (stratum-specific conservative slack).
    lambda clipped to [0, 1/(1+2*eps)] per item.
    Returns dict with 'M_fixed', 'M_mix', 'delta_per_item', 'eps_per_item'.
    """
    W_i, W_j = data['W_i'], data['W_j']
    p_i, p_j = data['p_i'], data['p_j']
    theta = data['theta']
    N = len(W_i)

    strata = assign_strata(theta)
    eps_k = stratum_epsilon(p_i, p_j, strata)   # shape (Q,)
    t = make_derangement_within_strata(strata, rng)

    U = W_i * W_j
    V = W_i[np.arange(N)] * W_j[t]              # i on item s, j on different item t(s)

    eps_per_item = eps_k[strata]                 # shape (N,)
    delta = 2.0 * eps_per_item

    M_fixed = accumulate_wealth(U, V, delta, lam=0.5, clip_lam=True, eps_per_item=eps_per_item)
    M_mix   = accumulate_wealth_mixture(U, V, delta, clip_lam=True, eps_per_item=eps_per_item)

    return dict(M_fixed=M_fixed, M_mix=M_mix, delta_per_item=delta, eps_per_item=eps_per_item)


def method_B(data: dict, rng: np.random.Generator) -> dict:
    """
    Method B — UNSTRATIFIED cross-item e-process (ablation).
    t(s) is a derangement over the WHOLE stream (ignores strata).
    delta = 0. Expected to pick up shared-difficulty covariance under H0.
    """
    W_i, W_j = data['W_i'], data['W_j']
    N = len(W_i)

    t = make_derangement_global(N, rng)
    U = W_i * W_j
    V = W_i * W_j[t]
    delta = np.zeros(N)

    M_fixed = accumulate_wealth(U, V, delta, lam=0.5)
    M_mix   = accumulate_wealth_mixture(U, V, delta)

    return dict(M_fixed=M_fixed, M_mix=M_mix)


def method_C(data: dict) -> dict:
    """
    Method C — NAIVE plug-in baseline (ablation).
    e_s = 1 + lambda*(U_s - a_hat * b_hat)  where a_hat,b_hat are global sample means.
    delta = 0. Uses the whole-stream mean — vulnerable to drift.
    NOTE: this is a fixed baseline, not a martingale under H0 with drifting marginals.
    """
    W_i, W_j = data['W_i'], data['W_j']
    N = len(W_i)

    a_hat = W_i.mean()
    b_hat = W_j.mean()
    U = W_i * W_j
    V = np.full(N, a_hat * b_hat)   # fixed constant baseline
    delta = np.zeros(N)

    M_fixed = accumulate_wealth(U, V, delta, lam=0.5)
    M_mix   = accumulate_wealth_mixture(U, V, delta)

    return dict(M_fixed=M_fixed, M_mix=M_mix, a_hat=a_hat, b_hat=b_hat)


# ── Ville stopping criterion ───────────────────────────────────────────────────

THRESHOLD = 20.0   # = 1/alpha at alpha = 0.05


def ville_reject(M: np.ndarray, threshold: float = THRESHOLD) -> tuple[bool, int | None]:
    """
    Return (rejected, first_t) where first_t is the first index (0-based) with M[t] >= threshold.
    If never rejected, returns (False, None).
    """
    crossing = np.where(M >= threshold)[0]
    if len(crossing) == 0:
        return False, None
    return True, int(crossing[0])
