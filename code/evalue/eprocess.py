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

Q = 5      # number of strata
BLOCK = 20 # items per simultaneous-batch block (matches the §5b harness); used by the
           # fresh-pair within-block pairing that provides anytime validity (Clio Prop. 3).

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


def make_within_block_pairing(strata: np.ndarray, block_ids: np.ndarray,
                              Q: int = Q) -> np.ndarray:
    """
    Build t(s): a FRESH-PAIR, WITHIN-BLOCK pairing (Clio's Prop. 3 repair).

    For each block m and each stratum k, cyclically pair the same-stratum items
    that fall inside that block:  the j-th such item is paired with the (j+1)-th
    (mod count) same-stratum item of the SAME block.  A (block, stratum) cell with
    fewer than 2 items places NO bet (sentinel t = self), exactly as the past-only
    code does.

    WHY THIS IS THE FIX (vs make_online_past_pairing, which is WRONG — see Prop. 2):
      The past-only map sets t(s) to an EARLIER same-stratum item, so W_j[t(s)] is
      already F_{s-1}-measurable.  Then on the OBSERVABLE event {W_j[t(s)]=0} the
      increment U - V - delta has conditional mean  p_i*p_j - delta > 0, and a
      predictable bettor that bets only on that event achieves Type-I -> 1.
      Here t(s) is another item in the SAME block, so when the bet lambda for block m
      is set (it is F_{m-1}-measurable, depending only on strata/block structure),
      NEITHER W_j[s] NOR W_j[t(s)] has been revealed.  Hence
        E[W_i^s (W_j^s - W_j^{t(s)}) | F_{m-1}, theta] = p_i(s)(p_j(s) - p_j(t(s)))
        <= p_i * 2 eps_k <= delta_k,
      so E[X | F_{m-1}] <= 0 for EVERY predictable lambda (ONS included), under
      arbitrary cross-stratum dependence.

    The map depends ONLY on strata and block structure, never on outcomes, so it is
    predictable.  Returns array t of length N (global indices); t[s] == s is the
    no-bet sentinel for cells of size < 2.
    """
    N = len(strata)
    t = np.arange(N, dtype=int)   # default: sentinel (self-pair = no bet)
    blocks = np.unique(block_ids)
    for m in blocks:
        blk = np.where(block_ids == m)[0]          # global indices in this block
        s_blk = strata[blk]
        for k in range(Q):
            cell = blk[s_blk == k]                 # same-stratum items in this block
            if len(cell) < 2:
                continue                           # singleton cell: no bet
            # cyclic shift by one within the cell: item j -> item (j+1) mod len
            t[cell] = np.roll(cell, -1)
    return t


def make_theta_neighbour_block_pairing(theta: np.ndarray, p_i: np.ndarray, p_j: np.ndarray,
                                       block_ids: np.ndarray, Q: int = Q
                                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    THETA-NEIGHBOUR within-block pairing (Clio's starvation repair).

    The (block, stratum) fresh-pair map (make_within_block_pairing) STARVES at large Q:
    with BLOCK fixed and Q large, each (block, stratum) cell has BLOCK/Q items, so cells
    become singletons that place no bet and power collapses.  This map instead forms a
    Q-INDEPENDENT set of pairs:

      1. Sort the block's items by theta (F_0-measurable: theta is a difficulty covariate
         known at time 0, NOT an outcome).
      2. Pair ADJACENT items in that sorted order: (1,2), (3,4), ...  For BLOCK=20 this
         yields exactly 10 pairs, REGARDLESS of Q.  (An odd leftover item is a no-bet
         sentinel.)
      3. Each pair is an n=2 micro-cell.  Its slack uses the Prop-1 pair bound: the pair's
         OWN half-range eps_pair = max over the two judges of (max-min)/2 of that judge's
         two within-pair marginals, giving delta_pair = 2*eps_pair**2 = 1/2 R^2 with
         R = 2 eps_pair (census_prop1_derivation.md Sec. 3).
      4. The pair's BET is ASSIGNED to the stratum of the pair's MIDPOINT theta (which of
         the Q equal-width bins (theta_s + theta_t)/2 falls in).  This is F_0-measurable,
         depends only on theta, never on any observed W.

    WHY VALIDITY HOLDS.  Both members of every pair are drawn from the SAME block m, so
    when the block-m bet is set (F_{m-1}-measurable: the map depends only on theta and
    block structure) NEITHER W_j[s] NOR W_j[t(s)] has been revealed — the fresh-pair
    property of make_within_block_pairing is preserved.  Adjacency in theta makes the two
    partners as close in difficulty as possible, so the within-pair marginal gap (hence
    eps_pair, hence the bias E[W_i^s(W_j^s - W_j^{t(s)})]) is SMALL, and
      E[U - V | F_{m-1}, theta] = p_i(s)(p_j(s) - p_j(t(s))) <= p_i * 2 eps_pair <= delta_pair,
    so E[increment | F_{m-1}] <= 0 for every predictable lambda.

    Returns (t, delta_per_item, eps_per_item, stratum_per_item):
      t[s]               global index of s's theta-neighbour partner (t[s]==s = no-bet sentinel)
      delta_per_item[s]  pair slack delta_pair = 2*eps_pair**2 (0 for sentinels)
      eps_per_item[s]    pair half-range eps_pair (0 for sentinels)
      stratum_per_item[s] midpoint-theta stratum of the pair (for diagnostics; -1 for sentinels)
    """
    N = len(theta)
    t = np.arange(N, dtype=int)                 # default: self-pair = no bet
    delta_per_item = np.zeros(N)
    eps_per_item = np.zeros(N)
    stratum_per_item = np.full(N, -1, dtype=int)

    bins = np.linspace(0.0, 1.0, Q + 1)
    for m in np.unique(block_ids):
        blk = np.where(block_ids == m)[0]       # global indices in this block
        order = blk[np.argsort(theta[blk], kind='stable')]  # sorted by theta
        n_pairs = len(order) // 2
        for p in range(n_pairs):
            a = order[2 * p]
            b = order[2 * p + 1]
            t[a] = b
            t[b] = a
            # pair's own half-range per judge; eps_pair = max over judges
            eps_i = abs(p_i[a] - p_i[b]) / 2.0
            eps_j = abs(p_j[a] - p_j[b]) / 2.0
            eps_pair = max(eps_i, eps_j)
            delta_pair = 2.0 * eps_pair ** 2
            eps_per_item[a] = eps_per_item[b] = eps_pair
            delta_per_item[a] = delta_per_item[b] = delta_pair
            # stratum of the pair's MIDPOINT theta (F_0-measurable)
            mid = 0.5 * (theta[a] + theta[b])
            k = int(np.searchsorted(bins[1:], mid, side='left'))
            k = min(max(k, 0), Q - 1)
            stratum_per_item[a] = stratum_per_item[b] = k
    return t, delta_per_item, eps_per_item, stratum_per_item


def make_online_past_pairing(strata: np.ndarray, rng: np.random.Generator, Q: int = Q) -> np.ndarray:
    """
    DEPRECATED / BUGGY (Clio's Prop. 2): this PAST-ONLY pairing BREAKS anytime validity.

    The partner t(s) is an earlier same-stratum item, so its outcome W_j[t(s)] is
    already F_{s-1}-measurable.  On the OBSERVABLE event {W_j[t(s)] = 0} the increment
    U - V - delta has positive conditional mean (p_i*p_j - delta > 0), so a predictable
    bettor can put the whole cap on strata whose partners all failed-to-co-fail and
    drive Type-I to 1.000 under the STATIC null.  Kept only as the documented-wrong
    baseline; do NOT use for a validity claim.  Use make_within_block_pairing (Prop. 3).

    Build t(s): a PAST-ONLY (predictable, online) pairing for anytime validity.

    For each item s in arrival order:
      - If at least one prior item (index < s) exists in the same stratum k:
          draw t(s) uniformly at random from those prior same-stratum items.
          This is F_{s-1}-measurable: t(s) is chosen before item s is observed.
      - If no prior same-stratum item exists (s is the first item in its stratum):
          set t(s) = s  (sentinel: no bet — caller must skip betting for these items).

    Returns array t of length N (global indices), with:
      t[s] < s  whenever a valid prior partner exists in the same stratum, and
      t[s] == s (sentinel) for the first item in each stratum.

    Callers that need a "has_partner" mask can compute: has_partner = (t != np.arange(N)).
    This works globally (N = len(strata)) but NOT on stratum sub-slices; use
    online_has_partner() for per-stratum sentinel detection.

    This guarantees F_{n-1}-measurability of t(n): every pairing partner is strictly a
    past observation, so the per-step factor e_n is adapted to the filtration and the
    supermartingale property holds without look-ahead.
    """
    N = len(strata)
    t = np.arange(N, dtype=int)   # default: sentinel (self-pair = no bet)

    # Vectorised per-stratum: within each stratum, items arrive in index order.
    # For the j-th item in stratum k (j=0 is sentinel), draw r ~ Uniform{0,...,j-1}
    # and set t to the global index of the r-th prior item in that stratum.
    for k in range(Q):
        idx = np.where(strata == k)[0]   # global indices, sorted ascending
        m = len(idx)
        if m <= 1:
            continue   # only sentinel; t[idx] stays as-is
        # Batch: for j=1..m-1 draw r[j] ~ Uniform{0,...,j-1}
        # rng.integers(j) gives uniform in {0,...,j-1} but doesn't vectorise across
        # different upper bounds; use the Fisher-Yates partial shuffle trick instead:
        # generate m-1 uniform floats and multiply by j to get the correct index.
        # This is equivalent to independently drawing r[j] ~ Uniform{0,...,j-1}.
        uniforms = rng.random(m - 1)   # shape (m-1,), each in [0,1)
        r = (uniforms * np.arange(1, m)).astype(int)   # r[j-1] in {0,...,j-1}
        t[idx[1:]] = idx[r]

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
        delta:        per-item conservative slack delta_k = 2*epsilon_k**2 (N,)
                      (QUADRATIC slack 1/2 R^2 = 2 eps^2, R = 2 eps; see
                      census_prop1_derivation.md Sec. 3.  The old LINEAR 2*eps was
                      valid but loose by a factor 1/eps — Clio's Prop. 2 review.)
        lam:          fixed betting fraction
        clip_lam:     if True, clip lam per item to [0, 1/(1+delta)] to keep e_s >= 0.
                      The positivity bound is lam <= 1/(1+delta) with delta = 2*eps**2
                      (increment U-V-delta in [-(1+delta), 1-delta]); this is the tight
                      per-item cap consistent with the quadratic slack.  With the
                      universal design bound eps=1/2 (delta=1/2) it reduces to the
                      b=3/2 => lam_max=2/3 universal bound of Clio's note.
        eps_per_item: required when clip_lam=True; per-item epsilon

    Returns:
        M: wealth array of shape (N,), M[t] = product over s=0..t
    """
    if clip_lam and eps_per_item is not None:
        # Positivity cap consistent with the QUADRATIC slack delta = 2*eps**2:
        # increment U - V - delta lies in [-(1+delta), 1-delta], so the factor
        # 1 + lam*(U-V-delta) >= 0 requires lam <= 1/(1+delta) = 1/(1+2*eps**2).
        # (The old 1/(1+2*eps) was tied to the linear slack; it is still valid here
        # since 2*eps >= 2*eps**2, but 1/(1+2*eps**2) is the tight per-item cap.)
        lam_max = 1.0 / (1.0 + 2.0 * eps_per_item ** 2)
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


# ── Online (past-only) wealth accumulation ────────────────────────────────────

def accumulate_wealth_online(
    U: np.ndarray,
    V: np.ndarray,
    delta: np.ndarray,
    has_partner: np.ndarray,
    lam: float,
    clip_lam: bool = False,
    eps_per_item: np.ndarray | None = None,
) -> np.ndarray:
    """
    Wealth accumulation for the online past-only pairing.

    Items where has_partner[s] is False (no prior same-stratum partner — sentinels)
    contribute a factor of 1.0 to the product: they make no bet.
    All other items contribute 1 + lam*(U_s - V_s - delta_s) as usual.

    Args:
        U:           same-item co-failure indicator (N,)
        V:           cross-item baseline (N,); V[s] = W_j[t[s]] (sentinel positions ignored)
        delta:       per-item slack (N,)
        has_partner: boolean mask (N,); True where item s has a valid F_{s-1}-measurable partner
        lam:         betting fraction
        clip_lam:    if True, clip lam per item to [0, 1/(1+2*eps)]
        eps_per_item: required when clip_lam=True

    Returns:
        M: wealth process of shape (N,), M[n] = product over s=0..n
    """
    N = len(U)

    if clip_lam and eps_per_item is not None:
        # Positivity cap for the QUADRATIC slack delta = 2*eps**2 (see accumulate_wealth).
        lam_max = 1.0 / (1.0 + 2.0 * eps_per_item ** 2)
        lam_eff = np.minimum(lam, lam_max)
    else:
        lam_eff = np.full(N, lam) if np.isscalar(lam) else np.array(lam, dtype=float)

    increments = np.ones(N, dtype=float)
    if has_partner.any():
        increments[has_partner] = (
            1.0 + lam_eff[has_partner] * (U[has_partner] - V[has_partner] - delta[has_partner])
        )

    assert np.all(increments >= -1e-12), (
        "negative increment in online past-pairing: check lambda bound and DGP"
    )
    increments = np.maximum(increments, 0.0)
    M = np.cumprod(increments)
    return M


def accumulate_wealth_online_mixture(
    U: np.ndarray,
    V: np.ndarray,
    delta: np.ndarray,
    has_partner: np.ndarray,
    clip_lam: bool = False,
    eps_per_item: np.ndarray | None = None,
) -> np.ndarray:
    """
    Robbins-style mixture for the online past-only pairing.
    Averages over LAMBDA_GRID; sentinel items (has_partner=False) contribute factor 1.
    """
    wealths = []
    for lam in LAMBDA_GRID:
        W = accumulate_wealth_online(U, V, delta, has_partner, lam,
                                     clip_lam=clip_lam, eps_per_item=eps_per_item)
        wealths.append(W)
    return np.mean(np.stack(wealths, axis=0), axis=0)


# ── The three methods ──────────────────────────────────────────────────────────

def method_A(data: dict, rng: np.random.Generator) -> dict:
    """
    Method A — STRATIFIED cross-item e-process (the proposed method).
    t(s) is a FRESH-PAIR within-block, within-stratum partner (Clio Prop. 3):
    neither partner's outcome is in F_{m-1} when the block-m bet is set, so the
    e-process is anytime-valid (unlike the old whole-stream derangement / past-only
    pairing).  rng is accepted for signature compatibility but unused: the fresh-pair
    map is a deterministic cyclic roll within each (block, stratum) cell.
    delta_k = 2 * epsilon_k**2 (QUADRATIC slack 1/2 R^2 = 2 eps^2, R = 2 eps;
    census_prop1_derivation.md Sec. 3).  lambda clipped to [0, 1/(1+2*eps**2)] per item.
    Returns dict with 'M_fixed', 'M_mix', 'delta_per_item', 'eps_per_item'.
    """
    W_i, W_j = data['W_i'], data['W_j']
    p_i, p_j = data['p_i'], data['p_j']
    theta = data['theta']
    N = len(W_i)

    strata = assign_strata(theta)
    eps_k = stratum_epsilon(p_i, p_j, strata)   # shape (Q,)
    block_ids = np.arange(N) // BLOCK
    t = make_within_block_pairing(strata, block_ids)   # fresh disjoint-item pairs

    U = W_i * W_j
    V = W_i[np.arange(N)] * W_j[t]              # i on item s, j on different item t(s)

    eps_per_item = eps_k[strata]                 # shape (N,)
    delta = 2.0 * eps_per_item ** 2             # QUADRATIC slack (1/2 R^2 = 2 eps^2)

    M_fixed = accumulate_wealth(U, V, delta, lam=0.5, clip_lam=True, eps_per_item=eps_per_item)
    M_mix   = accumulate_wealth_mixture(U, V, delta, clip_lam=True, eps_per_item=eps_per_item)

    return dict(M_fixed=M_fixed, M_mix=M_mix, delta_per_item=delta, eps_per_item=eps_per_item)


def method_A_online(data: dict, rng: np.random.Generator) -> dict:
    """
    Method A (online/anytime-valid) — STRATIFIED cross-item e-process with predictable
    FRESH-PAIR pairing.

    Uses make_within_block_pairing (Clio Prop. 3): t(s) is another SAME-stratum item in
    the SAME block, so when the block-m bet is set (F_{m-1}-measurable) neither W_j[s] nor
    W_j[t(s)] has been revealed.  This is the correct anytime/sequential-validity
    construction.  (The previous make_online_past_pairing is DEPRECATED/BUGGY — its
    partner outcome is already F_{s-1}-measurable, letting a predictable bettor drive
    Type-I to 1.000 on the observable {W_j[t(s)]=0} event; see its docstring.)
    rng is accepted for signature compatibility but unused (deterministic cyclic roll).
    Cells of size < 2 make no bet (factor = 1), via the has_partner sentinel.

    delta_k = 2 * epsilon_k**2 (quadratic slack); lambda clipped to [0, 1/(1+2*eps**2)].
    Returns dict with 'M_fixed', 'M_mix', 'delta_per_item', 'eps_per_item'.
    """
    W_i, W_j = data['W_i'], data['W_j']
    p_i, p_j = data['p_i'], data['p_j']
    theta = data['theta']
    N = len(W_i)

    strata = assign_strata(theta)
    eps_k = stratum_epsilon(p_i, p_j, strata)
    block_ids = np.arange(N) // BLOCK
    t = make_within_block_pairing(strata, block_ids)   # fresh disjoint-item pairs

    # has_partner[s] is True iff t[s] != s (the cell has >= 2 items so a fresh partner
    # exists).  Sentinel: t[s] == s (singleton cell); these make no bet (factor=1).
    has_partner = (t != np.arange(N))

    U = W_i * W_j
    # V[s] = W_j[t[s]]; for sentinel positions (t[s]==s) the value W_j[s] is computed
    # but ignored because has_partner[s]=False zeros out the bet in accumulate_wealth_online.
    V = W_i * W_j[t]

    eps_per_item = eps_k[strata]
    delta = 2.0 * eps_per_item ** 2             # QUADRATIC slack (1/2 R^2 = 2 eps^2)

    M_fixed = accumulate_wealth_online(U, V, delta, has_partner, lam=0.5,
                                        clip_lam=True, eps_per_item=eps_per_item)
    M_mix   = accumulate_wealth_online_mixture(U, V, delta, has_partner,
                                               clip_lam=True, eps_per_item=eps_per_item)

    return dict(M_fixed=M_fixed, M_mix=M_mix, delta_per_item=delta, eps_per_item=eps_per_item)


def method_B(data: dict, rng: np.random.Generator) -> dict:
    """
    Method B — UNSTRATIFIED cross-item e-process (ablation).
    t(s) is a derangement over the WHOLE stream (ignores strata).
    delta = 0. Expected to pick up shared-difficulty covariance under H0.

    NOTE: this is a deliberately-INVALID ablation (no stratification, delta=0) kept to
    demonstrate that the unstratified version falsely fires on shared-difficulty
    marginal correlation.  It is NOT a validity claim, so it intentionally does NOT use
    the fresh-pair within-block construction (that would change what the ablation shows).
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
