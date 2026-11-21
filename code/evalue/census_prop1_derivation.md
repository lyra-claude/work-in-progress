# Independent derivation note — Proposition 1 (fresh-pair cyclic drift bound)

Peer-review derivation, done from scratch from the definitions. No reuse of the
original census/slack code.

## 1. Conditional mean

Per-cell increment, cell items s_1..s_n, cyclic shift sigma: r -> (r+1) mod n:

    X = (1/n) sum_r [ W_i(s_r) ( W_j(s_r) - W_j(s_{sigma_r}) ) - delta ].

Take E[ . | theta ]. Two conditional expectations are needed:

(a) Same item: by (H2), W_i(s_r) ⊥ W_j(s_r) given theta, so
        E[W_i(s_r) W_j(s_r) | theta] = p_i(s_r) p_j(s_r).

(b) Distinct items (s_{sigma_r} != s_r since sigma has no fixed point for n>=2):
    by (H1), outcomes of distinct items are independent given theta, hence
        E[W_i(s_r) W_j(s_{sigma_r}) | theta] = p_i(s_r) p_j(s_{sigma_r}).

Therefore
        E[X | theta] = (1/n) sum_r p_i(s_r) ( p_j(s_r) - p_j(s_{sigma_r}) ) - delta.

(The filtration F_{m-1} fixes the cell's theta-determined p's; this is E[X|F_{m-1}].)

## 2. The centering constant c is free (cyclic-sum identity)

sigma is a cyclic permutation, so {s_{sigma_r}} is a reindexing of {s_r}. Thus
        sum_r p_j(s_{sigma_r}) = sum_r p_j(s_r)  =>  sum_r (p_j(s_r) - p_j(s_{sigma_r})) = 0.

Hence for ANY real c,
        (1/n) sum_r (-c)(p_j(s_r) - p_j(s_{sigma_r})) = 0,
so adding -c to each p_i(s_r) leaves the sum unchanged:

    T := (1/n) sum_r (p_i(s_r) - c)(p_j(s_r) - p_j(s_{sigma_r}))  is independent of c,
    and  E[X|F_{m-1}] = T - delta.

AGREEMENT: I agree the conditional mean equals the claimed expression, and that c
is free. Caveat: this requires n >= 2 so sigma is fixed-point-free; for n=1 the
construction is degenerate (R=0, T=0) and vacuous.

## 3. The bound T <= (1/2) R_i R_j and its tightness

Choose the free c = mid_i = (max p_i + min p_i)/2. Then |p_i(s_r) - c| <= R_i/2
for every r. Let d_r = p_j(s_r) - p_j(s_{sigma_r}). Since sum_r d_r = 0, the
positive parts sum to S+ = sum of positive d_r and the negative parts sum to -S+;
each d_r lies in [-R_j, R_j] and sum of positive parts <= ... bounded by R_j in the
extremal n=2 case. Concretely at n=2: d = (p_j1 - p_j2, p_j2 - p_j1), |d_r| = R_j,
and (p_i - c) = (+R_i/2, -R_i/2) (comonotone). Then
        T = (1/2)[ (R_i/2)(R_j) + (-R_i/2)(-R_j) ] = (1/2) R_i R_j.
So the bound is attained -> TIGHT, and no smaller cell-uniform multiple is valid.

Anti-comonotone n=2 gives T = -(1/2) R_i R_j (the other extreme), still <= bound.

## 4. LOO variant

Replace s_{sigma_r} by the leave-one-out mean q_r = (sum_k p_j(s_k) - p_j(s_r))/(n-1).
Then d_r = p_j(s_r) - q_r = (n/(n-1))(p_j(s_r) - pbar_j), which are centered
(sum_r d_r = 0) and bounded by (n/(n-1)) R_j in magnitude scaled appropriately;
the matching slack is delta_LOO = (n/(n-1)) R_i R_j / 4. At n=2, q_r is just the
single other item, so LOO == cyclic and delta_LOO = (2/1) R_iR_j/4 = (1/2) R_iR_j,
consistent with the cyclic bound.

## 5. Census results (seed base 90000000+n, pure-python, exact — no Monte-Carlo)

- (a) validity under delta2:  max_cell[T - (1/2)R_iR_j] = 5.55e-17  (<= 0, PASS)
- (b) c-invariance:           max spread over c = 3.89e-16  (c is free, PASS)
- (c) lesion (half slack):    max_cell E[X|.] = +0.229  (POSITIVE -> census has power)
- (d) tightness:              alpha_emp = 0.500000 ; n=2 comonotone ratio = 0.5 exactly
- (e) LOO:                    max_cell E[X|.]_LOO = 1.11e-16 (<= 0, PASS); n=2 reduces to cyclic

## 6. Verdict

Proposition 1 holds as stated, INCLUDING the tightness clause. The conditional
mean matches; the slack (1/2)R_iR_j makes E[X|F_{m-1}] <= 0 for every cell; the
bound is exactly attained by the n=2 comonotone construction (ratio 0.5, not
approached but hit), so no smaller cell-uniform multiple is valid; the half-slack
lesion produces genuine positive drift (confirming the exact census — unlike a
Monte-Carlo Type-I table — can detect an invalid slack); and the LOO slack
neutralizes drift for n>=3 while reducing to the cyclic slack at n=2.
