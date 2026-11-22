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
for every r. Let d_r = p_j(s_r) - p_j(s_{sigma_r}). Each d_r is a difference of two
p_j values, so |d_r| <= R_j for every r (all n). The general-n bound then follows by
a single triangle inequality, with NO appeal to the n=2 geometry:

    T = (1/n) sum_r (p_i(s_r) - c) d_r
      <= (1/n) sum_r |p_i(s_r) - c| |d_r|
      <= (1/n) sum_r (R_i/2)(R_j)            [since |p_i-c|<=R_i/2 and |d_r|<=R_j]
      =  (1/2) R_i R_j.

So T <= (1/2) R_i R_j for ALL n >= 2, not merely n=2. (Note: the per-r bound
|d_r| <= R_j is what carries the proof; the "sum of positive parts" is 2 S+ which can
exceed R_j for n > 2 -- e.g. p_j alternating 0,1,0,1 gives S+ = n/4 -- so that route is
NOT the right one. The termwise |d_r| <= R_j bound is.) Concretely at n=2 the bound is
ATTAINED: d = (p_j1 - p_j2, p_j2 - p_j1), |d_r| = R_j,
and (p_i - c) = (+R_i/2, -R_i/2) (comonotone). Then
        T = (1/2)[ (R_i/2)(R_j) + (-R_i/2)(-R_j) ] = (1/2) R_i R_j.
So the bound is attained -> TIGHT, and no smaller cell-uniform multiple is valid.

Anti-comonotone n=2 gives T = -(1/2) R_i R_j (the other extreme), still <= bound.

## 4. LOO variant

Replace s_{sigma_r} by the leave-one-out mean q_r = (sum_k p_j(s_k) - p_j(s_r))/(n-1).
Then d_r = p_j(s_r) - q_r = (n/(n-1))(p_j(s_r) - pbar_j), which are centered
(sum_r d_r = 0). Since |p_j(s_r) - pbar_j| <= ((n-1)/n) R_j (max deviation of a point
from the mean given range R_j), the magnitude bound is in fact |d_r| <= R_j (the factor
n/(n-1) cancels the (n-1)/n deviation bound). The matching slack delta_LOO =
(n/(n-1)) R_i R_j / 4 is derived -- and tight -- as follows. Because sum_r d_r = 0, the
c-centering of p_i is free, so with c = mid_i:

    T_LOO = (1/n) sum_r (p_i(s_r) - c) d_r
          = (1/(n-1)) sum_r (p_i(s_r) - c)(p_j(s_r) - pbar_j)
          <= (R_i / (2(n-1))) sum_r |p_j(s_r) - pbar_j|
          <= (R_i / (2(n-1))) * (n/2) R_j            [max sum of abs-devs is (n/2)R_j, split-half]
          =  (n/(n-1)) R_i R_j / 4.

So E[X|F_{m-1}]_LOO = T_LOO - delta_LOO <= 0 for every n, i.e. the slack is valid; it is
EXACT (attained) for n even by the comonotone split-half construction and slightly slack
for n odd. At n=2, q_r is just the single other item, so LOO == cyclic and delta_LOO =
(2/1) R_iR_j/4 = (1/2) R_iR_j, consistent with the cyclic bound.

## 4.5 From the drift bound to an anytime-valid e-process (factor / supermartingale / Ville)

Sections 1-4 bound the per-cell conditional drift. This section closes the loop: that
bound makes the wealth process a non-negative test supermartingale, hence Ville gives an
anytime-valid test. Per block m, the increment is the vector X_m = (X_{m,k})_k with
X_{m,k} the mean over the (block m, stratum k) cell of inc = U - V - delta_k; the bet is
the affine factor  1 + <lam_m, X_m>  with lam_m >= 0, sum_k lam_m,k <= 1/b, and
b = 1 + max_k delta_k. (lam_m is F_{m-1}-measurable: it depends only on the theta-fixed
stratum/block structure and outcomes of blocks < m.)

(a) POSITIVITY. Each inc = U - V - delta_k with U,V in {0,1} lies in [-(1+delta_k), 1-delta_k],
and a cell mean stays in that interval, so |X_{m,k}| <= 1 + delta_k <= b. Hence
<lam_m, X_m> >= -(sum_k lam_m,k) * max_k|X_{m,k}| >= -(1/b) * b = -1, so 1 + <lam_m, X_m> >= 0.

(b) SUPERMARTINGALE under H0. Since lam_m is F_{m-1}-measurable,
E[1 + <lam_m, X_m> | F_{m-1}] = 1 + <lam_m, E[X_m | F_{m-1}]>. By Sections 1-3,
E[X_{m,k} | F_{m-1}] = T - delta_k <= (1/2) R_i R_j - delta_2 = 0 (and <= 0 under delta_LOO),
coordinatewise, under arbitrary cross-stratum dependence (the fresh-pair map keeps both
partners out of F_{m-1}). As lam_m >= 0, <lam_m, E[X_m|F_{m-1}]> <= 0, so
E[M_m | F_{m-1}] <= M_{m-1}: the wealth M_t = prod_{m<=t}(1 + <lam_m, X_m>) is a
non-negative supermartingale with M_0 = 1.

(c) VILLE. For a non-negative supermartingale with E[M_0] = 1, Ville's inequality gives
P(sup_t M_t >= 1/alpha) <= alpha. Rejecting when M_t first crosses 1/alpha is therefore an
anytime-valid level-alpha test.

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
